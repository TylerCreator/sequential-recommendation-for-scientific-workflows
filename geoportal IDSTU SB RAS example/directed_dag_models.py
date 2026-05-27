#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Directed DAG Sequence Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Скрипт сравнивает несколько моделей, которые продолжают последовательность
в Directed Acyclic Graph, строго учитывая направленную структуру:

1. Popularity baseline
2. DirectedDAGNN  (APPNP-style propagation c направленными весами)
3. DA-GCN (Zhu et al., ACM TOIS 2024) - персонализированные направленные графы с attentive aggregation
4. DeepDAG (2022) (depth-aware attention)
5. DAG-GNN (Yu et al., 2019 адаптация) с обучаемыми весами на рёбрах
6. SR-GNN (Wu et al., AAAI 2019) - session-based GNN
7. DAGNN2021 (Thost & Chen) - топологическая обработка с attention
8. GRU4Rec (маскирует выходы по направлению графа)

Usage (пример):
    python directed_dag_models.py --data compositionsDAG.json --epochs 150
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, ndcg_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("directed_dag_models")

# ---------------------------------------------------------------------------
# Data utilities (match sequence_dag_recommender_final.py)
# ---------------------------------------------------------------------------


def load_dag_from_json(json_path: Path) -> Tuple[nx.DiGraph, List]:
    logger.info(f"Loading DAG from {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dag = nx.DiGraph()
    for composition in data:
        id_to_mid = {}
        for node in composition["nodes"]:
            node_id = str(node["id"])
            if "mid" in node:
                id_to_mid[node_id] = f"service_{node['mid']}"
            else:
                id_to_mid[node_id] = f"table_{node['id']}"

        for link in composition["links"]:
            source = str(link["source"])
            target = str(link["target"])
            if source not in id_to_mid or target not in id_to_mid:
                continue
            src_node = id_to_mid[source]
            tgt_node = id_to_mid[target]
            dag.add_node(src_node, type='service' if src_node.startswith("service") else 'table')
            dag.add_node(tgt_node, type='service' if tgt_node.startswith("service") else 'table')
            dag.add_edge(src_node, tgt_node)

    logger.info(f"Loaded DAG with {dag.number_of_nodes()} nodes and {dag.number_of_edges()} edges")
    return dag, data


def extract_paths_from_compositions(data: List[dict]) -> List[Tuple[List[str], int]]:
    logger.info("Extracting REAL paths from compositions (not synthetic DFS paths)")
    all_paths = []

    for comp_idx, entry in enumerate(data):
        composition = entry["composition"] if "composition" in entry else entry
        comp_graph = nx.DiGraph()
        id_to_mid = {}
        for node in composition["nodes"]:
            node_id = str(node["id"])
            if "mid" in node:
                node_name = f"service_{node['mid']}"
            else:
                node_name = f"table_{node['id']}"
            id_to_mid[node_id] = node_name

        for link in composition["links"]:
            source = str(link["source"])
            target = str(link["target"])
            if source in id_to_mid and target in id_to_mid:
                comp_graph.add_edge(id_to_mid[source], id_to_mid[target])

        start_nodes = [n for n in comp_graph.nodes() if comp_graph.in_degree(n) == 0]
        end_nodes = [n for n in comp_graph.nodes() if comp_graph.out_degree(n) == 0]
        for start in start_nodes:
            for end in end_nodes:
                try:
                    for path in nx.all_simple_paths(comp_graph, start, end):
                        if len(path) > 1:
                            all_paths.append((path, comp_idx))
                except nx.NetworkXNoPath:
                    continue

    logger.info(f"Extracted {len(all_paths)} REAL paths from {len(data)} compositions")
    return all_paths


def create_training_pairs(paths_with_idx: List[Tuple[List[str], int]]) -> Tuple[List[Tuple[str, ...]], List[str], List[int]]:
    X, y, comp_ids = [], [], []
    for path, comp_idx in paths_with_idx:
        for idx in range(1, len(path)):
            context = tuple(path[:idx])
            target = path[idx]
            if target.startswith("service"):
                X.append(context)
                y.append(target)
                comp_ids.append(comp_idx)
    return X, y, comp_ids


def build_graph(paths: List[List[str]]) -> nx.DiGraph:
    """Build graph with edge weights based on transition frequency in compositions."""
    g = nx.DiGraph()
    edge_counts = defaultdict(int)
    nodes = set()
    
    # Count how many times each edge appears in paths
    for path in paths:
        nodes.update(path)
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            edge_counts[edge] += 1
    
    # Add edges with their frequency weights
    for (u, v), count in edge_counts.items():
        g.add_edge(u, v, weight=count)
    
    if nodes:
        g.add_nodes_from(nodes)
    if edge_counts:
        logger.info(f"Built graph with {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")
        logger.info(f"Edge weight range: {min(edge_counts.values())} - {max(edge_counts.values())}")
    else:
        logger.warning("Built graph with %d nodes but no edges (empty training transitions).", g.number_of_nodes())
    
    return g


def build_composition_graphs(compositions: List[dict]) -> Tuple[List[Data], List[Dict[str, int]]]:
    graphs = []
    node_maps = []
    for entry in compositions:
        composition = entry["composition"] if "composition" in entry else entry
        id_to_mid = {}
        for node in composition["nodes"]:
            node_id = str(node["id"])
            if "mid" in node:
                id_to_mid[node_id] = f"service_{node['mid']}"
            else:
                id_to_mid[node_id] = f"table_{node['id']}"
        node_names = list(dict.fromkeys(id_to_mid.values()))
        node_map = {name: idx for idx, name in enumerate(node_names)}
        features = torch.zeros((len(node_names), 2), dtype=torch.float32)
        for name, idx in node_map.items():
            if name.startswith("service"):
                features[idx, 0] = 1.0
            else:
                features[idx, 1] = 1.0
        edges = []
        for link in composition["links"]:
            source = str(link["source"])
            target = str(link["target"])
            if source in id_to_mid and target in id_to_mid:
                src_name = id_to_mid[source]
                tgt_name = id_to_mid[target]
                edges.append([node_map[src_name], node_map[tgt_name]])
        edge_index = torch.tensor(edges, dtype=torch.long).t() if edges else torch.empty((2, 0), dtype=torch.long)
        graphs.append(Data(x=features, edge_index=edge_index))
        node_maps.append(node_map)
    return graphs, node_maps


def prepare_pyg(graph: nx.DiGraph, nodes: List[str]) -> Tuple[Data, Dict[str, int]]:
    encoder = LabelEncoder()
    node_ids = encoder.fit_transform(nodes)
    node_map = {node: idx for node, idx in zip(nodes, node_ids)}
    
    # Extract edges and their weights
    edges = []
    edge_weights = []
    for u, v, data in graph.edges(data=True):
        edges.append([node_map[u], node_map[v]])
        edge_weights.append(data.get('weight', 1.0))  # Default weight is 1.0
    
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    
    features = torch.zeros((len(nodes), 2), dtype=torch.float32)
    for node, idx in node_map.items():
        if node.startswith("service"):
            features[idx, 0] = 1.0
        else:
            features[idx, 1] = 1.0
    
    data = Data(x=features, edge_index=edge_index, edge_weight=edge_weight)
    return data, node_map


def prepare_sequences(contexts: List[Tuple[str, ...]], targets: List[str],
                      node_map: Dict[str, int], service_map: Dict[str, int],
                      max_len: int = 10) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sequences, lengths, labels = [], [], []
    for ctx, target in zip(contexts, targets):
        idxs = [node_map[n] + 1 for n in ctx]  # padding=0
        if len(idxs) >= max_len:
            idxs = idxs[-max_len:]
        else:
            idxs = [0] * (max_len - len(idxs)) + idxs
        sequences.append(idxs)
        lengths.append(min(len(ctx), max_len))
        labels.append(service_map[target])
    return (torch.tensor(sequences, dtype=torch.long),
            torch.tensor(lengths, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


def build_node_owner_map(compositions: List[dict]) -> Dict[str, str]:
    node_owner: Dict[str, str] = {}
    service_owner: Dict[str, str] = {}
    for entry in compositions:
        composition = entry["composition"] if "composition" in entry else entry
        for node in composition["nodes"]:
            node_id = str(node["id"])
            if "mid" in node:
                node_name = f"service_{node['mid']}"
                owner = node.get("owner", "unknown")
            else:
                node_name = f"table_{node_id}"
                owner = node_owner.get(node_name, "unknown")
            node_owner[node_name] = owner
            if node_name.startswith("service_"):
                service_owner[node_name] = owner

    for entry in compositions:
        composition = entry["composition"] if "composition" in entry else entry
        id_to_name = {}
        for node in composition["nodes"]:
            node_id = str(node["id"])
            if "mid" in node:
                id_to_name[node_id] = f"service_{node['mid']}"
            else:
                id_to_name[node_id] = f"table_{node['id']}"

        for link in composition["links"]:
            source = str(link["source"])
            target = str(link["target"])
            if source not in id_to_name or target not in id_to_name:
                continue
            src_name = id_to_name[source]
            tgt_name = id_to_name[target]
            if src_name.startswith("table_") and tgt_name in service_owner:
                node_owner.setdefault(src_name, service_owner[tgt_name])
            if tgt_name.startswith("table_") and src_name in service_owner:
                node_owner.setdefault(tgt_name, service_owner[src_name])

    for name in list(node_owner.keys()):
        if node_owner[name] is None:
            node_owner[name] = "unknown"
    return node_owner


def split_data(contexts: List[Tuple[str, ...]], targets: List[str], comp_indices: List[int],
               test_size: float, seed: int):
    lb = LabelEncoder().fit(targets)
    y_enc = lb.transform(targets)
    counts = Counter(y_enc)
    min_count = min(counts.values())
    stratify = y_enc if min_count >= 2 else None
    if stratify is None:
        logger.warning("Too few samples for stratified split (min=%d). Using random split.", min_count)
    ctx_train, ctx_test, y_train, y_test, comp_train, comp_test = train_test_split(
        contexts, targets, comp_indices, test_size=test_size, random_state=seed, stratify=stratify
    )
    return ctx_train, ctx_test, y_train, y_test, comp_train, comp_test


def build_global_srgnn_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    base = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    if edge_index.numel() > 0:
        src, dst = edge_index
        for s, d in zip(src.tolist(), dst.tolist()):
            base[s, d] += 1.0
    in_sum = base.sum(0)
    in_sum[in_sum == 0] = 1.0
    A_in = base / in_sum.unsqueeze(0)
    out_sum = base.sum(1)
    out_sum[out_sum == 0] = 1.0
    A_out = base.t() / out_sum.unsqueeze(1)
    return torch.cat([A_in, A_out], dim=1)


def build_srgnn_samples_from_contexts(
        contexts: List[Tuple[str, ...]],
        targets: List[int],
        node_map: Dict[str, int]
) -> List[Tuple[List[int], int]]:
    samples: List[Tuple[List[int], int]] = []
    skipped = 0
    for ctx, target in zip(contexts, targets):
        indices = []
        valid = True
        for node in ctx:
            if node not in node_map:
                valid = False
                break
            indices.append(node_map[node])
        if not valid or not indices:
            skipped += 1
            continue
        samples.append((indices, target))
    if skipped:
        logger.info("SR-GNN skipped %d contexts without known nodes", skipped)
    return samples


# ---------------------------------------------------------------------------
# Loss Functions (from original GRU4Rec)
# ---------------------------------------------------------------------------


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking loss (BPR-max from GRU4Rec).
    BPR-max maximizes the difference for the hardest negative samples.
    Original paper: Hidasi et al., ICLR 2016
    """
    def forward(self, pos_scores, neg_scores):
        """
        Args:
            pos_scores: Scores for positive items (batch,)
            neg_scores: Scores for negative items (batch, n_neg)
        """
        # BPR-max: maximize difference for hardest negatives (max over negatives)
        # For each positive, find the hardest negative (highest score)
        diff = pos_scores.unsqueeze(1) - neg_scores  # (batch, n_neg)
        # BPR-max: take max over negatives (hardest negative)
        max_diff = diff.max(dim=1)[0]  # (batch,)
        # Standard BPR loss: -log(sigmoid(diff))
        loss = -torch.log(torch.sigmoid(max_diff) + 1e-24).mean()
        return loss


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DirectedAPPNPPropagation(nn.Module):
    def __init__(self, K: int, alpha: float = 0.1):
        super().__init__()
        self.K = K
        self.alpha = alpha

    def forward(self, x, edge_index, edge_weight=None, h0=None):
        if h0 is None:
            h0 = x
        row, col = edge_index
        num_nodes = x.size(0)
        
        # Use edge weights for normalization if provided
        if edge_weight is not None:
            # Compute sum of outgoing edge weights for each node
            weight_sum = torch.zeros(num_nodes, dtype=torch.float32, device=x.device)
            weight_sum.index_add_(0, row, edge_weight)
            weight_sum = weight_sum.clamp(min=1.0)
            # Normalize by weight sum (like weighted out-degree)
            edge_norm = edge_weight / weight_sum[row]
        else:
            # Fall back to degree-based normalization
            deg = torch.bincount(row, minlength=num_nodes).float().clamp(min=1.0).to(x.device)
            edge_norm = 1.0 / deg[row]
        
        messages = x[row] * edge_norm.unsqueeze(-1)
        agg = torch.zeros_like(x).index_add(0, col, messages)
        return (1 - self.alpha) * agg + self.alpha * h0


class DirectedDAGNN(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, K: int = 10, dropout: float = 0.4):
        super().__init__()
        self.dropout = dropout
        self.num_propagations = K
        
        # First encoding layer
        self.lin1 = nn.Linear(in_channels, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        
        # Second layer with residual connection (like DAGNNRecommender)
        self.lin2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        
        # Propagation
        self.prop = DirectedAPPNPPropagation(K=K, alpha=0.1)
        att_hidden = max(1, hidden // 2)
        self.layer_attention = nn.Sequential(
            nn.Linear(hidden * 2, att_hidden),
            nn.GELU(),
            nn.Linear(att_hidden, K + 1)
        )

        # Output head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_channels)
        )

    def forward(self, x, edge_index, edge_weight=None):
        # First layer
        x = self.lin1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Residual block (like in DAGNNRecommender)
        identity = x
        x = self.lin2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x + identity  # Residual connection
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Propagation with attention (using edge weights)
        base_features = x
        xs = [x]
        h = x
        for _ in range(self.num_propagations):
            h = self.prop(h, edge_index, edge_weight, h0=base_features)
            h = F.dropout(h, p=self.dropout, training=self.training)
            xs.append(h)
        stacked = torch.stack(xs, dim=-1)
        node_context = torch.cat([xs[0], xs[-1]], dim=-1)
        att_logits = self.layer_attention(node_context)
        att_weights = F.softmax(att_logits, dim=-1).unsqueeze(1)
        fused = (stacked * att_weights).sum(dim=-1)
        
        return self.head(fused)


class DeepDAGBlock(nn.Module):
    def __init__(self, hidden: int, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        assert hidden % heads == 0
        self.gat = torch.nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden)
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x, attn_mask=None):
        # Use attention mask to restrict attention to graph edges only
        h, _ = self.gat(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + h)
        h2 = self.ffn(x)
        return self.norm2(x + h2)


class DeepDAGEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int, depth_emb: int = 16,
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.depth_encoder = nn.Sequential(
            nn.Linear(1, depth_emb),
            nn.GELU(),
            nn.Linear(depth_emb, depth_emb),
        )
        self.input_proj = nn.Linear(in_channels + depth_emb, hidden)
        self.blocks = nn.ModuleList([DeepDAGBlock(hidden, heads, dropout) for _ in range(num_layers)])

    def forward(self, x, depth, attn_mask=None):
        depth_feat = self.depth_encoder(depth.unsqueeze(-1))
        h = self.input_proj(torch.cat([x, depth_feat], dim=-1))
        h = h.unsqueeze(0)  # treat nodes as sequence (1, N, H)
        for block in self.blocks:
            h = block(h, attn_mask)
        return h.squeeze(0)


class DeepDAGRecommender(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 num_layers: int = 3, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder = DeepDAGEncoder(in_channels, hidden, num_layers=num_layers, heads=heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.LayerNorm(hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_channels)
        )

    def create_attention_mask(self, edge_index, num_nodes, device):
        """
        Create attention mask from edge_index.
        Mask allows attention only along graph edges (and self-loops).
        Returns mask of shape (num_nodes, num_nodes) where:
        - 0.0 = allowed attention
        - -inf = blocked attention
        """
        # Initialize mask: block all connections
        mask = torch.full((num_nodes, num_nodes), float('-inf'), device=device)
        
        # Allow self-attention (each node can attend to itself)
        mask.fill_diagonal_(0.0)
        
        # Allow attention along edges (src -> dst)
        if edge_index.numel() > 0:
            src, dst = edge_index
            mask[dst, src] = 0.0  # dst can attend to src
        
        return mask

    def forward(self, x, edge_index):
        depth = compute_normalized_depth(edge_index, x.size(0), x.device)
        # Create attention mask to restrict attention to graph structure
        attn_mask = self.create_attention_mask(edge_index, x.size(0), x.device)
        h = self.encoder(x, depth, attn_mask)
        return self.head(h)


class SRGNNGNN(nn.Module):
    """
    Original SR-GNN propagation block (Wu et al., AAAI 2019).
    """

    def __init__(self, hidden: int, step: int = 1):
        super().__init__()
        self.hidden = hidden
        self.step = step
        self.input_size = hidden * 2
        self.gate_size = hidden * 3
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size))
        self.w_hh = Parameter(torch.Tensor(self.gate_size, hidden))
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(hidden))
        self.b_oah = Parameter(torch.Tensor(hidden))

        self.linear_edge_in = nn.Linear(hidden, hidden, bias=True)
        self.linear_edge_out = nn.Linear(hidden, hidden, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def gnn_cell(self, A: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        seq_len = A.size(1)
        input_in = torch.matmul(A[:, :, :seq_len], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, seq_len:], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], dim=2)
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        i_r, i_i, i_n = gi.chunk(3, dim=2)
        h_r, h_i, h_n = gh.chunk(3, dim=2)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + resetgate * h_n)
        hy = newgate + inputgate * (hidden - newgate)
        return hy

    def forward(self, A: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        for _ in range(self.step):
            hidden = self.gnn_cell(A, hidden)
        return hidden


class SRGNNRecommender(nn.Module):
    """
    SessionGraph implementation from SR-GNN (Wu et al., AAAI 2019).
    """

    def __init__(self, num_nodes: int, hidden: int, step: int = 1, non_hybrid: bool = False):
        super().__init__()
        self.hidden = hidden
        self.non_hybrid = non_hybrid
        self.embedding = nn.Embedding(num_nodes + 1, hidden)
        self.gnn = SRGNNGNN(hidden, step=step)
        self.linear_one = nn.Linear(hidden, hidden, bias=True)
        self.linear_two = nn.Linear(hidden, hidden, bias=True)
        self.linear_three = nn.Linear(hidden, 1, bias=False)
        self.linear_transform = nn.Linear(hidden * 2, hidden, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, items: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(items)
        hidden = self.gnn(A, hidden)
        return hidden

    def gather_sequence(self, hidden: torch.Tensor, alias_inputs: torch.Tensor) -> torch.Tensor:
        batch_indices = torch.arange(alias_inputs.size(0), device=alias_inputs.device).unsqueeze(-1)
        return hidden[batch_indices, alias_inputs]

    def compute_scores(self,
                       seq_hidden: torch.Tensor,
                       mask: torch.Tensor,
                       candidate_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = mask.sum(dim=1).long()
        last_hidden = seq_hidden[torch.arange(seq_hidden.size(0), device=seq_hidden.device), seq_len - 1]
        q1 = self.linear_one(last_hidden).unsqueeze(1)
        q2 = self.linear_two(seq_hidden)
        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        mask_expanded = mask.unsqueeze(-1).float()
        session_rep = torch.sum(alpha * seq_hidden * mask_expanded, dim=1)
        if not self.non_hybrid:
            session_rep = self.linear_transform(torch.cat([session_rep, last_hidden], dim=1))
        candidates = self.embedding.weight[1:]
        scores = torch.matmul(session_rep, candidates.t())
        if candidate_indices is not None:
            scores = scores.index_select(dim=1, index=candidate_indices - 1)
        return scores


def compute_normalized_depth(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    graph = [[] for _ in range(num_nodes)]
    indeg = [0] * num_nodes
    row, col = edge_index
    for u, v in zip(row.tolist(), col.tolist()):
        graph[u].append(v)
        indeg[v] += 1
    depth = [0] * num_nodes
    queue = [i for i in range(num_nodes) if indeg[i] == 0]
    for node in queue:
        for child in graph[node]:
            depth[child] = max(depth[child], depth[node] + 1)
            indeg[child] -= 1
            if indeg[child] == 0:
                queue.append(child)
    depth_tensor = torch.tensor(depth, dtype=torch.float32, device=device)
    max_depth = depth_tensor.max().clamp(min=1.0)
    return depth_tensor / max_depth


class DAGGNNLayer(nn.Module):
    """
    Реализация DAG-GNN (Yu et al., 2019) с фиксированной структурой графа:
    - отдельные матрицы для сообщений от родителей и детей;
    - аддитивный self-term и LayerNorm.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.parent_weight = nn.Parameter(torch.randn(hidden, hidden) * 0.02)
        self.child_weight = nn.Parameter(torch.randn(hidden, hidden) * 0.02)
        self.self_lin = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.act = nn.GELU()

    def forward(self, h, edge_index, rev_edge_index, dropout: float, training: bool):
        parent_msg = self.aggregate(h, edge_index, self.parent_weight)
        child_msg = self.aggregate(h, rev_edge_index, self.child_weight)
        out = self.self_lin(h) + parent_msg + child_msg
        out = self.act(out)
        out = F.dropout(out, p=dropout, training=training)
        return self.norm(h + out)

    @staticmethod
    def aggregate(h, edge_index, weight):
        if edge_index.numel() == 0:
            return torch.zeros_like(h)
        src, dst = edge_index
        transformed = h[src] @ weight
        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, transformed)
        deg = torch.bincount(dst, minlength=h.size(0)).clamp(min=1).unsqueeze(-1).to(h.device)
        return agg / deg


class DAGGNNRecommender(nn.Module):
    def __init__(self, in_channels: int, hidden: int, out_channels: int, edge_index: torch.Tensor,
                 num_layers: int = 3, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden)
        self.layers = nn.ModuleList([DAGGNNLayer(hidden) for _ in range(num_layers)])
        self.dropout = dropout
        self.edge_index = edge_index
        self.rev_edge_index = edge_index.flip(0)
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels)
        )

    def forward(self, x, edge_index=None):
        if edge_index is None:
            edge_index = self.edge_index
            rev_edge_index = self.rev_edge_index
        else:
            rev_edge_index = edge_index.flip(0)
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, rev_edge_index, self.dropout, self.training)
        return self.head(h)


class DAGNN2021Encoder(nn.Module):
    """
    Shared backbone for DAGNN2021-style models.
    Produces node representations that concatenate states from all propagation layers.
    """

    def __init__(self, in_channels: int, hidden: int, num_layers: int = 3,
                 num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.hidden = hidden
        self.output_dim = hidden * (num_layers + 1)

        self.input_proj = nn.Linear(in_channels, hidden)
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.combine_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(num_layers)
        ])

    def topological_sort(self, edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
        graph = [[] for _ in range(num_nodes)]
        indeg = [0] * num_nodes

        if edge_index.numel() > 0:
            src = edge_index[0].detach().cpu().tolist()
            dst = edge_index[1].detach().cpu().tolist()
            for u, v in zip(src, dst):
                graph[u].append(v)
                indeg[v] += 1

        queue = [i for i in range(num_nodes) if indeg[i] == 0]
        topo_order = []

        while queue:
            current_level = queue[:]
            queue = []
            topo_order.append(current_level)

            for node in current_level:
                for child in graph[node]:
                    indeg[child] -= 1
                    if indeg[child] == 0:
                        queue.append(child)

        return topo_order

    @staticmethod
    def get_predecessors(node: int, edge_index: torch.Tensor) -> List[int]:
        if edge_index.numel() == 0:
            return []
        src, dst = edge_index
        mask = dst == node
        return src[mask].tolist()

    @staticmethod
    def get_terminal_nodes(edge_index: torch.Tensor, num_nodes: int) -> List[int]:
        if edge_index.numel() == 0:
            return list(range(num_nodes))

        src = edge_index[0]
        has_successor = torch.zeros(num_nodes, dtype=torch.bool, device=src.device)
        if src.numel() > 0:
            has_successor[src.unique()] = True

        terminal = [i for i in range(num_nodes) if not has_successor[i].item()]
        return terminal if terminal else list(range(num_nodes))

    def compute_layer_states(self, x: torch.Tensor, edge_index: torch.Tensor) -> List[torch.Tensor]:
        num_nodes = x.size(0)
        h_layers = [self.input_proj(x)]
        topo_levels = self.topological_sort(edge_index, num_nodes)

        for layer_idx in range(self.num_layers):
            prev_h = h_layers[layer_idx]
            h_new = prev_h.clone()

            for level in topo_levels:
                for node in level:
                    preds = self.get_predecessors(node, edge_index)
                    if not preds:
                        continue

                    pred_features = h_new[preds].unsqueeze(0)
                    query = prev_h[node].unsqueeze(0).unsqueeze(0)
                    aggregated, _ = self.attention_layers[layer_idx](
                        query, pred_features, pred_features
                    )
                    aggregated = aggregated.squeeze(0).squeeze(0)
                    combined_input = torch.cat([prev_h[node], aggregated])
                    h_new[node] = self.combine_layers[layer_idx](combined_input)

            h_layers.append(h_new)

        return h_layers

    def build_node_representations(self, h_layers: List[torch.Tensor]) -> torch.Tensor:
        num_nodes = h_layers[0].size(0)
        node_reprs = []
        for node in range(num_nodes):
            node_repr = torch.cat([layer[node] for layer in h_layers], dim=0)
            node_reprs.append(node_repr)
        return torch.stack(node_reprs, dim=0)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h_layers = self.compute_layer_states(x, edge_index)
        return self.build_node_representations(h_layers)


class DAGNN2021(nn.Module):
    """
    Node-level DAGNN classifier (Thost & Chen, 2021) that keeps track of per-node embeddings.
    """

    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.encoder = DAGNN2021Encoder(
            in_channels=in_channels,
            hidden=hidden,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        self.readout = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        node_reprs = self.encoder(x, edge_index)
        return self.readout(node_reprs)


class DAGCNLayer(nn.Module):
    """
    DA-GCN Layer from "Multi-Behavior Recommendation with Personalized Directed 
    Acyclic Behavior Graphs" (ACM TOIS 2024).
    
    Features:
    - Directed edge encoding with separate weights for each edge type
    - Attentive aggregation from predecessor nodes
    - Layer normalization and residual connections
    """
    def __init__(self, hidden: int, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.hidden = hidden
        self.num_heads = num_heads
        assert hidden % num_heads == 0, "hidden must be divisible by num_heads"
        self.head_dim = hidden // num_heads
        
        # Directed edge encoder - separate transformations for source and target
        self.edge_src_transform = nn.Linear(hidden, hidden)
        self.edge_tgt_transform = nn.Linear(hidden, hidden)
        
        # Multi-head attention for aggregation from predecessors
        self.attention = nn.MultiheadAttention(hidden, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, hidden)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = dropout
        
    def encode_edges(self, h, edge_index, edge_weight=None):
        """
        Encode directed edges with GCN-based approach.
        Each edge embedding is a function of source and target node features.
        """
        if edge_index.numel() == 0:
            return h
        
        src, dst = edge_index
        
        # Transform source and target node features
        h_src = self.edge_src_transform(h[src])  # (num_edges, hidden)
        h_tgt = self.edge_tgt_transform(h[dst])  # (num_edges, hidden)
        
        # Edge embeddings combine source and target
        edge_emb = h_src + h_tgt  # (num_edges, hidden)
        
        # Apply edge weights if provided
        if edge_weight is not None:
            edge_emb = edge_emb * edge_weight.unsqueeze(-1)
        
        # Aggregate edge embeddings to target nodes
        num_nodes = h.size(0)
        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, dst, edge_emb)
        
        # Normalize by in-degree
        in_degree = torch.bincount(dst, minlength=num_nodes).clamp(min=1).float().unsqueeze(-1).to(h.device)
        aggregated = aggregated / in_degree
        
        return aggregated
    
    def forward(self, h, edge_index, edge_weight=None):
        """
        Forward pass with attentive aggregation from predecessors.
        
        Args:
            h: Node features (num_nodes, hidden)
            edge_index: Edge connectivity (2, num_edges)
            edge_weight: Optional edge weights (num_edges,)
        """
        # Edge encoding and aggregation
        edge_aggregated = self.encode_edges(h, edge_index, edge_weight)
        
        # Self-attention on aggregated features (treats nodes as sequence)
        h_expanded = h.unsqueeze(0)  # (1, num_nodes, hidden)
        edge_expanded = edge_aggregated.unsqueeze(0)  # (1, num_nodes, hidden)
        
        attn_out, _ = self.attention(h_expanded, edge_expanded, edge_expanded)
        attn_out = attn_out.squeeze(0)  # (num_nodes, hidden)
        
        # Residual connection and normalization
        h = self.norm1(h + F.dropout(attn_out, p=self.dropout, training=self.training))
        
        # Feed-forward network
        ffn_out = self.ffn(h)
        h = self.norm2(h + F.dropout(ffn_out, p=self.dropout, training=self.training))
        
        return h


class DAGCNRecommender(nn.Module):
    """
    DA-GCN (Directed Acyclic Graph Convolutional Network) for sequence recommendation.
    
    Based on "Multi-Behavior Recommendation with Personalized Directed Acyclic 
    Behavior Graphs" (Zhu et al., ACM TOIS 2024).
    
    Architecture:
    - Input projection layer
    - Multiple DA-GCN layers with directed edge encoding
    - Attentive aggregation from predecessor behaviors
    - Output classification head
    """
    def __init__(self, in_channels: int, hidden: int, out_channels: int,
                 num_layers: int = 3, num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # DA-GCN layers
        self.dagcn_layers = nn.ModuleList([
            DAGCNLayer(hidden, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer-wise attention for combining representations from all layers
        self.layer_attention = nn.Sequential(
            nn.Linear(hidden * (num_layers + 1), hidden),
            nn.GELU(),
            nn.Linear(hidden, num_layers + 1),
            nn.Softmax(dim=-1)
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_channels)
        )
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass through DA-GCN.
        
        Args:
            x: Input node features (num_nodes, in_channels)
            edge_index: Edge connectivity (2, num_edges)
            edge_weight: Optional edge weights (num_edges,)
            
        Returns:
            logits: Output predictions (num_nodes, out_channels)
        """
        # Input projection
        h = self.input_proj(x)
        
        # Store representations from all layers
        layer_outputs = [h]
        
        # Apply DA-GCN layers
        for layer in self.dagcn_layers:
            h = layer(h, edge_index, edge_weight)
            layer_outputs.append(h)
        
        # Combine all layer outputs with learnable attention
        stacked = torch.stack(layer_outputs, dim=-1)  # (num_nodes, hidden, num_layers+1)
        pooled = stacked.mean(dim=1)  # (num_nodes, num_layers+1)
        
        # Compute attention weights
        att_weights = self.layer_attention(
            torch.cat(layer_outputs, dim=-1)
        ).unsqueeze(1)  # (num_nodes, 1, num_layers+1)
        
        # Weighted combination of layer outputs
        h_combined = (stacked * att_weights).sum(dim=-1)  # (num_nodes, hidden)
        
        # Output predictions
        logits = self.output_head(h_combined)
        
        return logits


class GRU4Rec(nn.Module):
    """
    GRU4Rec with techniques from the original ICLR 2016 paper.
    Enhanced with:
    - Separate dropout for embeddings and hidden layers
    - Support for BPR-max and TOP1-max losses
    - DAG structure awareness through output masking
    """
    def __init__(self, num_nodes: int, num_services: int, embedding_dim: int = 64, hidden: int = 128, num_layers: int = 2,
                 dropout_embed: float = 0.25, dropout_hidden: float = 0.4, 
                 dag_successors: Dict[int, List[int]] = None,
                 dag_successor_nodes: Dict[int, List[int]] = None,
                 num_users: int = 0,
                 user_emb_dim: int = None):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes + 1, embedding_dim, padding_idx=0)
        self.dropout_embed = dropout_embed
        self.dropout_hidden = dropout_hidden
        
        # GRU with dropout only between layers (not on output)
        self.gru = nn.GRU(embedding_dim, hidden, num_layers=num_layers, 
                         batch_first=True, dropout=dropout_hidden if num_layers > 1 else 0)
        
        self.user_emb_dim = user_emb_dim or hidden
        self.user_embedding = None
        fc_input_dim = hidden
        if num_users and num_users > 0:
            self.user_embedding = nn.Embedding(num_users, self.user_emb_dim)
            fc_input_dim = hidden + self.user_emb_dim

        self.fc = nn.Linear(fc_input_dim, num_services)
        self.dag_successors = dag_successors or {}
        self.dag_successor_nodes = dag_successor_nodes or {}
        self.num_services = num_services

    def forward(self, sequences, lengths, last_nodes=None, compute_scores=False, user_ids=None):
        """
        Args:
            sequences: (batch, seq_len) node indices
            lengths: (batch,) sequence lengths
            last_nodes: (batch,) last node indices for DAG masking
            compute_scores: If True, return raw scores instead of logits (for BPR/TOP1)
        """
        # Embedding with separate dropout
        emb = self.embedding(sequences)
        emb = F.dropout(emb, p=self.dropout_embed, training=self.training)
        
        # GRU processing
        gru_out, _ = self.gru(emb)
        
        # Additional dropout on GRU output
        gru_out = F.dropout(gru_out, p=self.dropout_hidden, training=self.training)
        
        # Get last hidden state
        last_hidden = gru_out[torch.arange(gru_out.size(0)), lengths - 1]
        
        # Compute logits/scores
        if self.user_embedding is not None and user_ids is not None:
            user_emb = self.user_embedding(user_ids)
            last_hidden = torch.cat([last_hidden, user_emb], dim=1)

        logits = self.fc(last_hidden)
        
        # DAG structure masking
        if last_nodes is not None and not compute_scores:
            mask = torch.zeros_like(logits)
            for idx, node in enumerate(last_nodes.tolist()):
                succ = self.dag_successors.get(node, [])
                if succ:
                    mask[idx] = -1e9
                    mask[idx, succ] = 0.0
            logits = logits + mask
        
        return logits


class PerDAGGRU(nn.Module):
    def __init__(self, graph_in_channels: int, graph_hidden: int, seq_hidden: int,
                 out_channels: int, max_len: int = 10, num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.graph_encoder = DeepDAGEncoder(
            in_channels=graph_in_channels,
            hidden=graph_hidden,
            depth_emb=graph_hidden // 8,
            num_layers=3,
            heads=4,
            dropout=dropout
        )
        self.max_len = max_len
        self.gru = nn.GRU(graph_hidden, seq_hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(seq_hidden, out_channels)

    def encode_graph(self, data: Data):
        depth = compute_normalized_depth(data.edge_index, data.x.size(0), data.x.device)
        return self.graph_encoder(data.x, depth)

    def build_sequence_tensor(self, contexts: List[Tuple[str, ...]], node_map: Dict[str, int],
                              embeddings: torch.Tensor):
        sequences = []
        lengths = []
        kept_indices = []
        hidden_dim = embeddings.size(1)
        zero_vec = torch.zeros(hidden_dim, dtype=embeddings.dtype, device=embeddings.device)
        for idx, ctx in enumerate(contexts):
            emb_list = []
            valid = True
            for node in ctx:
                if node not in node_map:
                    valid = False
                    break
                emb_list.append(embeddings[node_map[node]])
            if not valid or not emb_list:
                continue
            emb_list = emb_list[-self.max_len:]
            lengths.append(len(emb_list))
            if len(emb_list) < self.max_len:
                pad = [zero_vec] * (self.max_len - len(emb_list))
                emb_list = pad + emb_list
            sequences.append(torch.stack(emb_list, dim=0))
            kept_indices.append(idx)
        if not sequences:
            return None, None, None
        seq_tensor = torch.stack(sequences, dim=0)
        len_tensor = torch.tensor(lengths, dtype=torch.long, device=embeddings.device)
        return seq_tensor, len_tensor, kept_indices

    def forward(self, seq_batch, lengths):
        gru_out, _ = self.gru(seq_batch)
        last_hidden = gru_out[torch.arange(gru_out.size(0)), lengths - 1]
        return self.head(last_hidden)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def train_graph_model(model, data_pyg, train_idx, test_idx, targets_train, targets_test,
                      optimizer, epochs: int, name: str):
    import time
    criterion = nn.CrossEntropyLoss()
    logger.info("Training %s ...", name)
    
    # Check if model supports edge_weight (DirectedDAGNN and DAGCNRecommender do)
    edge_weight = data_pyg.edge_weight if hasattr(data_pyg, 'edge_weight') else None
    use_edge_weight = edge_weight is not None and isinstance(model, (DirectedDAGNN, DAGCNRecommender))
    
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        if use_edge_weight:
            logits = model(data_pyg.x, data_pyg.edge_index, edge_weight)[train_idx]
        else:
            logits = model(data_pyg.x, data_pyg.edge_index)[train_idx]
        loss = criterion(logits, targets_train)
        loss.backward()
        optimizer.step()
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 10 == 0:
            logger.info("%s Epoch %d/%d loss=%.4f time=%.2fs", name, epoch + 1, epochs, loss.item(), epoch_time)
    model.eval()
    with torch.no_grad():
        if use_edge_weight:
            logits = model(data_pyg.x, data_pyg.edge_index, edge_weight)[test_idx]
        else:
            logits = model(data_pyg.x, data_pyg.edge_index)[test_idx]
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)
    return compute_metrics(preds.numpy(), targets_test.numpy(), probs.numpy(), name)


def train_deepdag_per_composition(model, comp_graphs, comp_node_maps,
                                  contexts_train, contexts_test,
                                  comp_indices_train, comp_indices_test,
                                  targets_train_indices, targets_test_indices,
                                  optimizer, epochs: int, name: str):
    criterion = nn.CrossEntropyLoss()
    comp_to_train = defaultdict(list)
    for idx, comp_idx in enumerate(comp_indices_train):
        comp_to_train[comp_idx].append(idx)

    # Pre-compute depth for all compositions to avoid recomputing every forward pass
    logger.info("Pre-computing depths for %d compositions...", len(comp_graphs))
    comp_depths = []
    for data in comp_graphs:
        depth = compute_normalized_depth(data.edge_index, data.x.size(0), data.x.device)
        comp_depths.append(depth)

    logger.info("Training %s (per-composition) with %d compositions...", name, len(comp_to_train))
    import time
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_samples = 0
        for comp_idx, sample_indices in comp_to_train.items():
            data = comp_graphs[comp_idx]
            # Use pre-computed depth instead of recomputing
            depth = comp_depths[comp_idx]
            # Forward pass with cached depth
            h = model.encoder(data.x, depth)
            logits = model.head(h)
            
            node_ids, label_ids = [], []
            node_map = comp_node_maps[comp_idx]
            for sample_idx in sample_indices:
                node_name = contexts_train[sample_idx][-1]
                if node_name not in node_map:
                    continue
                node_ids.append(node_map[node_name])
                label_ids.append(targets_train_indices[sample_idx])
            if not node_ids:
                continue
            node_tensor = torch.tensor(node_ids, dtype=torch.long)
            label_tensor = torch.tensor(label_ids, dtype=torch.long)
            batch_logits = logits[node_tensor]
            loss = criterion(batch_logits, label_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(node_ids)
            total_samples += len(node_ids)
        if total_samples == 0:
            logger.warning("%s training skipped (no samples).", name)
            break
        epoch_time = time.time() - epoch_start
        # Log every 10 epochs instead of every 50 to show progress
        if (epoch + 1) % 10 == 0:
            logger.info("%s Epoch %d/%d loss=%.4f time=%.2fs", name, epoch + 1, epochs, total_loss / total_samples, epoch_time)

    model.eval()
    preds_list, probs_list, labels_list = [], [], []
    with torch.no_grad():
        for idx, comp_idx in enumerate(comp_indices_test):
            data = comp_graphs[comp_idx]
            # Use pre-computed depth for evaluation as well
            depth = comp_depths[comp_idx]
            h = model.encoder(data.x, depth)
            logits = model.head(h)
            
            node_map = comp_node_maps[comp_idx]
            node_name = contexts_test[idx][-1]
            if node_name not in node_map:
                continue
            node_id = node_map[node_name]
            logit = logits[node_id].unsqueeze(0)
            prob = F.softmax(logit, dim=1)
            preds_list.append(logit.argmax(dim=1).cpu().numpy())
            probs_list.append(prob.cpu().numpy())
            labels_list.append(targets_test_indices[idx])

    if not preds_list:
        raise RuntimeError(f"No valid test samples for {name}")

    preds = np.concatenate(preds_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    labels = np.array(labels_list)
    metrics = compute_metrics(preds, labels, probs, name)
    return metrics


def train_srgnn_per_composition(model, comp_graphs, comp_node_maps,
                                contexts_train, contexts_test,
                                comp_indices_train, comp_indices_test,
                                targets_train_indices, targets_test_indices,
                                optimizer, epochs: int, name: str,
                                device: torch.device):
    import time
    criterion = nn.CrossEntropyLoss()
    comp_to_train = defaultdict(list)
    for idx, comp_idx in enumerate(comp_indices_train):
        comp_to_train[comp_idx].append(idx)

    logger.info("Training %s (per-composition) with %d compositions...", name, len(comp_to_train))
    model.to(device)

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_samples = 0
        for comp_idx, sample_indices in comp_to_train.items():
            data = comp_graphs[comp_idx]
            if data.x.numel() == 0:
                continue
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            node_map = comp_node_maps[comp_idx]
            node_states = model.forward(x, edge_index)

            logits_list = []
            label_list = []
            for sample_idx in sample_indices:
                context_nodes = contexts_train[sample_idx]
                mapped = [node_map.get(node) for node in context_nodes if node in node_map]
                if not mapped:
                    continue
                context_idx = torch.tensor(mapped, dtype=torch.long, device=device)
                last_idx = context_idx[-1].item()
                logits = model.session_logits(node_states, context_idx, last_idx)
                logits_list.append(logits)
                label_list.append(targets_train_indices[sample_idx])

            if not logits_list:
                continue
            batch_logits = torch.cat(logits_list, dim=0)
            label_tensor = torch.tensor(label_list, dtype=torch.long, device=device)
            loss = criterion(batch_logits, label_tensor)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(logits_list)
            total_samples += len(logits_list)
        if total_samples == 0:
            logger.warning("%s training skipped (no samples).", name)
            break
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 10 == 0:
            logger.info("%s Epoch %d/%d loss=%.4f time=%.2fs", name, epoch + 1, epochs, total_loss / total_samples, epoch_time)

    model.eval()
    preds_list, probs_list, labels_list = [], [], []
    with torch.no_grad():
        for idx, comp_idx in enumerate(comp_indices_test):
            data = comp_graphs[comp_idx]
            if data.x.numel() == 0:
                continue
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            node_map = comp_node_maps[comp_idx]
            node_states = model.forward(x, edge_index)
            context_nodes = contexts_test[idx]
            mapped = [node_map.get(node) for node in context_nodes if node in node_map]
            if not mapped:
                continue
            context_idx = torch.tensor(mapped, dtype=torch.long, device=device)
            last_idx = context_idx[-1].item()
            logit = model.session_logits(node_states, context_idx, last_idx)
            prob = F.softmax(logit, dim=1)
            preds_list.append(logit.argmax(dim=1).cpu().numpy())
            probs_list.append(prob.cpu().numpy())
            labels_list.append(targets_test_indices[idx])

    if not preds_list:
        raise RuntimeError(f"No valid test samples for {name}")

    preds = np.concatenate(preds_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    labels = np.array(labels_list)
    return compute_metrics(preds, labels, probs, name)


def train_srgnn_global_graph(model: SRGNNRecommender,
                             node_map: Dict[str, int],
                             data_pyg: Data,
                             contexts_train: List[Tuple[str, ...]],
                             contexts_test: List[Tuple[str, ...]],
                             train_targets: List[int],
                             test_targets: List[int],
                             service_indices: torch.Tensor,
                             optimizer,
                             epochs: int,
                             name: str,
                             device: torch.device,
                             batch_size: int = 512):
    """
    Train SR-GNN on the same global graph that используется другими моделями.
    """
    import time

    train_samples = build_srgnn_samples_from_contexts(contexts_train, train_targets, node_map)
    test_samples = build_srgnn_samples_from_contexts(contexts_test, test_targets, node_map)
    if not train_samples or not test_samples:
        raise RuntimeError("Insufficient SR-GNN samples for training/testing")

    criterion = nn.CrossEntropyLoss()
    model.to(device)
    service_indices = service_indices.to(device)

    num_nodes = len(node_map)
    items_tensor = (torch.arange(num_nodes, dtype=torch.long, device=device) + 1).unsqueeze(0)
    global_adj = build_global_srgnn_adj(data_pyg.edge_index, num_nodes).to(device).unsqueeze(0)

    def run_epoch(samples: List[Tuple[List[int], int]], training: bool):
        if not samples:
            return None, None

        if training:
            total_loss = 0.0
            total_count = 0
        else:
            pred_chunks, prob_chunks, label_chunks = [], [], []

        for start in range(0, len(samples), batch_size):
            sample_batch = samples[start:start + batch_size]
            max_len = max(len(seq_indices) for seq_indices, _ in sample_batch)
            alias_tensor = torch.zeros((len(sample_batch), max_len), dtype=torch.long, device=device)
            mask_tensor = torch.zeros((len(sample_batch), max_len), dtype=torch.float32, device=device)
            labels = []

            for row_idx, (seq_indices, label) in enumerate(sample_batch):
                seq_len = len(seq_indices)
                alias_tensor[row_idx, :seq_len] = torch.tensor(seq_indices, dtype=torch.long, device=device)
                mask_tensor[row_idx, :seq_len] = 1.0
                labels.append(label)

            hidden = model(items_tensor, global_adj)
            hidden_batch = hidden.expand(alias_tensor.size(0), -1, -1)
            seq_hidden = model.gather_sequence(hidden_batch, alias_tensor)
            batch_logits = model.compute_scores(seq_hidden, mask_tensor, service_indices)
            label_tensor = torch.tensor(labels, dtype=torch.long, device=device)

            if training:
                loss = criterion(batch_logits, label_tensor)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * label_tensor.size(0)
                total_count += label_tensor.size(0)
            else:
                probs = F.softmax(batch_logits, dim=1).detach()
                preds = batch_logits.detach().argmax(dim=1).cpu().numpy()
                pred_chunks.append(preds)
                prob_chunks.append(probs.cpu().numpy())
                label_chunks.append(label_tensor.cpu().numpy())

        if training:
            return total_loss / max(total_count, 1), total_count

        preds = np.concatenate(pred_chunks, axis=0)
        probs = np.concatenate(prob_chunks, axis=0)
        labels = np.concatenate(label_chunks, axis=0)
        return (preds, probs, labels)

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        loss_value, sample_count = run_epoch(train_samples, training=True)
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 10 == 0 and loss_value is not None:
            logger.info("%s Epoch %d/%d loss=%.4f time=%.2fs",
                        name, epoch + 1, epochs, loss_value, epoch_time)

    model.eval()
    preds, probs, labels = run_epoch(test_samples, training=False)
    if preds is None:
        raise RuntimeError(f"No valid SR-GNN evaluation samples for {name}")
    return compute_metrics(preds, labels, probs, name)


def train_dagcn_per_composition(model, comp_graphs, comp_node_maps,
                                contexts_train, contexts_test,
                                comp_indices_train, comp_indices_test,
                                targets_train_indices, targets_test_indices,
                                optimizer, epochs: int, name: str):
    """
    Train DA-GCN on per-composition graphs (personalized approach from original paper).
    """
    criterion = nn.CrossEntropyLoss()
    comp_to_train = defaultdict(list)
    for idx, comp_idx in enumerate(comp_indices_train):
        comp_to_train[comp_idx].append(idx)

    logger.info("Training %s (per-composition) with %d compositions...", name, len(comp_to_train))
    import time
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_samples = 0
        for comp_idx, sample_indices in comp_to_train.items():
            data = comp_graphs[comp_idx]
            
            # Forward pass through DA-GCN
            logits = model(data.x, data.edge_index)
            
            node_ids, label_ids = [], []
            node_map = comp_node_maps[comp_idx]
            for sample_idx in sample_indices:
                node_name = contexts_train[sample_idx][-1]
                if node_name not in node_map:
                    continue
                node_ids.append(node_map[node_name])
                label_ids.append(targets_train_indices[sample_idx])
            if not node_ids:
                continue
            node_tensor = torch.tensor(node_ids, dtype=torch.long)
            label_tensor = torch.tensor(label_ids, dtype=torch.long)
            batch_logits = logits[node_tensor]
            loss = criterion(batch_logits, label_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(node_ids)
            total_samples += len(node_ids)
        if total_samples == 0:
            logger.warning("%s training skipped (no samples).", name)
            break
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 10 == 0:
            logger.info("%s Epoch %d/%d loss=%.4f time=%.2fs", name, epoch + 1, epochs, total_loss / total_samples, epoch_time)

    model.eval()
    preds_list, probs_list, labels_list = [], [], []
    with torch.no_grad():
        for idx, comp_idx in enumerate(comp_indices_test):
            data = comp_graphs[comp_idx]
            logits = model(data.x, data.edge_index)
            
            node_map = comp_node_maps[comp_idx]
            node_name = contexts_test[idx][-1]
            if node_name not in node_map:
                continue
            node_id = node_map[node_name]
            logit = logits[node_id].unsqueeze(0)
            prob = F.softmax(logit, dim=1)
            preds_list.append(logit.argmax(dim=1).cpu().numpy())
            probs_list.append(prob.cpu().numpy())
            labels_list.append(targets_test_indices[idx])

    if not preds_list:
        raise RuntimeError(f"No valid test samples for {name}")

    preds = np.concatenate(preds_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    labels = np.array(labels_list)
    metrics = compute_metrics(preds, labels, probs, name)
    return metrics


def sample_negatives(targets, num_classes, n_sample, sample_alpha=0.75, item_popularity=None):
    """
    Sample negative items using popularity-based sampling (like original GRU4Rec).
    
    Args:
        targets: Positive target indices (batch,)
        num_classes: Total number of classes
        n_sample: Number of negative samples per positive
        sample_alpha: Sampling temperature (0=uniform, 1=popularity)
        item_popularity: Popularity counts for each class (num_classes,)
    
    Returns:
        Negative sample indices (batch, n_sample)
    """
    batch_size = targets.size(0)
    
    if item_popularity is None or sample_alpha == 0:
        # Uniform sampling
        neg_samples = torch.randint(0, num_classes, (batch_size, n_sample), device=targets.device)
    else:
        # Popularity-based sampling: prob ~ popularity^sample_alpha
        probs = item_popularity.float() ** sample_alpha
        probs = probs / probs.sum()
        neg_samples = torch.multinomial(probs, batch_size * n_sample, replacement=True)
        neg_samples = neg_samples.view(batch_size, n_sample)
    
    return neg_samples


def train_gru_model(model: GRU4Rec, seq_train, len_train, seq_test, len_test,
                    targets_train, targets_test, last_nodes_train, last_nodes_test,
                    optimizer, epochs: int, loss_type='ce', n_sample=0, sample_alpha=0.75,
                    user_ids_train=None, user_ids_test=None):
    """
    Train GRU4Rec with original techniques.
    
    Args:
        loss_type: 'ce' (cross-entropy) or 'bpr' (BPR-max)
        n_sample: Number of negative samples (0 = use only in-batch negatives)
        sample_alpha: Sampling exponent for popularity-based sampling
    """
    import time
    
    # Initialize loss function
    if loss_type == 'bpr':
        criterion = BPRLoss()
        logger.info("Using BPR-max loss with %d negative samples", n_sample if n_sample > 0 else "in-batch")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Using Cross-Entropy loss")
    
    # Compute item popularity for sampling
    item_popularity = None
    if n_sample > 0 and sample_alpha > 0:
        item_popularity = torch.bincount(targets_train, minlength=model.num_services)
        logger.info("Using popularity-based sampling with alpha=%.2f", sample_alpha)
    
    logger.info("Training GRU4Rec ...")
    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        
        if loss_type == 'bpr' and n_sample > 0:
            # BPR loss with negative sampling
            logits = model(seq_train, len_train, last_nodes_train, compute_scores=True, user_ids=user_ids_train)
            
            # Get positive scores
            pos_scores = logits[torch.arange(logits.size(0)), targets_train]
            
            # Sample negatives
            neg_indices = sample_negatives(targets_train, model.num_services, n_sample, 
                                          sample_alpha, item_popularity)
            neg_scores = logits.gather(1, neg_indices)
            
            loss = criterion(pos_scores, neg_scores)
        else:
            # Cross-entropy loss (standard)
            logits = model(seq_train, len_train, last_nodes_train, user_ids=user_ids_train)
            loss = criterion(logits, targets_train)
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_time = time.time() - epoch_start
        
        if (epoch + 1) % 10 == 0:
            logger.info("GRU4Rec Epoch %d/%d loss=%.4f time=%.2fs", epoch + 1, epochs, loss.item(), epoch_time)
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(seq_test, len_test, last_nodes_test, user_ids=user_ids_test)
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)
    return compute_metrics(preds.numpy(), targets_test.numpy(), probs.numpy(), "GRU4Rec")


def train_per_dag_gru(model: PerDAGGRU, comp_graphs, comp_node_maps,
                      train_samples_by_comp, test_samples_by_comp,
                      optimizer, epochs: int, name: str, device: torch.device):
    import time
    criterion = nn.CrossEntropyLoss()
    logger.info("Training %s with %d compositions...", name, len(train_samples_by_comp))
    model.to(device)

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_count = 0
        for comp_idx, samples in train_samples_by_comp.items():
            if not samples:
                continue
            data = comp_graphs[comp_idx]
            embeddings = model.encode_graph(data).to(device)
            contexts = [ctx for ctx, _ in samples]
            targets = torch.tensor([t for _, t in samples], dtype=torch.long)
            seqs, lens, kept = model.build_sequence_tensor(contexts, comp_node_maps[comp_idx], embeddings)
            if seqs is None:
                continue
            seqs = seqs.to(device)
            lens = lens.to(device)
            target_tensor = targets[kept].to(device)
            optimizer.zero_grad()
            logits = model(seqs, lens)
            loss = criterion(logits, target_tensor)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * target_tensor.size(0)
            total_count += target_tensor.size(0)
        if total_count == 0:
            logger.warning("No samples for %s training.", name)
            break
        epoch_time = time.time() - epoch_start
        if (epoch + 1) % 10 == 0:
            logger.info("%s Epoch %d/%d loss=%.4f time=%.2fs", name, epoch + 1, epochs, total_loss / total_count, epoch_time)

    model.eval()
    preds_list, probs_list, labels_list = [], [], []
    with torch.no_grad():
        for comp_idx, samples in test_samples_by_comp.items():
            if not samples:
                continue
            data = comp_graphs[comp_idx]
            embeddings = model.encode_graph(data).to(device)
            contexts = [ctx for ctx, _ in samples]
            targets = torch.tensor([t for _, t in samples], dtype=torch.long)
            seqs, lens, kept = model.build_sequence_tensor(contexts, comp_node_maps[comp_idx], embeddings)
            if seqs is None:
                continue
            logits = model(seqs.to(device), lens.to(device))
            probs = F.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = targets[kept].numpy()
            preds_list.append(preds)
            probs_list.append(probs)
            labels_list.append(labels)

    if not preds_list:
        raise RuntimeError(f"No valid test samples for {name}")

    preds = np.concatenate(preds_list, axis=0)
    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    return compute_metrics(preds, labels, probs, name)


def compute_metrics(preds, labels, probs, name: str) -> Dict[str, float]:
    ndcg_k = min(10, probs.shape[1])
    metrics = {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "ndcg": ndcg_score(np.eye(probs.shape[1])[labels], probs, k=ndcg_k),
    }
    logger.info("%s metrics: %s", name, metrics)
    return metrics


def popularity_baseline(targets_train, targets_test, num_classes, name="Popularity"):
    counter = Counter(targets_train.numpy())
    top_label = counter.most_common(1)[0][0]
    preds = np.full_like(targets_test.numpy(), top_label)
    probs = np.zeros((len(preds), num_classes))
    probs[:, top_label] = 1.0
    return compute_metrics(preds, targets_test.numpy(), probs, name)


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------


def main(args):
    dag, compositions = load_dag_from_json(Path(args.data))
    paths_with_idx = extract_paths_from_compositions(compositions)
    contexts, targets, comp_indices = create_training_pairs(paths_with_idx)
    
    logger.info(f"Total training pairs created: {len(contexts)}")
    
    ctx_train, ctx_test, y_train, y_test, comp_train_idx, comp_test_idx = split_data(
        contexts, targets, comp_indices, args.test_size, args.seed
    )
    
    logger.info(f"Train set size: {len(ctx_train)} samples ({len(ctx_train)/len(contexts)*100:.1f}%)")
    logger.info(f"Test set size: {len(ctx_test)} samples ({len(ctx_test)/len(contexts)*100:.1f}%)")

    train_comp_set = set(comp_train_idx)
    train_paths_only = [path for path, comp_idx in paths_with_idx if comp_idx in train_comp_set]
    all_nodes = sorted({node for path, _ in paths_with_idx for node in path})
    
    # Build graph (optionally with cycle handling)
    # Note: DAG-Transformer can handle cycles automatically, but preprocessing can help
    # IMPORTANT: Build original graph first for GRU4Rec successors mapping
    # GRU4Rec needs full graph structure for proper DAG masking
    original_graph = build_graph(train_paths_only)
    original_graph.add_nodes_from(all_nodes)
    
    use_cycle_handling = True  # Set to False to use standard build_graph
    if use_cycle_handling:
        try:
            from cycle_handling import build_dag_from_paths, is_dag, detect_cycles
            graph = build_dag_from_paths(train_paths_only, cycle_handling="remove_weakest")
            graph.add_nodes_from(all_nodes)
            if not is_dag(graph):
                cycles = detect_cycles(graph)
                logger.info(f"Graph preprocessing: {len(cycles)} cycles remaining (will be handled by models)")
            logger.info(f"Cycle handling: removed {original_graph.number_of_edges() - graph.number_of_edges()} edges")
        except ImportError:
            logger.warning("cycle_handling module not available, using standard build_graph")
            graph = original_graph
    else:
        graph = original_graph
    
    nodes = sorted(graph.nodes())
    data_pyg, node_map = prepare_pyg(graph, nodes)
    comp_graphs, comp_node_maps = build_composition_graphs(compositions)

    services = sorted({y for y in targets})
    service_map = {svc: idx for idx, svc in enumerate(services)}
    logger.info(f"Number of unique services (classes): {len(services)}")

    node_owner_map = build_node_owner_map(compositions)
    owner_set = sorted(set(node_owner_map.values()) | {"unknown"})
    owner_to_idx = {owner: idx for idx, owner in enumerate(owner_set)}

    def get_owner_idx(node_name: str) -> int:
        owner = node_owner_map.get(node_name, "unknown")
        return owner_to_idx.get(owner, owner_to_idx["unknown"])
    
    targets_tensor = torch.tensor([service_map[y] for y in targets], dtype=torch.long)

    train_idx = torch.tensor([node_map[ctx[-1]] for ctx in ctx_train], dtype=torch.long)
    test_idx = torch.tensor([node_map[ctx[-1]] for ctx in ctx_test], dtype=torch.long)
    targets_train = torch.tensor([service_map[y] for y in y_train], dtype=torch.long)
    targets_test = torch.tensor([service_map[y] for y in y_test], dtype=torch.long)

    # Build successors mapping from ORIGINAL graph (not cycle-handled)
    # GRU4Rec needs full graph structure for proper DAG masking
    # Cycle handling removes edges which makes masking too restrictive
    successors = defaultdict(list)
    successor_nodes = defaultdict(list)
    for u, v in original_graph.edges():
        if u in node_map and v in node_map:  # Ensure nodes are in node_map
            if v.startswith("service") and v in service_map:
                successors[node_map[u]].append(service_map[v])
            successor_nodes[node_map[u]].append(node_map[v])

    seq_all, len_all, labels_all = prepare_sequences(contexts, targets, node_map, service_map, max_len=args.max_len)
    owner_ids_all = np.array([get_owner_idx(ctx[-1]) for ctx in contexts], dtype=np.int64)
    label_counts = Counter(labels_all.numpy())
    min_seq_count = min(label_counts.values())
    stratify_seq = labels_all.numpy() if min_seq_count >= 2 else None
    if stratify_seq is None:
        logger.warning("Too few samples for stratified sequential split (min=%d). Using random split.", min_seq_count)
    seq_train, seq_test, len_train, len_test, lab_train, lab_test, user_seq_train, user_seq_test = train_test_split(
        seq_all, len_all, labels_all, owner_ids_all,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify_seq
    )
    last_nodes_train = torch.tensor([node_map[ctx[-1]] for ctx in ctx_train], dtype=torch.long)
    last_nodes_test = torch.tensor([node_map[ctx[-1]] for ctx in ctx_test], dtype=torch.long)
    user_seq_train = torch.tensor(user_seq_train, dtype=torch.long)
    user_seq_test = torch.tensor(user_seq_test, dtype=torch.long)

    train_samples_by_comp = defaultdict(list)
    test_samples_by_comp = defaultdict(list)
    train_target_indices = [service_map[y] for y in y_train]
    test_target_indices = [service_map[y] for y in y_test]
    for ctx, comp_idx, target_idx in zip(ctx_train, comp_train_idx, train_target_indices):
        train_samples_by_comp[comp_idx].append((ctx, target_idx))
    for ctx, comp_idx, target_idx in zip(ctx_test, comp_test_idx, test_target_indices):
        test_samples_by_comp[comp_idx].append((ctx, target_idx))

    service_node_indices = torch.tensor([node_map[svc] + 1 for svc in services], dtype=torch.long)

    results = {}
    results["Popularity"] = popularity_baseline(targets_train, targets_test, len(service_map))

    directed_dagnn = DirectedDAGNN(in_channels=2, hidden=args.hidden, out_channels=len(service_map),
                                   K=args.K, dropout=args.dropout)
    opt_dagnn = torch.optim.Adam(directed_dagnn.parameters(), lr=args.lr)
    results["DirectedDAGNN"] = train_graph_model(
        directed_dagnn, data_pyg, train_idx, test_idx, targets_train, targets_test, opt_dagnn, args.epochs, "DirectedDAGNN"
    )
    
    # DA-GCN (Zhu et al., ACM TOIS 2024) - Global graph
    dagcn = DAGCNRecommender(in_channels=2, hidden=args.hidden, out_channels=len(service_map),
                             num_layers=3, num_heads=4, dropout=args.dropout)
    opt_dagcn = torch.optim.Adam(dagcn.parameters(), lr=args.lr)
    results["DA-GCN (global)"] = train_graph_model(
        dagcn, data_pyg, train_idx, test_idx, targets_train, targets_test, opt_dagcn, args.epochs, "DA-GCN (global)"
    )
    
    # DA-GCN Per-Composition (personalized approach from original paper)
    # Local DA-GCN обучение отключено ради ускорения сравнения.
    # dagcn_local = DAGCNRecommender(in_channels=2, hidden=args.hidden, out_channels=len(service_map),
    #                                num_layers=3, num_heads=4, dropout=args.dropout)
    # opt_dagcn_local = torch.optim.Adam(dagcn_local.parameters(), lr=args.lr)
    # results["DA-GCN (local)"] = train_dagcn_per_composition(
    #     dagcn_local, comp_graphs, comp_node_maps,
    #     ctx_train, ctx_test,
    #     comp_train_idx, comp_test_idx,
    #     train_target_indices, test_target_indices,
    #     opt_dagcn_local, args.epochs, "DA-GCN (local)"
    # )

    srgnn_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    srgnn_global = SRGNNRecommender(
        num_nodes=len(node_map),
        hidden=args.sr_hidden,
        step=args.sr_steps,
        non_hybrid=args.sr_nonhybrid
    )
    opt_srgnn_global = torch.optim.Adam(srgnn_global.parameters(), lr=args.sr_lr)
    results["SR-GNN"] = train_srgnn_global_graph(
        srgnn_global,
        node_map,
        data_pyg,
        ctx_train,
        ctx_test,
        train_target_indices,
        test_target_indices,
        service_node_indices,
        opt_srgnn_global,
        args.epochs,
        "SR-GNN",
        srgnn_device
    )

    # DeepDAG2022 - now using global graph like other GNN models
    deepdag = DeepDAGRecommender(in_channels=2, hidden=args.hidden * 2, out_channels=len(service_map),
                                 num_layers=3, heads=4, dropout=args.dropout)
    opt_deepdag = torch.optim.Adam(deepdag.parameters(), lr=args.lr * 0.8)
    results["DeepDAG2022"] = train_graph_model(
        deepdag, data_pyg, train_idx, test_idx, targets_train, targets_test, opt_deepdag, args.epochs, "DeepDAG2022"
    )

    daggnn = DAGGNNRecommender(in_channels=2, hidden=args.hidden, out_channels=len(service_map),
                               edge_index=data_pyg.edge_index, num_layers=3, dropout=args.dropout)
    opt_daggnn = torch.optim.Adam(daggnn.parameters(), lr=args.lr * 0.5)
    results["DAG-GNN"] = train_graph_model(
        daggnn, data_pyg, train_idx, test_idx, targets_train, targets_test, opt_daggnn, args.epochs, "DAG-GNN"
    )

    # True DAGNN (2021) from Thost & Chen paper
    dagnn2021 = DAGNN2021(in_channels=2, hidden=args.hidden, out_channels=len(service_map),
                          num_layers=3, num_heads=4, dropout=args.dropout)
    opt_dagnn2021 = torch.optim.Adam(dagnn2021.parameters(), lr=args.lr * 0.5)
    results["DAGNN2021"] = train_graph_model(
        dagnn2021, data_pyg, train_idx, test_idx, targets_train, targets_test,
        opt_dagnn2021, args.epochs, "DAGNN2021"
    )
    
    # GRU4Rec with original techniques
    gru_model = GRU4Rec(
        num_nodes=len(node_map), 
        num_services=len(service_map),
        embedding_dim=64, 
        hidden=args.hidden * 2, 
        num_layers=2,
        dropout_embed=args.dropout_embed,
        dropout_hidden=args.dropout_hidden,
        dag_successors=successors,
        dag_successor_nodes=successor_nodes
    )
    opt_gru = torch.optim.Adam(gru_model.parameters(), lr=args.lr)
    results["GRU4Rec"] = train_gru_model(
        gru_model, seq_train, len_train, seq_test, len_test, lab_train, lab_test,
        last_nodes_train, last_nodes_test, opt_gru, args.epochs,
        loss_type=args.loss, n_sample=args.n_sample, sample_alpha=args.sample_alpha
    )

    # DAG-Transformer: Modern Transformer-based model with DAG-aware attention
    try:
        from dag_transformer_integration import add_dag_transformer_to_main
        dag_transformer_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results["DAG-Transformer"] = add_dag_transformer_to_main(
            ctx_train, ctx_test, y_train, y_test,
            node_map, service_map, data_pyg, args, dag_transformer_device, 
            graph=graph, original_graph=original_graph
        )
    except Exception as e:
        logger.warning(f"Failed to train DAG-Transformer: {e}")

    # PerDAG-GRU - COMMENTED OUT (slow)
    # per_dag_gru = PerDAGGRU(
    #     graph_in_channels=2,
    #     graph_hidden=args.hidden * 2,
    #     seq_hidden=args.hidden * 2,
    #     out_channels=len(service_map),
    #     max_len=args.max_len,
    #     num_layers=2,
    #     dropout=args.dropout
    # )
    # opt_per_dag = torch.optim.Adam(per_dag_gru.parameters(), lr=args.lr * 0.8)
    # results["PerDAG-GRU"] = train_per_dag_gru(
    #     per_dag_gru,
    #     comp_graphs,
    #     comp_node_maps,
    #     train_samples_by_comp,
    #     test_samples_by_comp,
    #     opt_per_dag,
    #     args.epochs,
    #     "PerDAG-GRU",
    #     torch.device("cpu")
    # )

    print("\n=== SUMMARY ===")
    for name, metrics in results.items():
        print(f"{name:15s} | acc={metrics['accuracy']:.4f} | ndcg={metrics['ndcg']:.4f} | f1={metrics['f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Directed DAG sequence models comparison")
    parser.add_argument("--data", type=str, default="compositionsDAG.json")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-len", type=int, default=10)
    parser.add_argument("--K", type=int, default=10, help="Propagation steps for DirectedDAGNN")
    
    # GRU4Rec original techniques
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "bpr"],
                       help="Loss function: ce (cross-entropy), bpr (BPR-max)")
    parser.add_argument("--n-sample", type=int, default=0,
                       help="Number of negative samples (0 = use only in-batch negatives)")
    parser.add_argument("--sample-alpha", type=float, default=0.75,
                       help="Sampling exponent for popularity-based sampling (0=uniform, 1=popularity)")
    parser.add_argument("--dropout-embed", type=float, default=0.25,
                       help="Dropout rate for embeddings")
    parser.add_argument("--dropout-hidden", type=float, default=0.4,
                       help="Dropout rate for hidden layers")
    parser.add_argument("--sr-hidden", type=int, default=128,
                       help="Hidden dimension for SR-GNN embeddings")
    parser.add_argument("--sr-steps", type=int, default=1,
                       help="Number of SR-GNN propagation steps")
    parser.add_argument("--sr-nonhybrid", action="store_true",
                       help="Disable hybrid preference (use session context only)")
    parser.add_argument("--sr-lr", type=float, default=5e-4,
                       help="Learning rate for SR-GNN optimizer")
    
    args = parser.parse_args()
    main(args)


