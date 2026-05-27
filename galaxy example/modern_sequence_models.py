#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modern sequence baselines for workflow recommendation.

This module adds three models that can be compared in the notebook:
- Graph Transformer
- SASRec
- GPT-style decoder

All models share the same train/test split and the same metric function
used by the rest of the project.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data


def prepare_sequence_model_data(
    contexts: List[Tuple[str, ...]],
    targets: List[str],
    node_map: Dict[str, int],
    service_map: Dict[str, int],
    max_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert contexts/targets into padded sequence tensors.

    Notes:
    - `input_ids` use 0 as padding, so real node ids are shifted by +1.
    - `raw_node_ids` keep original node ids and use -1 for padding.
    - Sequences are right-padded so `length - 1` points to the last
      real token for autoregressive models.
    """
    input_ids: List[List[int]] = []
    raw_node_ids: List[List[int]] = []
    lengths: List[int] = []
    labels: List[int] = []
    last_node_ids: List[int] = []

    for ctx, target in zip(contexts, targets):
        if target not in service_map:
            continue

        node_ids = [node_map[node_name] for node_name in ctx if node_name in node_map]
        if not node_ids:
            continue

        if len(node_ids) > max_len:
            node_ids = node_ids[-max_len:]

        seq_len = len(node_ids)
        pad_len = max_len - seq_len

        input_ids.append([node_id + 1 for node_id in node_ids] + ([0] * pad_len))
        raw_node_ids.append(node_ids + ([-1] * pad_len))
        lengths.append(seq_len)
        labels.append(service_map[target])
        last_node_ids.append(node_ids[-1])

    if not input_ids:
        raise ValueError("No valid sequence samples were created.")

    attention_mask = [[token != 0 for token in seq] for seq in input_ids]

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "raw_node_ids": torch.tensor(raw_node_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.bool),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "targets": torch.tensor(labels, dtype=torch.long),
        "last_node_ids": torch.tensor(last_node_ids, dtype=torch.long),
    }


def build_successor_service_map(
    graph: nx.DiGraph,
    node_map: Dict[str, int],
    service_map: Dict[str, int],
) -> Dict[int, List[int]]:
    """Map each node id to valid successor service ids."""
    successors: Dict[int, set] = {}
    for u, v in graph.edges():
        if u in node_map and v in service_map:
            successors.setdefault(node_map[u], set()).add(service_map[v])
    return {node_id: sorted(service_ids) for node_id, service_ids in successors.items()}


def build_graph_relation_lookup(
    graph: nx.DiGraph,
    node_map: Dict[str, int],
    max_distance: int = 4,
) -> Tuple[torch.Tensor, int]:
    """
    Build relation buckets for Graph Transformer.

    Buckets:
    - 0: self
    - 1..max_distance: forward reachable with path length d
    - max_distance+1 .. 2*max_distance: backward reachable with path length d
    - 2*max_distance+1: disconnected or farther than max_distance
    - 2*max_distance+2: padding bucket
    """
    num_nodes = len(node_map)
    far_bucket = 2 * max_distance + 1
    pad_bucket = far_bucket + 1
    lookup = torch.full((num_nodes, num_nodes), far_bucket, dtype=torch.long)

    for node_idx in range(num_nodes):
        lookup[node_idx, node_idx] = 0

    index_to_node = {idx: name for name, idx in node_map.items()}

    for src_idx, src_name in index_to_node.items():
        forward_lengths = nx.single_source_shortest_path_length(graph, src_name, cutoff=max_distance)
        for dst_name, dist in forward_lengths.items():
            dst_idx = node_map.get(dst_name)
            if dst_idx is None:
                continue
            bucket = min(dist, max_distance)
            lookup[src_idx, dst_idx] = bucket
            if src_idx != dst_idx:
                reverse_bucket = max_distance + min(dist, max_distance)
                current = lookup[dst_idx, src_idx].item()
                if current == far_bucket:
                    lookup[dst_idx, src_idx] = reverse_bucket

    return lookup, pad_bucket


def apply_candidate_mask(
    logits: torch.Tensor,
    last_node_ids: Optional[torch.Tensor],
    successor_service_map: Optional[Dict[int, List[int]]],
) -> torch.Tensor:
    """Optionally restrict predictions to observed successor services."""
    if last_node_ids is None or not successor_service_map:
        return logits

    masked_logits = logits.clone()
    for row_idx, node_id in enumerate(last_node_ids.tolist()):
        allowed = successor_service_map.get(node_id)
        if not allowed:
            continue
        row_mask = torch.full_like(masked_logits[row_idx], -1e9)
        row_mask[allowed] = 0.0
        masked_logits[row_idx] = masked_logits[row_idx] + row_mask

    return masked_logits


def stabilize_logits(logits: torch.Tensor) -> torch.Tensor:
    """Clamp non-finite values before loss, softmax and metrics."""
    if torch.isfinite(logits).all():
        return logits
    return torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)


def _build_composition_graph(composition_entry: Dict) -> nx.DiGraph:
    """Build a local composition graph using the same node naming as the notebook."""
    composition = composition_entry.get("composition", composition_entry)
    graph = nx.DiGraph()
    id_to_name: Dict[str, str] = {}

    for node in composition["nodes"]:
        node_id = str(node["id"])
        if "mid" in node:
            node_name = f"service_{node['mid']}"
        else:
            node_name = f"table_{node['id']}"
        id_to_name[node_id] = node_name
        graph.add_node(node_name)

    for link in composition["links"]:
        source = str(link["source"])
        target = str(link["target"])
        if source in id_to_name and target in id_to_name:
            graph.add_edge(id_to_name[source], id_to_name[target])

    return graph


def _safe_topological_order(graph: nx.DiGraph) -> List[str]:
    """Return a deterministic node order, falling back if the graph is not a DAG."""
    if graph.number_of_nodes() == 0:
        return []

    try:
        return list(nx.topological_sort(graph))
    except nx.NetworkXUnfeasible:
        working = graph.copy()
        while True:
            try:
                return list(nx.topological_sort(working))
            except nx.NetworkXUnfeasible:
                cycles = list(nx.simple_cycles(working))
                if not cycles:
                    break
                cycle = cycles[0]
                if len(cycle) >= 2 and working.has_edge(cycle[-1], cycle[0]):
                    working.remove_edge(cycle[-1], cycle[0])
                else:
                    break

        # Final fallback preserves determinism even if cycles remain.
        return sorted(graph.nodes())


def _build_local_edge_index(node_order: List[str], subgraph: nx.DiGraph) -> torch.Tensor:
    """Create local edge_index for a subgraph ordered by `node_order`."""
    local_map = {node_name: idx for idx, node_name in enumerate(node_order)}
    edges = [
        [local_map[u], local_map[v]]
        for u, v in subgraph.edges()
        if u in local_map and v in local_map
    ]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _build_srgnn_adjacency(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Build SR-GNN adjacency tensor from an arbitrary subgraph."""
    base = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    if edge_index.numel() > 0:
        src, dst = edge_index
        for s, d in zip(src.tolist(), dst.tolist()):
            base[s, d] += 1.0

    in_sum = base.sum(0)
    in_sum[in_sum == 0] = 1.0
    out_sum = base.sum(1)
    out_sum[out_sum == 0] = 1.0

    a_in = base / in_sum.unsqueeze(0)
    a_out = base.t() / out_sum.unsqueeze(1)
    return torch.cat([a_in, a_out], dim=1)


def prepare_incremental_graph_model_data(
    compositions: List[Dict],
    composition_indices: List[int],
    node_map: Dict[str, int],
    service_map: Dict[str, int],
) -> List[Data]:
    """
    Build incremental subgraph samples from full compositions.

    The extraction follows the idea from the `rec_system` discussion:
    at each step we keep the executed subgraph intact and duplicate the
    sample for every reachable next service, instead of flattening the
    whole composition into independent linear paths.
    """
    selected_indices = sorted(set(composition_indices))
    samples: List[Data] = []

    for comp_idx in selected_indices:
        if comp_idx < 0 or comp_idx >= len(compositions):
            continue

        comp_graph = _build_composition_graph(compositions[comp_idx])
        if comp_graph.number_of_nodes() == 0:
            continue

        topo_order = _safe_topological_order(comp_graph)
        if not topo_order:
            continue

        generations: List[List[str]] = []
        remaining = set(comp_graph.nodes())
        executed: set[str] = set()

        # Build generations by readiness to preserve branches/joins.
        while remaining:
            ready = [
                node_name
                for node_name in topo_order
                if node_name in remaining
                and set(comp_graph.predecessors(node_name)).issubset(executed)
            ]
            if not ready:
                # Fallback for malformed/cyclic inputs: progress with first remaining node.
                ready = [next(iter(sorted(remaining)))]
            generations.append(ready)
            executed.update(ready)
            remaining -= set(ready)

        if not generations:
            continue

        executed_nodes = set(generations[0])

        for frontier in generations[1:]:
            service_targets = [
                node_name
                for node_name in frontier
                if node_name.startswith("service_") and node_name in service_map
            ]

            if executed_nodes and service_targets:
                subgraph = comp_graph.subgraph(executed_nodes).copy()
                subgraph_order = _safe_topological_order(subgraph)
                if not subgraph_order:
                    executed_nodes.update(frontier)
                    continue

                global_node_ids = [node_map[node_name] for node_name in subgraph_order if node_name in node_map]
                if len(global_node_ids) != len(subgraph_order):
                    executed_nodes.update(frontier)
                    continue

                edge_index = _build_local_edge_index(subgraph_order, subgraph)
                x = torch.zeros((len(subgraph_order), 2), dtype=torch.float32)
                for idx, node_name in enumerate(subgraph_order):
                    if node_name.startswith("service_"):
                        x[idx, 0] = 1.0
                    else:
                        x[idx, 1] = 1.0

                out_degrees = dict(subgraph.out_degree())
                leaf_mask = torch.tensor(
                    [out_degrees.get(node_name, 0) == 0 for node_name in subgraph_order],
                    dtype=torch.bool,
                )

                for target_name in service_targets:
                    sample = Data(
                        x=x.clone(),
                        edge_index=edge_index.clone(),
                        global_node_ids=torch.tensor(global_node_ids, dtype=torch.long),
                        leaf_mask=leaf_mask.clone(),
                        y=torch.tensor(service_map[target_name], dtype=torch.long),
                        composition_idx=torch.tensor(comp_idx, dtype=torch.long),
                    )
                    samples.append(sample)

            executed_nodes.update(frontier)

    if not samples:
        raise ValueError("No incremental graph samples were created.")

    return samples


def prepare_graph_transformer_graph_data(
    samples: List[Data],
    max_len: int,
) -> Dict[str, torch.Tensor]:
    """
    Convert graph-context samples into padded tensors for Graph Transformer.

    The underlying context comes from real subgraphs, but the transformer
    still receives a deterministic topological node order for batching.
    """
    input_ids: List[List[int]] = []
    raw_node_ids: List[List[int]] = []
    lengths: List[int] = []
    targets: List[int] = []
    last_node_ids: List[int] = []

    for sample in samples:
        node_ids = sample.global_node_ids.tolist()
        if not node_ids:
            continue

        if len(node_ids) > max_len:
            node_ids = node_ids[-max_len:]

        pad_len = max_len - len(node_ids)
        input_ids.append([node_id + 1 for node_id in node_ids] + ([0] * pad_len))
        raw_node_ids.append(node_ids + ([-1] * pad_len))
        lengths.append(len(node_ids))
        targets.append(int(sample.y.item()))
        last_node_ids.append(node_ids[-1])

    if not input_ids:
        raise ValueError("No graph-transformer graph samples were created.")

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "raw_node_ids": torch.tensor(raw_node_ids, dtype=torch.long),
        "attention_mask": torch.tensor([[token != 0 for token in seq] for seq in input_ids], dtype=torch.bool),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "targets": torch.tensor(targets, dtype=torch.long),
        "last_node_ids": torch.tensor(last_node_ids, dtype=torch.long),
    }


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GraphMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_relation_buckets: int, dropout: float):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.relation_bias = nn.Embedding(num_relation_buckets, num_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        relation_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        rel_bias = self.relation_bias(relation_ids).permute(0, 3, 1, 2)
        attn_scores = attn_scores + rel_bias

        valid_keys = attention_mask.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(~valid_keys, -1e9)

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)

        valid_queries = attention_mask.unsqueeze(-1)
        output = output * valid_queries
        return output


class GraphTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_relation_buckets: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = GraphMultiHeadAttention(d_model, num_heads, num_relation_buckets, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)

    def forward(
        self,
        x: torch.Tensor,
        relation_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), relation_ids, attention_mask)
        x = x + self.ff(self.norm2(x))
        return x


class CausalDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        causal_mask: torch.Tensor,
        key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        attn_input = self.norm1(x)
        attn_out, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            attn_mask=causal_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm2(x))
        return x


class SequenceOutputHead(nn.Module):
    def __init__(self, d_model: int, num_services: int):
        super().__init__()
        self.service_embedding = nn.Embedding(num_services, d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, seq_repr: torch.Tensor) -> torch.Tensor:
        seq_repr = self.proj(seq_repr)
        return torch.matmul(seq_repr, self.service_embedding.weight.t())


class GraphTransformerRecommender(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        num_services: int,
        max_len: int,
        relation_lookup: torch.Tensor,
        pad_relation_bucket: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        successor_service_map: Optional[Dict[int, List[int]]] = None,
    ):
        super().__init__()
        self.register_buffer("relation_lookup", relation_lookup, persistent=False)
        self.pad_relation_bucket = pad_relation_bucket
        self.successor_service_map = successor_service_map

        self.node_embedding = nn.Embedding(num_nodes + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            GraphTransformerBlock(d_model, num_heads, pad_relation_bucket + 1, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = SequenceOutputHead(d_model, num_services)

    def _build_relation_ids(self, raw_node_ids: torch.Tensor) -> torch.Tensor:
        valid = raw_node_ids >= 0
        clamped = raw_node_ids.clamp(min=0)
        relation_ids = self.relation_lookup[clamped.unsqueeze(2), clamped.unsqueeze(1)]
        valid_pairs = valid.unsqueeze(2) & valid.unsqueeze(1)
        pad_tensor = torch.full_like(relation_ids, self.pad_relation_bucket)
        return torch.where(valid_pairs, relation_ids, pad_tensor)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        raw_node_ids: torch.Tensor,
        last_node_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.node_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        attention_mask = input_ids != 0
        relation_ids = self._build_relation_ids(raw_node_ids)

        for block in self.blocks:
            x = block(x, relation_ids, attention_mask)

        x = self.final_norm(x)
        last_positions = lengths.clamp(min=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        seq_repr = x[batch_idx, last_positions]

        logits = stabilize_logits(self.output_head(seq_repr))
        return apply_candidate_mask(logits, last_node_ids, self.successor_service_map)


class SASRecRecommender(nn.Module):
    """Standard SASRec-style autoregressive self-attention recommender."""

    def __init__(
        self,
        num_nodes: int,
        num_services: int,
        max_len: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        successor_service_map: Optional[Dict[int, List[int]]] = None,
    ):
        super().__init__()
        self.successor_service_map = successor_service_map
        self.node_embedding = nn.Embedding(num_nodes + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = SequenceOutputHead(d_model, num_services)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        raw_node_ids: torch.Tensor,
        last_node_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del raw_node_ids
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.node_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )
        key_padding_mask = input_ids == 0
        x = self.encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask)
        x = self.final_norm(x)
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        last_positions = lengths.clamp(min=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        seq_repr = x[batch_idx, last_positions]

        logits = stabilize_logits(self.output_head(seq_repr))
        return apply_candidate_mask(logits, last_node_ids, self.successor_service_map)


class GPTRecommender(nn.Module):
    """Decoder-only GPT-style next-step predictor."""

    def __init__(
        self,
        num_nodes: int,
        num_services: int,
        max_len: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
        successor_service_map: Optional[Dict[int, List[int]]] = None,
    ):
        super().__init__()
        self.successor_service_map = successor_service_map
        self.node_embedding = nn.Embedding(num_nodes + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            CausalDecoderBlock(d_model, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output_head = SequenceOutputHead(d_model, num_services)

    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        raw_node_ids: torch.Tensor,
        last_node_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del raw_node_ids
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        x = self.node_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=input_ids.device),
            diagonal=1,
        )
        key_padding_mask = input_ids == 0

        for block in self.blocks:
            x = block(x, causal_mask, key_padding_mask)

        x = self.final_norm(x)
        x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        last_positions = lengths.clamp(min=1) - 1
        batch_idx = torch.arange(input_ids.size(0), device=input_ids.device)
        seq_repr = x[batch_idx, last_positions]

        logits = stabilize_logits(self.output_head(seq_repr))
        return apply_candidate_mask(logits, last_node_ids, self.successor_service_map)


def train_srgnn_graph_model(
    model: nn.Module,
    train_samples: List[Data],
    test_samples: List[Data],
    service_node_indices: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
    model_name: str,
    batch_size: int = 256,
) -> Dict[str, float]:
    """Train SR-GNN on incremental subgraph contexts instead of flat paths."""
    from directed_dag_models import compute_metrics

    model = model.to(device)
    service_node_indices = service_node_indices.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_count = len(train_samples)

    def build_batch(sample_batch: List[Data]):
        valid_samples = [sample for sample in sample_batch if int(sample.global_node_ids.numel()) > 0]
        if not valid_samples:
            return None

        max_nodes = max(int(sample.global_node_ids.numel()) for sample in valid_samples)
        items = torch.zeros((len(valid_samples), max_nodes), dtype=torch.long, device=device)
        alias_inputs = torch.zeros((len(valid_samples), max_nodes), dtype=torch.long, device=device)
        mask = torch.zeros((len(valid_samples), max_nodes), dtype=torch.float32, device=device)
        adjacency = torch.zeros((len(valid_samples), max_nodes, max_nodes * 2), dtype=torch.float32, device=device)
        labels = []

        for batch_idx, sample in enumerate(valid_samples):
            num_nodes = int(sample.global_node_ids.numel())
            items[batch_idx, :num_nodes] = sample.global_node_ids.to(device) + 1
            alias_inputs[batch_idx, :num_nodes] = torch.arange(num_nodes, dtype=torch.long, device=device)
            mask[batch_idx, :num_nodes] = 1.0
            adjacency[batch_idx, :num_nodes, : num_nodes * 2] = _build_srgnn_adjacency(
                sample.edge_index, num_nodes
            ).to(device)
            labels.append(int(sample.y.item()))

        return items, adjacency, alias_inputs, mask, torch.tensor(labels, dtype=torch.long, device=device)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_count)
        total_loss = 0.0
        total_examples = 0

        for start in range(0, train_count, batch_size):
            batch_indices = permutation[start:start + batch_size].tolist()
            sample_batch = [train_samples[sample_idx] for sample_idx in batch_indices]
            batch_data = build_batch(sample_batch)
            if batch_data is None:
                continue
            items, adjacency, alias_inputs, mask, target = batch_data

            hidden = model(items, adjacency)
            seq_hidden = model.gather_sequence(hidden, alias_inputs)
            logits = model.compute_scores(seq_hidden, mask, service_node_indices)
            logits = stabilize_logits(logits)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * target.size(0)
            total_examples += target.size(0)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            avg_loss = total_loss / max(total_examples, 1)
            print(f"{model_name} Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    model.eval()
    preds_list: List[int] = []
    probs_list: List[List[float]] = []
    labels_list: List[int] = []

    with torch.no_grad():
        for start in range(0, len(test_samples), batch_size):
            sample_batch = test_samples[start:start + batch_size]
            batch_data = build_batch(sample_batch)
            if batch_data is None:
                continue
            items, adjacency, alias_inputs, mask, labels = batch_data
            hidden = model(items, adjacency)
            seq_hidden = model.gather_sequence(hidden, alias_inputs)
            logits = model.compute_scores(seq_hidden, mask, service_node_indices)
            logits = stabilize_logits(logits)
            probs = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            preds_list.extend(preds.cpu().tolist())
            probs_list.extend(probs.cpu().tolist())
            labels_list.extend(labels.cpu().tolist())

    if not preds_list:
        raise RuntimeError(f"No valid graph samples for {model_name}")

    return compute_metrics(
        torch.tensor(preds_list).numpy(),
        torch.tensor(labels_list).numpy(),
        torch.tensor(probs_list).numpy(),
        model_name,
    )


def _evaluate_model(
    model: nn.Module,
    data: Dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    model_name: str,
) -> Dict[str, float]:
    from directed_dag_models import compute_metrics

    model.eval()
    all_logits: List[torch.Tensor] = []

    with torch.no_grad():
        num_samples = data["input_ids"].size(0)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_slice = slice(start, end)
            logits = model(
                data["input_ids"][batch_slice].to(device),
                data["lengths"][batch_slice].to(device),
                data["raw_node_ids"][batch_slice].to(device),
                data["last_node_ids"][batch_slice].to(device),
            )
            all_logits.append(stabilize_logits(logits).cpu())

    logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

    return compute_metrics(
        preds.numpy(),
        data["targets"].numpy(),
        probs.numpy(),
        model_name,
    )


def train_sequence_model(
    model: nn.Module,
    train_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    model_name: str,
    weight_decay: float = 1e-4,
) -> Dict[str, float]:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_size = train_data["input_ids"].size(0)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(train_size)
        total_loss = 0.0
        total_examples = 0

        for start in range(0, train_size, batch_size):
            end = min(start + batch_size, train_size)
            batch_indices = permutation[start:end]

            logits = model(
                train_data["input_ids"][batch_indices].to(device),
                train_data["lengths"][batch_indices].to(device),
                train_data["raw_node_ids"][batch_indices].to(device),
                train_data["last_node_ids"][batch_indices].to(device),
            )
            logits = stabilize_logits(logits)
            targets = train_data["targets"][batch_indices].to(device)

            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch_count = batch_indices.numel()
            total_loss += loss.item() * batch_count
            total_examples += batch_count

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            avg_loss = total_loss / max(total_examples, 1)
            print(f"{model_name} Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

    return _evaluate_model(model, test_data, batch_size, device, model_name)


def run_graph_transformer_experiment(
    train_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    graph: nx.DiGraph,
    node_map: Dict[str, int],
    service_map: Dict[str, int],
    max_len: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    successor_service_map: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, float]:
    relation_lookup, pad_relation_bucket = build_graph_relation_lookup(graph, node_map)
    model = GraphTransformerRecommender(
        num_nodes=len(node_map),
        num_services=len(service_map),
        max_len=max_len,
        relation_lookup=relation_lookup,
        pad_relation_bucket=pad_relation_bucket,
        d_model=hidden_dim * 2,
        num_heads=4,
        num_layers=2,
        dropout=dropout,
        successor_service_map=successor_service_map,
    )
    return train_sequence_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr * 0.5,
        device=device,
        model_name="Graph Transformer",
    )


def run_sasrec_experiment(
    train_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    node_map: Dict[str, int],
    service_map: Dict[str, int],
    max_len: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    successor_service_map: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, float]:
    model = SASRecRecommender(
        num_nodes=len(node_map),
        num_services=len(service_map),
        max_len=max_len,
        d_model=hidden_dim * 2,
        num_heads=4,
        num_layers=2,
        dropout=dropout,
        successor_service_map=successor_service_map,
    )
    return train_sequence_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr * 0.5,
        device=device,
        model_name="SASRec",
    )


def run_srgnn_graph_experiment(
    train_samples: List[Data],
    test_samples: List[Data],
    num_nodes: int,
    service_node_indices: torch.Tensor,
    hidden_dim: int,
    epochs: int,
    lr: float,
    device: torch.device,
    batch_size: int = 256,
) -> Dict[str, float]:
    from directed_dag_models import SRGNNRecommender

    model = SRGNNRecommender(
        num_nodes=num_nodes,
        hidden=hidden_dim * 2,
        step=1,
        non_hybrid=False,
    )
    return train_srgnn_graph_model(
        model=model,
        train_samples=train_samples,
        test_samples=test_samples,
        service_node_indices=service_node_indices,
        epochs=epochs,
        lr=lr * 0.5,
        device=device,
        model_name="SR-GNN (graph)",
        batch_size=batch_size,
    )


def run_graph_transformer_graph_experiment(
    train_samples: List[Data],
    test_samples: List[Data],
    graph: nx.DiGraph,
    node_map: Dict[str, int],
    service_map: Dict[str, int],
    max_len: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    train_data = prepare_graph_transformer_graph_data(train_samples, max_len=max_len)
    test_data = prepare_graph_transformer_graph_data(test_samples, max_len=max_len)
    relation_lookup, pad_relation_bucket = build_graph_relation_lookup(graph, node_map)

    model = GraphTransformerRecommender(
        num_nodes=len(node_map),
        num_services=len(service_map),
        max_len=max_len,
        relation_lookup=relation_lookup,
        pad_relation_bucket=pad_relation_bucket,
        d_model=hidden_dim * 2,
        num_heads=4,
        num_layers=2,
        dropout=dropout,
        successor_service_map=None,
    )
    return train_sequence_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr * 0.5,
        device=device,
        model_name="Graph Transformer (graph)",
    )


def run_gpt_experiment(
    train_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    node_map: Dict[str, int],
    service_map: Dict[str, int],
    max_len: int,
    hidden_dim: int,
    dropout: float,
    epochs: int,
    lr: float,
    batch_size: int,
    device: torch.device,
    successor_service_map: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, float]:
    model = GPTRecommender(
        num_nodes=len(node_map),
        num_services=len(service_map),
        max_len=max_len,
        d_model=hidden_dim * 2,
        num_heads=4,
        num_layers=3,
        dropout=dropout,
        successor_service_map=successor_service_map,
    )
    return train_sequence_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr * 0.35,
        device=device,
        model_name="GPT",
    )
