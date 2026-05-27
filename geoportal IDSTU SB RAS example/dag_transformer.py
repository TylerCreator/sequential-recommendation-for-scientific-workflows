#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAG-Transformer: Transformer-based model for DAG workflow sequence recommendation

This model uses Transformer architecture with DAG-aware attention mechanism:
- Attention mask respects DAG structure (only predecessors can attend)
- Topological positional encoding based on node depth and order
- Depth-aware node embeddings
- Multi-head attention for aggregating information from predecessors

Based on:
- Transformer architecture (Vaswani et al., 2017)
- Graphormer (Ying et al., NeurIPS 2021)
- DAG-aware attention mechanisms
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import networkx as nx
import numpy as np


class TopologicalPositionalEncoding(nn.Module):
    """
    Positional encoding based on topological order in DAG.
    Nodes at the same topological level get similar positional encodings.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, topological_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            topological_positions: (batch_size, seq_len) - topological positions of nodes
        Returns:
            positional_encodings: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = topological_positions.shape
        # Clamp positions to valid range
        positions = topological_positions.clamp(0, self.pe.size(1) - 1).long()
        return self.pe[:, positions].squeeze(0)


class DepthAwareEmbedding(nn.Module):
    """
    Node embeddings that incorporate depth information from DAG.
    """
    
    def __init__(self, num_nodes: int, d_model: int, depth_emb_dim: int = 16):
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes + 1, d_model - depth_emb_dim, padding_idx=0)
        self.depth_embedding = nn.Embedding(100, depth_emb_dim)  # Support up to depth 100
        
    def forward(self, node_ids: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_ids: (batch_size, seq_len) - node indices
            depths: (batch_size, seq_len) - depth values (normalized 0-99)
        Returns:
            embeddings: (batch_size, seq_len, d_model)
        """
        node_emb = self.node_embedding(node_ids)
        # Clamp depths to valid range and handle NaN/Inf
        depths = torch.clamp(depths, 0.0, 1.0)
        depths = torch.where(torch.isfinite(depths), depths, torch.zeros_like(depths))
        depth_ids = (depths * 99).long().clamp(0, 99)
        depth_emb = self.depth_embedding(depth_ids)
        return torch.cat([node_emb, depth_emb], dim=-1)


class DAGAttentionMask:
    """
    Utility class for creating DAG-aware attention masks.
    """
    
    @staticmethod
    def compute_transitive_closure(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute transitive closure of DAG using NetworkX.
        Returns adjacency matrix where A[i,j] = 1 if j is ancestor of i.
        """
        # Convert to NetworkX graph
        G = nx.DiGraph()
        if edge_index.numel() > 0:
            src, dst = edge_index.cpu().numpy()
            G.add_edges_from(zip(src, dst))
        
        # Compute transitive closure
        closure = nx.transitive_closure(G)
        
        # Convert to tensor
        mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        for u, v in closure.edges():
            mask[u, v] = True
        
        # Add self-loops
        mask.fill_diagonal_(True)
        
        return mask
    
    @staticmethod
    def create_attention_mask(
        edge_index: torch.Tensor,
        num_nodes: int,
        batch_size: int,
        seq_len: int,
        node_positions: torch.Tensor,
        device: torch.device,
        handle_cycles: bool = True
    ) -> torch.Tensor:
        """
        Create attention mask for batch of sequences.
        Handles cycles in the graph if present.
        
        Args:
            edge_index: (2, num_edges) - global graph edges
            num_nodes: number of nodes in global graph
            batch_size: batch size
            seq_len: sequence length
            node_positions: (batch_size, seq_len) - node indices in sequences
            device: torch device
            handle_cycles: if True, use cycle-aware mask creation
            
        Returns:
            mask: (batch_size, seq_len, seq_len) - attention mask
                  True/0.0 = allowed, False/-inf = blocked
        """
        if handle_cycles:
            # Use cycle-aware mask creation
            try:
                from cycle_handling import create_attention_mask_with_cycles
                return create_attention_mask_with_cycles(
                    edge_index, num_nodes, batch_size, seq_len, node_positions, device
                )
            except ImportError:
                pass  # Fall back to standard computation
        
        # Standard computation (assumes DAG)
        # Compute transitive closure once (can be cached)
        closure = DAGAttentionMask.compute_transitive_closure(edge_index, num_nodes).to(device)
        
        # Create mask for each sequence in batch (use actual batch size)
        actual_batch_size = node_positions.size(0)
        masks = []
        for b in range(actual_batch_size):
            seq_nodes = node_positions[b]  # (seq_len,)
            seq_mask = torch.zeros((seq_len, seq_len), device=device)
            
            for i in range(seq_len):
                node_i = seq_nodes[i].item()
                # Skip padding tokens (node_id = 0)
                if node_i == 0:
                    # Block all attention from padding tokens
                    seq_mask[i, :] = -1e9  # Use large negative value instead of -inf
                    continue
                
                for j in range(seq_len):
                    node_j = seq_nodes[j].item()
                    # Block attention to padding tokens
                    if node_j == 0:
                        seq_mask[i, j] = -1e9  # Use large negative value instead of -inf
                    # Allow attention if j is ancestor of i (including self)
                    elif closure[node_i, node_j]:
                        seq_mask[i, j] = 0.0
                    else:
                        seq_mask[i, j] = -1e9  # Use large negative value instead of -inf
            
            masks.append(seq_mask)
        
        return torch.stack(masks, dim=0)


class DAGTransformerBlock(nn.Module):
    """
    Transformer block with DAG-aware attention.
    Processes each sequence in batch with its own attention mask.
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attn_mask: (batch_size, seq_len, seq_len) - attention mask for each sequence
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Process each sequence separately if we have per-sequence masks
        # This is necessary because MultiheadAttention with batch_first=True
        # expects a single (seq_len, seq_len) mask for the whole batch
        if attn_mask is not None and attn_mask.dim() == 3 and attn_mask.size(0) > 1:
            # Process each sequence in the batch separately
            batch_size = x.size(0)
            outputs = []
            for b in range(batch_size):
                x_seq = x[b:b+1]  # (1, seq_len, d_model)
                mask_seq = attn_mask[b]  # (seq_len, seq_len)
                # Replace -inf with large negative value to avoid NaN
                mask_seq = torch.where(mask_seq == float('-inf'), torch.tensor(-1e9, device=mask_seq.device), mask_seq)
                attn_output, _ = self.self_attn(x_seq, x_seq, x_seq, attn_mask=mask_seq, need_weights=False)
                x_seq = self.norm1(x_seq + self.dropout(attn_output))
                ffn_output = self.ffn(x_seq)
                x_seq = self.norm2(x_seq + ffn_output)
                outputs.append(x_seq)
            x = torch.cat(outputs, dim=0)
        else:
            # Single mask for whole batch (all sequences share same structure)
            mask_2d = attn_mask[0] if attn_mask is not None and attn_mask.dim() == 3 else attn_mask
            if mask_2d is not None:
                # Replace -inf with large negative value
                mask_2d = torch.where(mask_2d == float('-inf'), torch.tensor(-1e9, device=mask_2d.device), mask_2d)
            attn_output, _ = self.self_attn(x, x, x, attn_mask=mask_2d, need_weights=False)
            x = self.norm1(x + self.dropout(attn_output))
            ffn_output = self.ffn(x)
            x = self.norm2(x + ffn_output)
        
        return x


class DAGTransformer(nn.Module):
    """
    DAG-Transformer model for workflow sequence recommendation.
    
    Architecture:
    1. Depth-aware node embeddings
    2. Topological positional encoding
    3. Stack of DAG-aware transformer blocks
    4. Output head for next service prediction
    """
    
    def __init__(
        self,
        num_nodes: int,
        num_services: int,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 50,
        dag_successors: Optional[Dict[int, List[int]]] = None
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_services = num_services
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dag_successors = dag_successors or {}
        
        # Embeddings
        self.node_embedding = DepthAwareEmbedding(num_nodes, d_model)
        self.pos_encoding = TopologicalPositionalEncoding(d_model, max_len=max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            DAGTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Simplified and effective sequence aggregation (like SR-GNN but simpler)
        # Use attention-weighted sum similar to SR-GNN's compute_scores
        self.linear_one = nn.Linear(d_model, d_model, bias=True)
        self.linear_two = nn.Linear(d_model, d_model, bias=True)
        self.linear_three = nn.Linear(d_model, 1, bias=False)
        self.linear_transform = nn.Linear(d_model * 2, d_model, bias=True)
        
        # Output head with dot product to embeddings (like SR-GNN)
        self.output_embedding = nn.Embedding(num_services, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights properly to prevent NaN
        self._init_weights()
        
        # Initialize linear layers like SR-GNN (uniform initialization)
        stdv = 1.0 / math.sqrt(self.d_model)
        for weight in [self.linear_one, self.linear_two, self.linear_three, self.linear_transform]:
            if hasattr(weight, 'weight'):
                weight.weight.data.uniform_(-stdv, stdv)
            if hasattr(weight, 'bias') and weight.bias is not None:
                weight.bias.data.uniform_(-stdv, stdv)
    
    def _init_weights(self):
        """Initialize weights to prevent NaN during training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use standard gain (was too conservative with 0.5)
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.MultiheadAttention):
                # Initialize attention weights properly
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight, gain=1.0)
                if hasattr(module, 'out_proj') and module.out_proj.weight is not None:
                    nn.init.xavier_uniform_(module.out_proj.weight, gain=1.0)
    
    def forward(
        self,
        node_sequences: torch.Tensor,
        depth_sequences: torch.Tensor,
        topological_positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        last_nodes: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            node_sequences: (batch_size, seq_len) - node indices
            depth_sequences: (batch_size, seq_len) - depth values
            topological_positions: (batch_size, seq_len) - topological positions
            attn_mask: (batch_size, seq_len, seq_len) - optional attention mask
            last_nodes: (batch_size,) - last node indices for DAG masking
        
        Returns:
            logits: (batch_size, num_services) - prediction logits
        """
        batch_size, seq_len = node_sequences.shape
        
        # Embeddings
        x = self.node_embedding(node_sequences, depth_sequences)  # (batch, seq_len, d_model)
        pos_emb = self.pos_encoding(topological_positions)  # (batch, seq_len, d_model)
        x = x + pos_emb
        x = self.dropout(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, attn_mask=attn_mask)
        
        # Sequence representation using SR-GNN style attention (proven effective)
        non_padding_mask = (node_sequences != 0).float()  # (batch_size, seq_len)
        seq_lengths = non_padding_mask.sum(dim=1).clamp(min=1)  # (batch_size,)
        last_positions = (seq_lengths - 1).clamp(min=0).long()  # (batch_size,)
        batch_indices = torch.arange(batch_size, device=node_sequences.device)
        last_hidden = x[batch_indices, last_positions]  # (batch_size, d_model)
        
        # Attention-weighted aggregation (exactly like SR-GNN compute_scores)
        q1 = self.linear_one(last_hidden).unsqueeze(1)  # (batch_size, 1, d_model)
        q2 = self.linear_two(x)  # (batch_size, seq_len, d_model)
        alpha = self.linear_three(torch.sigmoid(q1 + q2)).squeeze(-1)  # (batch_size, seq_len)
        mask_expanded = non_padding_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)
        weighted_sum = torch.sum(alpha.unsqueeze(-1) * x * mask_expanded, dim=1)  # (batch_size, d_model)
        
        # Hybrid: combine weighted_sum with last_hidden (like SR-GNN non_hybrid=False)
        seq_repr = self.linear_transform(torch.cat([weighted_sum, last_hidden], dim=1))  # (batch_size, d_model)
        
        # Output using dot product with embeddings (like SR-GNN)
        candidates = self.output_embedding.weight  # (num_services, d_model)
        logits = torch.matmul(seq_repr, candidates.t())  # (batch_size, num_services)
        
        # DAG-aware output masking (like GRU4Rec)
        if last_nodes is not None and self.dag_successors:
            mask = torch.zeros_like(logits)
            for idx, node in enumerate(last_nodes.tolist()):
                succ = self.dag_successors.get(node, [])
                if succ:
                    mask[idx] = -1e9
                    mask[idx, succ] = 0.0
            logits = logits + mask
        
        return logits


def compute_topological_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_ids: List[int],
    handle_cycles: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute topological positions and depths for nodes.
    Handles cycles in the graph if present.
    
    Args:
        edge_index: (2, num_edges) - graph edges
        num_nodes: total number of nodes
        node_ids: list of node indices in sequence
        handle_cycles: if True, use cycle-aware computation
    
    Returns:
        topological_positions: (seq_len,) - topological positions
        depths: (seq_len,) - normalized depths
    """
    if handle_cycles:
        # Use cycle-aware computation
        try:
            from cycle_handling import compute_topological_features_with_cycles
            return compute_topological_features_with_cycles(edge_index, num_nodes, node_ids)
        except ImportError:
            pass  # Fall back to standard computation
    
    # Standard computation (assumes DAG)
    # Build graph
    G = nx.DiGraph()
    if edge_index.numel() > 0:
        src, dst = edge_index.cpu().numpy()
        G.add_edges_from(zip(src, dst))
    
    # Compute depths (distance from source nodes)
    source_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not source_nodes:
        source_nodes = list(G.nodes())[:1] if G.nodes() else [0]
    
    depths = {}
    for node in G.nodes():
        try:
            # Shortest path length from any source
            min_dist = min([nx.shortest_path_length(G, s, node) for s in source_nodes if nx.has_path(G, s, node)], default=0)
            depths[node] = min_dist
        except:
            depths[node] = 0
    
    # Topological sort
    try:
        topo_order = list(nx.topological_sort(G))
        topo_positions = {node: idx for idx, node in enumerate(topo_order)}
    except:
        # If not a DAG, use arbitrary ordering
        topo_positions = {node: idx for idx, node in enumerate(G.nodes())}
    
    # Extract features for sequence nodes
    seq_topo_pos = torch.tensor([topo_positions.get(nid, 0) for nid in node_ids], dtype=torch.long)
    seq_depths = torch.tensor([depths.get(nid, 0) for nid in node_ids], dtype=torch.float32)
    
    # Normalize depths with safety checks
    max_depth = seq_depths.max().item()
    if max_depth > 0:
        seq_depths = seq_depths / max_depth
    else:
        # If all depths are 0, set to small value to avoid issues
        seq_depths = torch.zeros_like(seq_depths)
    
    # Ensure no NaN or Inf
    seq_depths = torch.where(torch.isfinite(seq_depths), seq_depths, torch.zeros_like(seq_depths))
    seq_depths = torch.clamp(seq_depths, 0.0, 1.0)
    
    return seq_topo_pos, seq_depths


# Example usage and training function
def train_dag_transformer(
    model: DAGTransformer,
    train_loader,
    optimizer,
    criterion,
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
    epochs: int = 50
):
    """
    Training function for DAG-Transformer.
    """
    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # batch should contain:
            # - node_sequences: (batch_size, seq_len)
            # - targets: (batch_size,) - service indices
            # - node_ids_list: list of node id lists for computing topological features
            
            node_sequences = batch['node_sequences'].to(device)
            targets = batch['targets'].to(device)
            node_ids_list = batch['node_ids_list']
            
            # Compute topological features for each sequence
            batch_topo_pos = []
            batch_depths = []
            batch_attn_masks = []
            
            for seq_node_ids in node_ids_list:
                topo_pos, depths = compute_topological_features(edge_index, num_nodes, seq_node_ids)
                batch_topo_pos.append(topo_pos)
                batch_depths.append(depths)
            
            # Pad sequences
            max_len = node_sequences.size(1)
            batch_topo_pos = torch.stack([
                F.pad(pos, (max_len - len(pos), 0)) if len(pos) < max_len else pos[:max_len]
                for pos in batch_topo_pos
            ]).to(device)
            batch_depths = torch.stack([
                F.pad(dep, (max_len - len(dep), 0), value=0.0) if len(dep) < max_len else dep[:max_len]
                for dep in batch_depths
            ]).to(device)
            
            # Create attention masks
            attn_mask = DAGAttentionMask.create_attention_mask(
                edge_index, num_nodes, node_sequences.size(0), max_len,
                node_sequences, device
            )
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(node_sequences, batch_depths, batch_topo_pos, attn_mask)
            loss = criterion(logits, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    # Example initialization
    model = DAGTransformer(
        num_nodes=1000,
        num_services=50,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("DAG-Transformer model created successfully!")

