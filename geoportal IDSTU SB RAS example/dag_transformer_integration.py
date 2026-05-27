#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration functions for DAG-Transformer into existing codebase.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import numpy as np
from dag_transformer import DAGTransformer, compute_topological_features, DAGAttentionMask


def prepare_dag_transformer_data(
    contexts: List[Tuple[str, ...]],
    targets: List[str],
    node_map: Dict[str, int],
    service_map: Dict[str, int],
    edge_index: torch.Tensor,
    num_nodes: int,
    max_len: int = 50,
    handle_cycles: bool = True
) -> Dict[str, torch.Tensor]:
    """
    Prepare data for DAG-Transformer training.
    
    Args:
        contexts: List of context sequences (tuples of node names)
        targets: List of target service names
        node_map: Mapping from node names to indices
        service_map: Mapping from service names to indices
        edge_index: Graph edge index (2, num_edges)
        num_nodes: Total number of nodes
        max_len: Maximum sequence length
    
    Returns:
        Dictionary with:
            - node_sequences: (N, max_len) padded node indices
            - targets: (N,) target service indices
            - depths: (N, max_len) depth values
            - topological_positions: (N, max_len) topological positions
            - node_ids_list: List of node id lists for attention mask
    """
    node_sequences = []
    target_indices = []
    depths_list = []
    topo_pos_list = []
    node_ids_list = []
    
    for ctx, target in zip(contexts, targets):
        # Map node names to indices
        node_ids = []
        for node_name in ctx:
            if node_name in node_map:
                node_ids.append(node_map[node_name])
        
        if not node_ids:
            continue
        
        # Compute topological features (with cycle handling)
        topo_pos, depths = compute_topological_features(edge_index, num_nodes, node_ids, handle_cycles=True)
        
        # Pad or truncate
        if len(node_ids) > max_len:
            node_ids = node_ids[-max_len:]
            topo_pos = topo_pos[-max_len:]
            depths = depths[-max_len:]
        
        # Pad sequences
        pad_len = max_len - len(node_ids)
        node_ids_padded = [0] * pad_len + node_ids
        topo_pos_padded = F.pad(topo_pos, (pad_len, 0), value=0)
        depths_padded = F.pad(depths, (pad_len, 0), value=0.0)
        
        node_sequences.append(node_ids_padded)
        target_indices.append(service_map[target])
        depths_list.append(depths_padded)
        topo_pos_list.append(topo_pos_padded)
        node_ids_list.append(node_ids)
    
    return {
        'node_sequences': torch.tensor(node_sequences, dtype=torch.long),
        'targets': torch.tensor(target_indices, dtype=torch.long),
        'depths': torch.stack(depths_list),
        'topological_positions': torch.stack(topo_pos_list),
        'node_ids_list': node_ids_list
    }


def train_dag_transformer_model(
    model: DAGTransformer,
    train_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    edge_index: torch.Tensor,
    num_nodes: int,
    optimizer,
    epochs: int,
    batch_size: int = 32,
    device: torch.device = None,
    scheduler=None
) -> Dict[str, float]:
    """
    Train DAG-Transformer model.
    
    Returns:
        Dictionary with metrics on test set
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Prepare batches
    train_size = train_data['node_sequences'].size(0)
    num_batches = (train_size + batch_size - 1) // batch_size
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # Shuffle data
        indices = torch.randperm(train_size)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, train_size)
            batch_indices = indices[start_idx:end_idx]
            
            # Get batch data
            batch_node_seqs = train_data['node_sequences'][batch_indices].to(device)
            batch_targets = train_data['targets'][batch_indices].to(device)
            batch_depths = train_data['depths'][batch_indices].to(device)
            batch_topo_pos = train_data['topological_positions'][batch_indices].to(device)
            batch_node_ids_list = [train_data['node_ids_list'][i] for i in batch_indices.tolist()]
            
            # Create attention mask for batch (with cycle handling)
            # Pass actual batch node sequences for proper per-sequence masking
            attn_mask = DAGAttentionMask.create_attention_mask(
                edge_index, num_nodes, batch_node_seqs.size(0), batch_node_seqs.size(1),
                batch_node_seqs, device, handle_cycles=True
            )
            
            # Ensure mask is 3D: (batch_size, seq_len, seq_len)
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).expand(batch_node_seqs.size(0), -1, -1)
            
            # Get last nodes for DAG masking
            last_nodes = torch.tensor([
                train_data['node_ids_list'][i][-1] if train_data['node_ids_list'][i] else 0
                for i in batch_indices.tolist()
            ], dtype=torch.long, device=device)
            
            # Forward pass
            optimizer.zero_grad()
            logits = model(batch_node_seqs, batch_depths, batch_topo_pos, attn_mask, last_nodes)
            
            # Check for NaN/Inf in logits before computing loss
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                print(f"Warning: NaN/Inf detected in logits at epoch {epoch+1}, batch {batch_idx}, skipping...")
                continue
            
            loss = criterion(logits, batch_targets)
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf detected in loss at epoch {epoch+1}, batch {batch_idx}, skipping...")
                continue
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
        
        if scheduler is not None:
            scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / num_batches
            print(f"DAG-Transformer Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Evaluation
    model.eval()
    with torch.no_grad():
        test_node_seqs = test_data['node_sequences'].to(device)
        test_targets = test_data['targets'].to(device)
        test_depths = test_data['depths'].to(device)
        test_topo_pos = test_data['topological_positions'].to(device)
        test_node_ids_list = test_data['node_ids_list']
        
        # Create attention mask for test set (with cycle handling)
        attn_mask = DAGAttentionMask.create_attention_mask(
            edge_index, num_nodes, test_node_seqs.size(0), test_node_seqs.size(1),
            test_node_seqs, device, handle_cycles=True
        )
        
        # Ensure mask is 3D: (batch_size, seq_len, seq_len)
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0).expand(test_node_seqs.size(0), -1, -1)
        
        # Get last nodes for DAG masking
        test_last_nodes = torch.tensor([
            test_data['node_ids_list'][i][-1] if test_data['node_ids_list'][i] else 0
            for i in range(len(test_data['node_ids_list']))
        ], dtype=torch.long, device=device)
        
        logits = model(test_node_seqs, test_depths, test_topo_pos, attn_mask, test_last_nodes)
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)
    
    # Compute metrics (using same function as other models)
    from directed_dag_models import compute_metrics
    metrics = compute_metrics(
        preds.cpu().numpy(),
        test_targets.cpu().numpy(),
        probs.cpu().numpy(),
        "DAG-Transformer"
    )
    
    return metrics


def add_dag_transformer_to_main(
    contexts_train: List[Tuple[str, ...]],
    contexts_test: List[Tuple[str, ...]],
    y_train: List[str],
    y_test: List[str],
    node_map: Dict[str, int],
    service_map: Dict[str, int],
    data_pyg,
    args,
    device: torch.device,
    graph=None,
    original_graph=None
) -> Dict[str, float]:
    """
    Add DAG-Transformer to the main experiment pipeline.
    
    This function can be called from the main() function in directed_dag_models.py
    """
    # Build DAG successors mapping (like GRU4Rec)
    # IMPORTANT: Use original_graph if available (not cycle-handled)
    # Cycle handling removes edges which makes masking too restrictive
    from collections import defaultdict
    successors = defaultdict(list)
    
    # Prefer original_graph over cycle-handled graph for successors mapping
    graph_to_use = original_graph if original_graph is not None else graph
    
    if graph_to_use is not None:
        for u, v in graph_to_use.edges():
            # Check if v is a service (starts with "service")
            if v.startswith("service") and v in service_map:
                u_node_idx = node_map.get(u, None)
                if u_node_idx is not None:
                    successors[u_node_idx].append(service_map[v])
    # If graph not provided, successors will be empty dict (no masking)
    
    # Prepare data
    train_data = prepare_dag_transformer_data(
        contexts_train, y_train, node_map, service_map,
        data_pyg.edge_index, len(node_map), max_len=args.max_len
    )
    
    test_data = prepare_dag_transformer_data(
        contexts_test, y_test, node_map, service_map,
        data_pyg.edge_index, len(node_map), max_len=args.max_len
    )
    
    # Create model with optimized architecture (based on best practices)
    model = DAGTransformer(
        num_nodes=len(node_map),
        num_services=len(service_map),
        d_model=args.hidden * 2,  # Use larger hidden dimension
        nhead=8,
        num_layers=2,  # Reduced to 2 for faster convergence (like BERT4Rec uses 2-4 layers)
        dim_feedforward=args.hidden * 4,  # Reduced from 8x to 4x for stability
        dropout=args.dropout,
        max_seq_len=args.max_len,
        dag_successors=dict(successors)  # Add DAG-aware masking
    )
    
    # Use Adam optimizer with optimal learning rate (like SR-GNN uses 5e-4)
    # Lower learning rate for Transformer models works better
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.5, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.9)
    
    # Train
    metrics = train_dag_transformer_model(
        model, train_data, test_data, data_pyg.edge_index, len(node_map),
        optimizer, args.epochs, batch_size=32, device=device, scheduler=scheduler
    )
    
    return metrics

