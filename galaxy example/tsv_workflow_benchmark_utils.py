#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utilities for benchmarking local models on the EuGalaxy TSV dataset.

This module adapts the data preparation ideas from:
- BTR / bioinformatics_tool_recommendation (path-based SR-GNN benchmark)
- galaxy_tool_recommendation (path-based GRU4Rec benchmark)

The actual models are still the local project implementations.
"""

from __future__ import annotations

import csv
import os
import random
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import torch

from directed_dag_models import (
    build_graph,
    create_training_pairs,
    extract_paths_from_compositions,
    prepare_pyg,
    split_data,
)
from modern_sequence_models import (
    build_successor_service_map,
    prepare_incremental_graph_model_data,
    prepare_sequence_model_data,
)

WORKFLOW_TSV_URL = (
    "https://raw.githubusercontent.com/anuprulez/"
    "galaxy_tool_recommendation/master/data/worflow-connection-20-04.tsv"
)

WORKFLOW_TSV_COLUMNS = [
    "wf_id",
    "wf_updated",
    "in_id",
    "in_tool",
    "in_tool_v",
    "out_id",
    "out_tool",
    "out_tool_v",
    "published",
    "deleted",
    "has_errors",
]


def ensure_file_downloaded(url: str, output_path: str | Path) -> Path:
    """Download a file only if it does not already exist."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not output_path.exists():
        urllib.request.urlretrieve(url, output_path)
    return output_path


def format_tool_id(tool_link: str) -> str:
    """Match Galaxy helper logic: extract tool id from tool link if needed."""
    tool_id_split = str(tool_link).split("/")
    return tool_id_split[-2] if len(tool_id_split) > 1 else str(tool_link)


def _normalize_bool_column(series: pd.Series, default: bool = False) -> pd.Series:
    normalized = series.copy()
    normalized.loc[normalized == "t"] = True
    normalized.loc[normalized == "f"] = False
    normalized = normalized.fillna(default)
    return normalized.astype(bool)


def load_workflow_tsv(
    tsv_path: str | Path,
    *,
    require_not_deleted: bool = True,
    require_error_free: bool = True,
    require_published: bool = False,
) -> pd.DataFrame:
    """Load and normalize the EuGalaxy workflow TSV."""
    df = pd.read_csv(tsv_path, sep="\t", header=None, names=WORKFLOW_TSV_COLUMNS)

    df["published"] = _normalize_bool_column(df["published"], default=False)
    df["deleted"] = _normalize_bool_column(df["deleted"], default=False)
    df["has_errors"] = _normalize_bool_column(df["has_errors"], default=False)

    df.loc[df["in_tool"].isnull(), "in_tool"] = "Input dataset"
    df = df[df["out_tool"].notna()].copy()

    if require_published:
        df = df[df["published"]]
    if require_not_deleted:
        df = df[~df["deleted"]]
    if require_error_free:
        df = df[~df["has_errors"]]

    return df.reset_index(drop=True)


def _tool_only_workflow_graph(group_df: pd.DataFrame) -> nx.DiGraph:
    """
    Build a tool-only DAG inspired by BTR processing.

    Input datasets / non-tool nodes are kept in the raw graph, then bypassed so
    the resulting graph only contains tool-to-tool edges.
    """
    raw_graph = nx.DiGraph()

    for _, row in group_df.iterrows():
        in_id = str(row["in_id"])
        out_id = str(row["out_id"])
        in_tool_name = str(row["in_tool"])
        out_tool_name = str(row["out_tool"])

        in_type = "tool" if in_tool_name != "Input dataset" else "data_input"
        out_type = "tool" if out_tool_name != "Input dataset" else "data_input"

        raw_graph.add_node(in_id, name=format_tool_id(in_tool_name), type=in_type)
        raw_graph.add_node(out_id, name=format_tool_id(out_tool_name), type=out_type)
        raw_graph.add_edge(in_id, out_id)

    tool_graph = nx.DiGraph()

    for node_id, attrs in raw_graph.nodes(data=True):
        if attrs["type"] == "tool":
            tool_graph.add_node(node_id, name=attrs["name"])

    for node_id, attrs in raw_graph.nodes(data=True):
        if attrs["type"] != "tool":
            continue

        stack = list(raw_graph.successors(node_id))
        visited = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            current_attrs = raw_graph.nodes[current]
            if current_attrs["type"] == "tool":
                if current != node_id:
                    tool_graph.add_edge(node_id, current)
            else:
                stack.extend(raw_graph.successors(current))

    # Remove isolated nodes that never participate in a transition.
    used_nodes = {u for edge in tool_graph.edges() for u in edge}
    if used_nodes:
        tool_graph = tool_graph.subgraph(sorted(used_nodes)).copy()

    return tool_graph


def build_btr_eugalaxy_compositions(df: pd.DataFrame) -> List[Dict]:
    """
    Convert the EuGalaxy TSV into composition dictionaries compatible with the
    local notebook/project pipeline.
    """
    compositions: List[Dict] = []

    for _, group_df in df.groupby("wf_id", sort=True):
        tool_graph = _tool_only_workflow_graph(group_df)
        if tool_graph.number_of_nodes() < 2 or tool_graph.number_of_edges() == 0:
            continue

        node_ids = {old_id: idx for idx, old_id in enumerate(tool_graph.nodes(), start=1)}
        nodes = [
            {"id": idx, "mid": tool_graph.nodes[old_id]["name"]}
            for old_id, idx in node_ids.items()
        ]
        links = [
            {"source": node_ids[u], "target": node_ids[v]}
            for u, v in tool_graph.edges()
        ]
        compositions.append({"nodes": nodes, "links": links})

    return compositions


def _galaxy_paths_for_workflow(edges: List[Tuple[str, str]]) -> List[List[str]]:
    """
    Reproduce the Galaxy workflow path extraction logic on a single workflow.
    """
    tool_parents: Dict[str, List[str]] = {}
    for child_tool, parent_tool in edges:
        tool_parents.setdefault(parent_tool, [])
        if child_tool not in tool_parents[parent_tool]:
            tool_parents[parent_tool].append(child_tool)

    all_parents = set()
    for children in tool_parents.values():
        all_parents.update(children)

    child_nodes = set(tool_parents.keys())
    roots = sorted(all_parents.difference(child_nodes))
    leaves = sorted(child_nodes.difference(all_parents))

    def find_paths(graph: Dict[str, List[str]], start: str, end: str, path: List[str] | None = None):
        path = [] if path is None else path
        path = path + [end]
        if start == end:
            return [path]
        if end not in graph:
            return []
        results = []
        for node in graph[end]:
            if node not in path:
                results.extend(find_paths(graph, start, node, path))
        return results

    paths: List[List[str]] = []
    for root in roots:
        for leaf in leaves:
            paths.extend(find_paths(tool_parents, root, leaf))

    return paths


def extract_galaxy_paths_with_indices(df: pd.DataFrame) -> List[Tuple[List[str], int]]:
    """
    Extract unique tool paths using the Galaxy repository path logic.

    Returns paths as local-project node names, e.g. `service_cutadapt`.
    """
    workflows: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    workflow_ids = sorted(df["wf_id"].astype(str).unique())
    wf_to_idx = {wf_id: idx for idx, wf_id in enumerate(workflow_ids)}

    for _, row in df.iterrows():
        wf_id = str(row["wf_id"])
        in_tool = format_tool_id(row["in_tool"])
        out_tool = format_tool_id(row["out_tool"])
        if in_tool and out_tool and out_tool != in_tool and in_tool != "Input dataset" and out_tool != "Input dataset":
            workflows[wf_id].append((out_tool, in_tool))

    unique_paths: Dict[Tuple[str, ...], int] = {}
    for wf_id in workflow_ids:
        for path in _galaxy_paths_for_workflow(workflows[wf_id]):
            if len(path) < 2:
                continue
            normalized_path = tuple(f"service_{tool_name}" for tool_name in path)
            unique_paths.setdefault(normalized_path, wf_to_idx[wf_id])

    return [(list(path), wf_idx) for path, wf_idx in unique_paths.items()]


def _sample_workflows_df(df: pd.DataFrame, max_workflows: int | None, seed: int) -> pd.DataFrame:
    """Optionally subsample workflows for a faster benchmark."""
    if max_workflows is None:
        return df

    workflow_ids = sorted(df["wf_id"].astype(str).unique())
    if len(workflow_ids) <= max_workflows:
        return df

    chosen_ids = set(random.Random(seed).sample(workflow_ids, max_workflows))
    return df[df["wf_id"].astype(str).isin(chosen_ids)].reset_index(drop=True)


def _limit_paths_with_indices(
    paths_with_idx: List[Tuple[List[str], int]],
    *,
    max_paths_total: int | None = None,
    max_paths_per_workflow: int | None = None,
    seed: int = 42,
) -> List[Tuple[List[str], int]]:
    """Limit path count globally and/or per workflow to avoid path explosion."""
    if max_paths_per_workflow is not None:
        grouped: Dict[int, List[Tuple[List[str], int]]] = defaultdict(list)
        for path, comp_idx in paths_with_idx:
            grouped[comp_idx].append((path, comp_idx))

        limited_paths: List[Tuple[List[str], int]] = []
        rng = random.Random(seed)
        for comp_idx in sorted(grouped):
            group = grouped[comp_idx]
            if len(group) > max_paths_per_workflow:
                group = rng.sample(group, max_paths_per_workflow)
            limited_paths.extend(group)
        paths_with_idx = limited_paths

    if max_paths_total is not None and len(paths_with_idx) > max_paths_total:
        rng = random.Random(seed)
        paths_with_idx = rng.sample(paths_with_idx, max_paths_total)

    return paths_with_idx


def _limit_graph_samples(
    samples: List,
    max_samples: int | None,
    seed: int,
):
    """Limit graph-context samples for faster experiments."""
    if max_samples is None or len(samples) <= max_samples:
        return samples
    return random.Random(seed).sample(samples, max_samples)


def _prepare_local_benchmark_data(
    paths_with_idx: List[Tuple[List[str], int]],
    compositions: List[Dict] | None,
    test_size: float,
    seed: int,
    max_len: int,
) -> Dict:
    contexts, targets, comp_indices = create_training_pairs(paths_with_idx)

    ctx_train, ctx_test, y_train, y_test, comp_train_idx, comp_test_idx = split_data(
        contexts, targets, comp_indices, test_size, seed
    )

    train_comp_set = set(comp_train_idx)
    train_paths_only = [path for path, comp_idx in paths_with_idx if comp_idx in train_comp_set]
    all_nodes = sorted({node for path, _ in paths_with_idx for node in path})

    graph = build_graph(train_paths_only)
    graph.add_nodes_from(all_nodes)

    nodes = sorted(graph.nodes())
    data_pyg, node_map = prepare_pyg(graph, nodes)

    services = sorted(set(targets))
    service_map = {service_name: idx for idx, service_name in enumerate(services)}
    service_node_indices = torch.tensor([node_map[service_name] + 1 for service_name in services], dtype=torch.long)

    targets_train_tensor = torch.tensor([service_map[target] for target in y_train], dtype=torch.long)
    targets_test_tensor = torch.tensor([service_map[target] for target in y_test], dtype=torch.long)
    train_target_indices = [service_map[target] for target in y_train]
    test_target_indices = [service_map[target] for target in y_test]

    sequence_train_data = prepare_sequence_model_data(ctx_train, y_train, node_map, service_map, max_len=max_len)
    sequence_test_data = prepare_sequence_model_data(ctx_test, y_test, node_map, service_map, max_len=max_len)

    graph_train_samples = None
    graph_test_samples = None
    if compositions:
        graph_train_samples = prepare_incremental_graph_model_data(
            compositions, sorted(set(comp_train_idx)), node_map, service_map
        )
        graph_test_samples = prepare_incremental_graph_model_data(
            compositions, sorted(set(comp_test_idx)), node_map, service_map
        )

    successors = build_successor_service_map(graph, node_map, service_map)
    successor_nodes: Dict[int, List[int]] = defaultdict(list)
    for u, v in graph.edges():
        if u in node_map and v in node_map:
            successor_nodes[node_map[u]].append(node_map[v])

    return {
        "compositions": compositions,
        "paths_with_idx": paths_with_idx,
        "contexts": contexts,
        "targets": targets,
        "comp_indices": comp_indices,
        "ctx_train": ctx_train,
        "ctx_test": ctx_test,
        "y_train": y_train,
        "y_test": y_test,
        "comp_train_idx": comp_train_idx,
        "comp_test_idx": comp_test_idx,
        "graph": graph,
        "data_pyg": data_pyg,
        "node_map": node_map,
        "service_map": service_map,
        "service_node_indices": service_node_indices,
        "targets_train_tensor": targets_train_tensor,
        "targets_test_tensor": targets_test_tensor,
        "train_target_indices": train_target_indices,
        "test_target_indices": test_target_indices,
        "sequence_train_data": sequence_train_data,
        "sequence_test_data": sequence_test_data,
        "graph_train_samples": graph_train_samples,
        "graph_test_samples": graph_test_samples,
        "successors": successors,
        "successor_nodes": successor_nodes,
    }


def prepare_btr_path_and_graph_benchmark(
    tsv_path: str | Path,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    max_len: int = 25,
    max_workflows: int | None = None,
    max_paths_total: int | None = 50000,
    max_paths_per_workflow: int | None = 20,
    max_graph_samples_per_split: int | None = 25000,
) -> Dict:
    """
    Prepare BTR-style EuGalaxy data:
    TSV -> workflow DAGs -> tool-only compositions -> path + graph benchmarks.
    """
    df = load_workflow_tsv(tsv_path)
    df = _sample_workflows_df(df, max_workflows=max_workflows, seed=seed)
    compositions = build_btr_eugalaxy_compositions(df)
    paths_with_idx = extract_paths_from_compositions(compositions)
    paths_with_idx = _limit_paths_with_indices(
        paths_with_idx,
        max_paths_total=max_paths_total,
        max_paths_per_workflow=max_paths_per_workflow,
        seed=seed,
    )
    prepared = _prepare_local_benchmark_data(paths_with_idx, compositions, test_size, seed, max_len)
    prepared["graph_train_samples"] = _limit_graph_samples(
        prepared["graph_train_samples"], max_graph_samples_per_split, seed
    )
    prepared["graph_test_samples"] = _limit_graph_samples(
        prepared["graph_test_samples"], max_graph_samples_per_split, seed
    )
    prepared["data_source"] = "BTR / EuGalaxy tool-only DAGs"
    prepared["num_workflows"] = len(compositions)
    return prepared


def prepare_galaxy_path_benchmark(
    tsv_path: str | Path,
    *,
    test_size: float = 0.2,
    seed: int = 42,
    max_len: int = 25,
    max_workflows: int | None = None,
    max_paths_total: int | None = 50000,
    max_paths_per_workflow: int | None = 20,
) -> Dict:
    """
    Prepare Galaxy-style sequence/path data:
    TSV -> extracted workflow paths -> local path benchmark tensors.
    """
    df = load_workflow_tsv(tsv_path)
    df = _sample_workflows_df(df, max_workflows=max_workflows, seed=seed)
    paths_with_idx = extract_galaxy_paths_with_indices(df)
    paths_with_idx = _limit_paths_with_indices(
        paths_with_idx,
        max_paths_total=max_paths_total,
        max_paths_per_workflow=max_paths_per_workflow,
        seed=seed,
    )

    workflow_ids = sorted(df["wf_id"].astype(str).unique())
    prepared = _prepare_local_benchmark_data(paths_with_idx, None, test_size, seed, max_len)
    prepared["data_source"] = "Galaxy path extraction"
    prepared["num_workflows"] = len(workflow_ids)
    return prepared
