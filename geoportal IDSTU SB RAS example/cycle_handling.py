#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cycle Detection and Handling for DAG-based Models

Проблема: При построении глобального графа из всех последовательностей могут появиться циклы,
хотя отдельные последовательности являются DAG.

Решения:
1. Обнаружение циклов
2. Разбиение на Strongly Connected Components (SCC)
3. Удаление обратных ребер с меньшим весом
4. Использование per-composition графов
5. Модификация модели для работы с циклами
"""

import networkx as nx
import torch
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def detect_cycles(graph: nx.DiGraph) -> List[List]:
    """
    Обнаружить все циклы в графе.
    
    Returns:
        List of cycles (each cycle is a list of nodes)
    """
    try:
        cycles = list(nx.simple_cycles(graph))
        return cycles
    except:
        return []


def is_dag(graph: nx.DiGraph) -> bool:
    """Проверить, является ли граф DAG."""
    try:
        nx.topological_sort(graph)
        return True
    except nx.NetworkXError:
        return False


def break_cycles_by_weight(graph: nx.DiGraph, strategy: str = "remove_weakest") -> nx.DiGraph:
    """
    Разорвать циклы, удаляя ребра с меньшим весом.
    
    Args:
        graph: Граф с возможными циклами
        strategy: 
            - "remove_weakest": удалить ребро с минимальным весом в цикле
            - "remove_strongest": удалить ребро с максимальным весом (контр-интуитивно, но может быть полезно)
            - "remove_backward": удалить обратные ребра (A->B если уже есть B->A)
    
    Returns:
        Граф без циклов (DAG)
    """
    dag = graph.copy()
    
    if strategy == "remove_backward":
        # Удалить обратные ребра: если есть A->B и B->A, удалить то, у которого меньше вес
        edges_to_remove = []
        for u, v in list(dag.edges()):
            if dag.has_edge(v, u):
                weight_uv = dag[u][v].get('weight', 1.0)
                weight_vu = dag[v][u].get('weight', 1.0)
                if weight_uv < weight_vu:
                    edges_to_remove.append((u, v))
                else:
                    edges_to_remove.append((v, u))
        
        for edge in edges_to_remove:
            if dag.has_edge(*edge):
                dag.remove_edge(*edge)
    
    # Продолжаем удалять циклы пока они есть
    max_iterations = 100
    iteration = 0
    
    while not is_dag(dag) and iteration < max_iterations:
        cycles = detect_cycles(dag)
        if not cycles:
            break
        
        # Найти все ребра в циклах
        cycle_edges = set()
        for cycle in cycles:
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                if dag.has_edge(u, v):
                    weight = dag[u][v].get('weight', 1.0)
                    cycle_edges.add((u, v, weight))
        
        if not cycle_edges:
            break
        
        # Удалить ребро с минимальным весом
        if strategy == "remove_weakest":
            edge_to_remove = min(cycle_edges, key=lambda x: x[2])
            dag.remove_edge(edge_to_remove[0], edge_to_remove[1])
            logger.debug(f"Removed edge {edge_to_remove[0]}->{edge_to_remove[1]} (weight={edge_to_remove[2]})")
        elif strategy == "remove_strongest":
            edge_to_remove = max(cycle_edges, key=lambda x: x[2])
            dag.remove_edge(edge_to_remove[0], edge_to_remove[1])
            logger.debug(f"Removed edge {edge_to_remove[0]}->{edge_to_remove[1]} (weight={edge_to_remove[2]})")
        
        iteration += 1
    
    if not is_dag(dag):
        logger.warning(f"Failed to break all cycles after {max_iterations} iterations")
    
    return dag


def break_cycles_by_scc(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Разорвать циклы, используя Strongly Connected Components (SCC).
    
    Стратегия:
    1. Найти все SCC
    2. Для каждого SCC (который содержит циклы):
       - Найти минимальное остовное дерево (MST) или
       - Удалить ребра с минимальным весом до получения DAG
    
    Returns:
        Граф без циклов (DAG)
    """
    dag = graph.copy()
    
    # Найти все SCC
    sccs = list(nx.strongly_connected_components(dag))
    
    # Обработать каждый SCC
    for scc in sccs:
        if len(scc) == 1:
            continue  # Одиночные узлы не образуют циклов
        
        # Подграф для этого SCC
        subgraph = dag.subgraph(scc).copy()
        
        # Найти циклы в подграфе
        cycles = detect_cycles(subgraph)
        if not cycles:
            continue
        
        # Собрать все ребра в циклах
        cycle_edges = set()
        for cycle in cycles:
            for i in range(len(cycle)):
                u, v = cycle[i], cycle[(i + 1) % len(cycle)]
                if subgraph.has_edge(u, v):
                    weight = subgraph[u][v].get('weight', 1.0)
                    cycle_edges.add((u, v, weight))
        
        # Удалить ребра с минимальным весом до получения DAG
        temp_graph = subgraph.copy()
        while not is_dag(temp_graph) and cycle_edges:
            edge_to_remove = min(cycle_edges, key=lambda x: x[2])
            temp_graph.remove_edge(edge_to_remove[0], edge_to_remove[1])
            cycle_edges.remove(edge_to_remove)
        
        # Обновить основной граф
        for u, v in list(dag.edges()):
            if u in scc and v in scc and not temp_graph.has_edge(u, v):
                dag.remove_edge(u, v)
    
    return dag


def build_dag_from_paths(paths: List[List[str]], cycle_handling: str = "remove_weakest") -> nx.DiGraph:
    """
    Построить DAG из путей с обработкой циклов.
    
    Args:
        paths: Список путей (каждый путь - список узлов)
        cycle_handling: Стратегия обработки циклов:
            - "remove_weakest": удалить ребра с минимальным весом
            - "remove_backward": удалить обратные ребра
            - "scc": использовать SCC разбиение
            - "none": не обрабатывать (может привести к ошибкам)
    
    Returns:
        Граф без циклов (DAG)
    """
    # Построить граф со всеми ребрами
    g = nx.DiGraph()
    edge_counts = defaultdict(int)
    nodes = set()
    
    # Подсчитать частоту ребер
    for path in paths:
        nodes.update(path)
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            edge_counts[edge] += 1
    
    # Добавить ребра с весами
    for (u, v), count in edge_counts.items():
        g.add_edge(u, v, weight=count)
    
    if nodes:
        g.add_nodes_from(nodes)
    
    # Проверить на циклы
    cycles = detect_cycles(g)
    if cycles:
        logger.info(f"Detected {len(cycles)} cycles in global graph")
        logger.info(f"Cycle handling strategy: {cycle_handling}")
        
        if cycle_handling == "remove_weakest":
            g = break_cycles_by_weight(g, strategy="remove_weakest")
        elif cycle_handling == "remove_backward":
            g = break_cycles_by_weight(g, strategy="remove_backward")
        elif cycle_handling == "scc":
            g = break_cycles_by_scc(g)
        elif cycle_handling == "none":
            logger.warning("Cycle handling disabled - model may fail on cycles")
        else:
            raise ValueError(f"Unknown cycle handling strategy: {cycle_handling}")
        
        # Проверить результат
        if is_dag(g):
            logger.info("Successfully converted graph to DAG")
        else:
            remaining_cycles = detect_cycles(g)
            logger.warning(f"Still {len(remaining_cycles)} cycles remaining after processing")
    else:
        logger.info("No cycles detected - graph is already a DAG")
    
    return g


def compute_topological_features_with_cycles(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_ids: List[int],
    fallback_to_bfs: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Вычислить топологические признаки с обработкой циклов.
    
    Если граф содержит циклы:
    - Использовать BFS-based ordering вместо топологической сортировки
    - Использовать shortest path distance (может быть бесконечным в циклах)
    
    Args:
        edge_index: (2, num_edges) - граф
        num_nodes: количество узлов
        node_ids: список индексов узлов в последовательности
        fallback_to_bfs: использовать BFS ordering если есть циклы
    
    Returns:
        topological_positions: (seq_len,) - позиции
        depths: (seq_len,) - глубины
    """
    import networkx as nx
    
    # Построить граф
    G = nx.DiGraph()
    if edge_index.numel() > 0:
        src, dst = edge_index.cpu().numpy()
        G.add_edges_from(zip(src, dst))
    
    # Проверить на циклы
    is_dag_graph = is_dag(G)
    
    if is_dag_graph:
        # Стандартная обработка для DAG
        source_nodes = [n for n in G.nodes() if G.in_degree(n) == 0]
        if not source_nodes:
            source_nodes = list(G.nodes())[:1] if G.nodes() else [0]
        
        depths = {}
        for node in G.nodes():
            try:
                min_dist = min([nx.shortest_path_length(G, s, node) 
                               for s in source_nodes if nx.has_path(G, s, node)], default=0)
                depths[node] = min_dist
            except:
                depths[node] = 0
        
        try:
            topo_order = list(nx.topological_sort(G))
            topo_positions = {node: idx for idx, node in enumerate(topo_order)}
        except:
            topo_positions = {node: idx for idx, node in enumerate(G.nodes())}
    else:
        # Обработка для графа с циклами
        logger.debug("Graph contains cycles - using BFS-based ordering")
        
        # BFS-based ordering: начинаем с узлов с минимальным in-degree
        in_degrees = dict(G.in_degree())
        start_nodes = [n for n in G.nodes() if in_degrees.get(n, 0) == 0]
        if not start_nodes:
            start_nodes = [min(G.nodes())] if G.nodes() else [0]
        
        # BFS traversal для определения порядка
        visited = set()
        topo_positions = {}
        position = 0
        queue = start_nodes.copy()
        
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            topo_positions[node] = position
            position += 1
            
            # Добавить соседей
            for neighbor in G.successors(node):
                if neighbor not in visited:
                    queue.append(neighbor)
        
        # Добавить непосещенные узлы
        for node in G.nodes():
            if node not in topo_positions:
                topo_positions[node] = position
                position += 1
        
        # Вычислить глубины используя shortest path (может быть бесконечным в циклах)
        depths = {}
        for node in G.nodes():
            try:
                distances = []
                for s in start_nodes:
                    try:
                        dist = nx.shortest_path_length(G, s, node)
                        distances.append(dist)
                    except nx.NetworkXNoPath:
                        continue
                depths[node] = min(distances) if distances else 0
            except:
                depths[node] = 0
    
    # Извлечь признаки для последовательности
    seq_topo_pos = torch.tensor([topo_positions.get(nid, 0) for nid in node_ids], dtype=torch.long)
    seq_depths = torch.tensor([depths.get(nid, 0) for nid in node_ids], dtype=torch.float32)
    
    # Нормализовать глубины
    max_depth = seq_depths.max().item()
    if max_depth > 0:
        seq_depths = seq_depths / max_depth
    
    return seq_topo_pos, seq_depths


def create_attention_mask_with_cycles(
    edge_index: torch.Tensor,
    num_nodes: int,
    batch_size: int,
    seq_len: int,
    node_positions: torch.Tensor,
    device: torch.device,
    use_transitive_closure: bool = True
) -> torch.Tensor:
    """
    Создать attention mask с обработкой циклов.
    
    Если граф содержит циклы:
    - Использовать прямые ребра вместо транзитивного замыкания
    - Или использовать BFS-based reachability
    
    Args:
        edge_index: (2, num_edges) - граф
        num_nodes: количество узлов
        batch_size: размер батча
        seq_len: длина последовательности
        node_positions: (batch_size, seq_len) - позиции узлов
        device: torch device
        use_transitive_closure: использовать транзитивное замыкание (только для DAG)
    
    Returns:
        mask: (batch_size, seq_len, seq_len) - attention mask
    """
    import networkx as nx
    
    # Построить граф
    G = nx.DiGraph()
    if edge_index.numel() > 0:
        src, dst = edge_index.cpu().numpy()
        G.add_edges_from(zip(src, dst))
    
    # Проверить на циклы
    is_dag_graph = is_dag(G)
    
    if is_dag_graph and use_transitive_closure:
        # Использовать транзитивное замыкание (как в оригинале)
        closure = nx.transitive_closure(G)
        reachability = {}
        for u, v in closure.edges():
            if u not in reachability:
                reachability[u] = set()
            reachability[u].add(v)
    else:
        # Использовать BFS-based reachability для графов с циклами
        logger.debug("Using BFS-based reachability due to cycles")
        reachability = {}
        for node in G.nodes():
            reachable = set()
            queue = [node]
            visited = {node}
            while queue:
                current = queue.pop(0)
                reachable.add(current)
                for neighbor in G.successors(current):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            reachability[node] = reachable
    
    # Создать маску для каждого батча (use actual batch size)
    actual_batch_size = node_positions.size(0)
    masks = []
    for b in range(actual_batch_size):
        seq_nodes = node_positions[b].cpu().numpy()  # (seq_len,)
        seq_mask = torch.zeros((seq_len, seq_len), device=device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                node_i = int(seq_nodes[i])
                node_j = int(seq_nodes[j])
                
                # Разрешить attention если j достижим из i (или i == j)
                if node_i == node_j:
                    seq_mask[i, j] = 0.0
                elif node_i in reachability and node_j in reachability[node_i]:
                    seq_mask[i, j] = 0.0
                else:
                    seq_mask[i, j] = float('-inf')
        
        masks.append(seq_mask)
    
    return torch.stack(masks, dim=0)

