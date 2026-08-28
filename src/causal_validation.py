"""Validation helpers for Phase 2 datasets, DAGs, and SCMs."""

from __future__ import annotations

from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from src.phase2_schema import CAUSAL_FEATURES


def validate_numeric_frame(frame: pd.DataFrame, features: list[str] | None = None) -> None:
    features = features or CAUSAL_FEATURES
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing causal feature columns: {missing}")
    values = frame[features].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Causal dataset contains NaN or infinity.")


def constant_features(frame: pd.DataFrame, features: list[str] | None = None) -> list[str]:
    features = features or CAUSAL_FEATURES
    return [feature for feature in features if frame[feature].nunique(dropna=False) <= 1]


def validate_no_constant_features(frame: pd.DataFrame, features: list[str]) -> None:
    constants = constant_features(frame, features)
    if constants:
        raise ValueError(f"Constant causal variables remain after filtering: {constants}")


def validate_dag(dag: dict[str, list[dict[str, Any]]], nodes: list[str]) -> None:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    for source, edges in dag.items():
        for edge in edges:
            target = edge["target"]
            if source == target:
                raise ValueError(f"Self-loop found in DAG: {source} -> {target}")
            graph.add_edge(source, target)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph is not acyclic.")


def adjacency_to_dag(adjacency: np.ndarray, nodes: list[str]) -> dict[str, list[dict[str, float]]]:
    dag: dict[str, list[dict[str, float]]] = {node: [] for node in nodes}
    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes):
            weight = float(adjacency[i, j])
            if i != j and abs(weight) > 0:
                dag[source].append({"target": target, "weight": weight})
    return dag
