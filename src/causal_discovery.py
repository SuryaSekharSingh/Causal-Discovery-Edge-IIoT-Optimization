"""Configurable causal discovery algorithms for Phase 2."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

from src.causal_validation import adjacency_to_dag, validate_dag

LOG = logging.getLogger("phase2.discovery")


def _loss(W: np.ndarray, X: np.ndarray, loss_type: str) -> tuple[float, np.ndarray]:
    M = X @ W
    if loss_type != "l2":
        raise ValueError("Only NOTEARS l2 loss is implemented.")
    residual = X - M
    loss = 0.5 / X.shape[0] * np.square(residual).sum()
    gradient = -1.0 / X.shape[0] * X.T @ residual
    return loss, gradient


def _h(W: np.ndarray) -> tuple[float, np.ndarray]:
    E = expm(W * W)
    h_value = float(np.trace(E) - W.shape[0])
    gradient = E.T * W * 2
    return h_value, gradient


def notears_linear(X: np.ndarray, lambda1: float, loss_type: str, max_iter: int, h_tol: float, rho_max: float, w_threshold: float) -> np.ndarray:
    """Learn a weighted DAG with the linear NOTEARS augmented Lagrangian."""
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    d = X.shape[1]
    w_est = np.zeros(2 * d * d)
    rho, alpha, h_value = 1.0, 0.0, np.inf
    bnds = []
    for i in range(d):
        for j in range(d):
            bnds.append((0.0, 0.0) if i == j else (0.0, None))
    bnds = bnds + bnds

    def _adj(w: np.ndarray) -> np.ndarray:
        return (w[: d * d] - w[d * d :]).reshape(d, d)

    def _func(w: np.ndarray) -> tuple[float, np.ndarray]:
        W = _adj(w)
        loss, gradient_loss = _loss(W, X, loss_type)
        h_current, gradient_h = _h(W)
        objective = loss + 0.5 * rho * h_current * h_current + alpha * h_current + lambda1 * w.sum()
        gradient = gradient_loss + (rho * h_current + alpha) * gradient_h
        return objective, np.concatenate((gradient + lambda1, -gradient + lambda1), axis=None)

    for _ in range(max_iter):
        while rho < rho_max:
            result = minimize(_func, w_est, method="L-BFGS-B", jac=True, bounds=bnds)
            candidate = result.x
            h_new, _ = _h(_adj(candidate))
            if h_new > 0.25 * h_value:
                rho *= 10
            else:
                break
        w_est = candidate
        h_value = h_new
        alpha += rho * h_value
        if h_value <= h_tol or rho >= rho_max:
            break

    W = _adj(w_est)
    W[np.abs(W) < w_threshold] = 0.0
    np.fill_diagonal(W, 0.0)
    return W


def threshold_to_dag(weights: np.ndarray, threshold: float) -> np.ndarray:
    adjacency = weights.copy()
    adjacency[np.abs(adjacency) < threshold] = 0.0
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def discover_candidate_dag(train_frame: pd.DataFrame, features: list[str], config: dict[str, Any]) -> tuple[np.ndarray, dict[str, list[dict[str, float]]]]:
    method = config["discovery"]["method"]
    if method != "notears":
        raise ValueError(f"Unsupported causal discovery method: {method}")
    X = train_frame[features].to_numpy(dtype=float)
    X = (X - X.mean(axis=0)) / np.where(X.std(axis=0, ddof=0) == 0, 1.0, X.std(axis=0, ddof=0))
    params = config["discovery"]["notears"]
    weights = notears_linear(X, **params)
    adjacency = threshold_to_dag(weights, config["discovery"]["edge_weight_threshold"])
    dag = adjacency_to_dag(adjacency, features)
    validate_dag(dag, features)
    return adjacency, dag


def write_dag_json(path: Path, dag: dict[str, list[dict[str, float]]], metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "dag": dag}, handle, indent=2)


def draw_dag(path: Path, dag: dict[str, list[dict[str, float]]], title: str) -> None:
    import matplotlib.pyplot as plt

    graph = nx.DiGraph()
    for source, edges in dag.items():
        graph.add_node(source)
        for edge in edges:
            graph.add_edge(source, edge["target"], weight=edge["weight"])
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(graph, seed=42) if graph.number_of_edges() else nx.circular_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=1200, node_color="#d9ead3", edgecolors="#274e13")
    nx.draw_networkx_labels(graph, pos, font_size=8)
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="-|>", arrowsize=14, edge_color="#555555")
    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
