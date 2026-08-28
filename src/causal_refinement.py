"""Apply explicit domain constraints and stability filtering to candidate DAGs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.causal_validation import adjacency_to_dag, validate_dag


def _edge_lookup(features: list[str]) -> dict[str, int]:
    return {feature: index for index, feature in enumerate(features)}


def apply_constraints(adjacency: np.ndarray, features: list[str], constraints: dict[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    refined = adjacency.copy()
    index = _edge_lookup(features)
    log: list[dict[str, Any]] = []
    for edge in constraints.get("forbidden_edges", []):
        source, target = edge["source"], edge["target"]
        if source not in index or target not in index:
            raise ValueError(f"Forbidden edge references unknown feature: {source} -> {target}")
        i, j = index[source], index[target]
        if refined[i, j] != 0:
            log.append({"action": "remove_forbidden_edge", "source": source, "target": target, "previous_weight": float(refined[i, j])})
            refined[i, j] = 0.0
    for edge in constraints.get("required_edges", []):
        source, target = edge["source"], edge["target"]
        if source not in index or target not in index:
            raise ValueError(f"Required edge references unknown feature: {source} -> {target}")
        i, j = index[source], index[target]
        weight = float(edge.get("weight", refined[i, j] if refined[i, j] != 0 else 1.0))
        previous = float(refined[i, j])
        refined[i, j] = weight
        dag = adjacency_to_dag(refined, features)
        try:
            validate_dag(dag, features)
        except ValueError as exc:
            refined[i, j] = previous
            raise ValueError(f"Required edge would make graph invalid: {source} -> {target}") from exc
        log.append({"action": "add_required_edge", "source": source, "target": target, "previous_weight": previous, "new_weight": weight})
    dag = adjacency_to_dag(refined, features)
    validate_dag(dag, features)
    return refined, log


def apply_stability_filter(adjacency: np.ndarray, features: list[str], stability: dict[str, Any], minimum_stability: float) -> tuple[np.ndarray, list[dict[str, Any]]]:
    refined = adjacency.copy()
    log: list[dict[str, Any]] = []
    for i, source in enumerate(features):
        for j, target in enumerate(features):
            if i == j or refined[i, j] == 0:
                continue
            key = f"{source}->{target}"
            frequency = float(stability.get("edges", {}).get(key, {}).get("selection_frequency", 0.0))
            if frequency < minimum_stability:
                log.append({"action": "remove_unstable_edge", "source": source, "target": target, "weight": float(refined[i, j]), "selection_frequency": frequency})
                refined[i, j] = 0.0
    dag = adjacency_to_dag(refined, features)
    validate_dag(dag, features)
    return refined, log


def write_refinement_log(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(entries, handle, indent=2)
