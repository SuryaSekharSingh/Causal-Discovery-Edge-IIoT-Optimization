"""Build graph-level causal observations from Phase 1 PyG snapshots."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import load

from src.phase2_schema import (
    CAUSAL_FEATURE_DESCRIPTIONS,
    CAUSAL_FEATURES,
    EDGE_AGGREGATE_FEATURES,
    METADATA_COLUMNS,
    NODE_FEATURES,
    Phase2Paths,
)

LOG = logging.getLogger("phase2.dataset")


def _sorted_graphs(split_dir: Path) -> list[Path]:
    paths = sorted(split_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt graph snapshots found in {split_dir}")
    return paths


def _load_graph(path: Path):
    return torch.load(path, weights_only=False, map_location="cpu")


def _inverse_features(values: np.ndarray, scaler, expected_width: int, feature_kind: str) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] < expected_width:
        raise ValueError(f"{feature_kind} feature tensor has shape {values.shape}; expected at least {expected_width} columns.")
    return scaler.inverse_transform(values)[:, :expected_width]


def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    if values.size == 0 or weights.sum() <= 0:
        return 0.0
    return float(np.average(values, weights=np.maximum(weights, 0.0)))


def graph_to_observation(graph, graph_path: Path, split: str, scalers: dict[str, Any], window_seconds: float) -> dict[str, Any]:
    x = graph.x.detach().cpu().numpy().astype(float)
    edge_attr = graph.edge_attr.detach().cpu().numpy().astype(float)
    edge_index = graph.edge_index.detach().cpu().numpy().astype(int)

    node_raw = _inverse_features(x, scalers["node_scaler"], len(NODE_FEATURES), "Node")
    edge_raw = _inverse_features(edge_attr, scalers["edge_scaler"], len(EDGE_AGGREGATE_FEATURES), "Edge")
    node_raw = np.maximum(node_raw, 0.0)
    edge_raw = np.maximum(edge_raw, 0.0)

    packet_count = edge_raw[:, 0] if len(edge_raw) else np.array([], dtype=float)
    byte_count = edge_raw[:, 1] if len(edge_raw) else np.array([], dtype=float)
    active_nodes = int(x.shape[0])
    active_edges = int(edge_attr.shape[0])

    degrees = np.zeros(active_nodes, dtype=float)
    if edge_index.size:
        for node in edge_index.reshape(-1):
            if 0 <= node < active_nodes:
                degrees[node] += 1

    total_packets = float(packet_count.sum())
    total_bytes = float(byte_count.sum())
    observation = {
        "split": split,
        "graph_file": graph_path.name,
        "window_start": str(getattr(graph, "window_start", "")),
        "total_packets": total_packets,
        "total_bytes_proxy": total_bytes,
        "packet_rate": total_packets / window_seconds,
        "byte_rate_proxy": total_bytes / window_seconds,
        "mean_packet_size_proxy": _weighted_average(edge_raw[:, 2], packet_count) if len(edge_raw) else 0.0,
        "std_packet_size_proxy": _weighted_average(edge_raw[:, 3], packet_count) if len(edge_raw) else 0.0,
        "mean_interarrival_time": _weighted_average(edge_raw[:, 4], packet_count) if len(edge_raw) else 0.0,
        "std_interarrival_time": _weighted_average(edge_raw[:, 5], packet_count) if len(edge_raw) else 0.0,
        "active_nodes": active_nodes,
        "active_edges": active_edges,
        "average_degree": float(2 * active_edges / active_nodes) if active_nodes else 0.0,
        "max_degree": float(degrees.max()) if degrees.size else 0.0,
        "average_peer_count": float(np.mean(node_raw[:, 6])) if len(node_raw) else 0.0,
    }
    return observation


def build_causal_dataset(config: dict[str, Any], paths: Phase2Paths) -> pd.DataFrame:
    scaler_path = paths.phase1_artifacts_dir / "scaler.pkl"
    mapping_path = paths.phase1_artifacts_dir / "feature_mapping.json"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 scaler artifact: {scaler_path}")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing Phase 1 feature mapping artifact: {mapping_path}")
    scalers = load(scaler_path)

    rows: list[dict[str, Any]] = []
    for split_name, split_dir_name in config["graph_splits"].items():
        split_dir = paths.processed_dir / split_dir_name
        for graph_path in _sorted_graphs(split_dir):
            rows.append(graph_to_observation(_load_graph(graph_path), graph_path, split_name, scalers, config["window_seconds"]))

    dataset = pd.DataFrame(rows).sort_values(["split", "window_start", "graph_file"], kind="mergesort")
    dataset = dataset[METADATA_COLUMNS + CAUSAL_FEATURES]
    paths.interim_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(paths.causal_dataset_csv, index=False)
    LOG.info("Wrote causal dataset with %s observations to %s", len(dataset), paths.causal_dataset_csv)
    return dataset


def write_causal_feature_mapping(paths: Phase2Paths, config: dict[str, Any]) -> None:
    paths.phase2_artifacts_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "Phase 1 PyTorch Geometric graph snapshots",
        "labels_excluded": ["Attack_label", "Attack_type"],
        "identifiers_excluded": ["raw IP addresses", "node_ids", "edge_index"],
        "high_dimensional_edge_features_excluded": True,
        "window_seconds": config["window_seconds"],
        "features": {feature: CAUSAL_FEATURE_DESCRIPTIONS[feature] for feature in CAUSAL_FEATURES},
        "phase1_node_feature_order": NODE_FEATURES,
        "phase1_edge_aggregate_feature_order": EDGE_AGGREGATE_FEATURES,
    }
    with (paths.phase2_artifacts_dir / "causal_feature_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
