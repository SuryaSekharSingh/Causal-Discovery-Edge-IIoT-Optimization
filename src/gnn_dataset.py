"""Strict Phase 3 joins between PyG snapshots and Phase 2 causal targets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

SPLITS = ("train", "validation", "test")
METADATA_COLUMNS = ("split", "graph_file", "window_start")


@dataclass(frozen=True)
class NormalizationStats:
    causal_mean: np.ndarray
    causal_scale: np.ndarray
    graph_size_mean: np.ndarray
    graph_size_scale: np.ndarray

    def to_dict(self, causal_features: list[str]) -> dict[str, Any]:
        return {
            "fitted_on_split": "train",
            "causal_features": causal_features,
            "causal_mean": self.causal_mean.tolist(),
            "causal_scale": self.causal_scale.tolist(),
            "graph_size_features": ["log1p_node_count", "log1p_edge_count"],
            "graph_size_mean": self.graph_size_mean.tolist(),
            "graph_size_scale": self.graph_size_scale.tolist(),
        }


@dataclass
class Phase3DataBundle:
    datasets: dict[str, "GraphCausalDataset"]
    causal_features: list[str]
    normalization: NormalizationStats
    node_dim: int
    edge_dim: int
    split_counts: dict[str, int]
    temporal_pair_counts: dict[str, int]


@dataclass
class _PreparedRecord:
    graph: Any
    split: str
    graph_file: str
    window_start: str
    current_raw: np.ndarray
    next_raw: np.ndarray | None = None


class GraphCausalDataset(Dataset):
    """In-memory validated graph snapshots with standardized causal targets."""

    def __init__(
        self,
        records: list[_PreparedRecord],
        normalization: NormalizationStats,
    ) -> None:
        self.records = records
        self.normalization = normalization

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        graph = record.graph.clone()
        current_raw = record.current_raw.astype(np.float32, copy=False)
        current = (current_raw - self.normalization.causal_mean) / self.normalization.causal_scale
        if record.next_raw is None:
            next_raw = np.zeros_like(current_raw)
            next_target = np.zeros_like(current_raw)
            next_mask = 0.0
        else:
            next_raw = record.next_raw.astype(np.float32, copy=False)
            next_target = (next_raw - self.normalization.causal_mean) / self.normalization.causal_scale
            next_mask = 1.0
        graph.y_current = torch.tensor(current, dtype=torch.float32).view(1, -1)
        graph.y_next = torch.tensor(next_target, dtype=torch.float32).view(1, -1)
        graph.y_current_raw = torch.tensor(current_raw, dtype=torch.float32).view(1, -1)
        graph.y_next_raw = torch.tensor(next_raw, dtype=torch.float32).view(1, -1)
        graph.next_mask = torch.tensor([next_mask], dtype=torch.float32)
        graph.graph_file = record.graph_file
        graph.split_name = record.split
        return graph


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required Phase 2 artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_causal_feature_order(phase2_artifacts_dir: Path, phase2_reports_dir: Path) -> list[str]:
    dag_payload = _read_json(phase2_artifacts_dir / "refined_dag.json")
    decisions = _read_json(phase2_reports_dir / "causal_variable_decisions.json")
    dag_features = dag_payload.get("metadata", {}).get("features")
    used_features = decisions.get("used_features")
    if not isinstance(dag_features, list) or not dag_features:
        raise ValueError("refined_dag.json does not declare a non-empty metadata.features list.")
    if dag_features != used_features:
        raise ValueError("Phase 2 refined DAG and variable-decision feature orders do not match.")
    dag = dag_payload.get("dag", {})
    if set(dag) != set(dag_features):
        raise ValueError("Phase 2 refined DAG nodes do not match its declared features.")
    return [str(feature) for feature in dag_features]


def _load_graph(path: Path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _validate_graph(graph: Any, path: Path, node_dim: int, edge_dim: int) -> None:
    required = ("x", "edge_index", "edge_attr", "node_ids", "window_start")
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise ValueError(f"Graph {path} is missing attributes: {missing}")
    if graph.x.ndim != 2 or graph.x.shape[1] != node_dim:
        raise ValueError(f"Graph {path} has node feature shape {tuple(graph.x.shape)}; expected [N, {node_dim}].")
    if graph.edge_attr.ndim != 2 or graph.edge_attr.shape[1] != edge_dim:
        raise ValueError(f"Graph {path} has edge feature shape {tuple(graph.edge_attr.shape)}; expected [E, {edge_dim}].")
    if graph.edge_index.ndim != 2 or graph.edge_index.shape[0] != 2:
        raise ValueError(f"Graph {path} has invalid edge_index shape {tuple(graph.edge_index.shape)}.")
    if graph.edge_index.shape[1] != graph.edge_attr.shape[0]:
        raise ValueError(f"Graph {path} has inconsistent edge_index and edge_attr lengths.")
    if graph.node_ids.ndim != 1 or graph.node_ids.shape[0] != graph.x.shape[0]:
        raise ValueError(f"Graph {path} has inconsistent node_ids and x lengths.")
    if graph.x.shape[0] == 0:
        raise ValueError(f"Graph {path} contains no nodes.")
    if graph.edge_index.dtype != torch.long:
        raise ValueError(f"Graph {path} edge_index must use torch.int64.")
    if not torch.is_floating_point(graph.x) or not torch.is_floating_point(graph.edge_attr):
        raise ValueError(f"Graph {path} node and edge features must be floating point.")
    if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.edge_attr).all():
        raise ValueError(f"Graph {path} contains non-finite node or edge features.")
    if graph.edge_index.numel():
        if int(graph.edge_index.min()) < 0 or int(graph.edge_index.max()) >= graph.x.shape[0]:
            raise ValueError(f"Graph {path} contains an out-of-range edge index.")
    if len(torch.unique(graph.node_ids)) != len(graph.node_ids):
        raise ValueError(f"Graph {path} contains duplicate node_ids.")


def _safe_scale(values: np.ndarray) -> np.ndarray:
    scale = np.std(values, axis=0, ddof=0)
    return np.where(scale < 1e-8, 1.0, scale)


def _fit_normalization(train_records: list[_PreparedRecord]) -> NormalizationStats:
    causal = np.vstack([record.current_raw for record in train_records]).astype(np.float64)
    graph_sizes = np.array(
        [[np.log1p(record.graph.num_nodes), np.log1p(record.graph.num_edges)] for record in train_records],
        dtype=np.float64,
    )
    return NormalizationStats(
        causal_mean=np.mean(causal, axis=0).astype(np.float32),
        causal_scale=_safe_scale(causal).astype(np.float32),
        graph_size_mean=np.mean(graph_sizes, axis=0).astype(np.float32),
        graph_size_scale=_safe_scale(graph_sizes).astype(np.float32),
    )


def build_phase3_data(root: Path, config: dict[str, Any]) -> Phase3DataBundle:
    processed_dir = (root / config["processed_dir"]).resolve()
    causal_csv = (root / config["causal_dataset_csv"]).resolve()
    phase2_artifacts = (root / config["phase2_artifacts_dir"]).resolve()
    phase2_reports = (root / config["phase2_reports_dir"]).resolve()
    causal_features = load_causal_feature_order(phase2_artifacts, phase2_reports)
    if not causal_csv.exists():
        raise FileNotFoundError(f"Missing Phase 2 causal dataset: {causal_csv}")

    frame = pd.read_csv(causal_csv)
    required_columns = set(METADATA_COLUMNS) | set(causal_features)
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Causal dataset is missing columns: {sorted(missing_columns)}")
    if frame.duplicated(["split", "graph_file"]).any():
        raise ValueError("Causal dataset contains duplicate (split, graph_file) keys.")
    if not set(frame["split"]).issubset(SPLITS):
        raise ValueError(f"Causal dataset contains unsupported splits: {sorted(set(frame['split']) - set(SPLITS))}")
    numeric = frame[causal_features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError("Causal dataset contains missing or non-finite selected feature values.")
    frame = frame.copy()
    frame[causal_features] = numeric
    frame["_parsed_time"] = pd.to_datetime(frame["window_start"], errors="coerce")
    if frame["_parsed_time"].isna().any():
        raise ValueError("Causal dataset contains invalid window_start timestamps.")

    node_dim = int(config["expected_schema"]["node_features"])
    edge_dim = int(config["expected_schema"]["edge_features"])
    records_by_split: dict[str, list[_PreparedRecord]] = {}
    pair_counts: dict[str, int] = {}
    expected_keys: set[tuple[str, str]] = set()

    for split in SPLITS:
        split_dir_name = config["graph_splits"][split]
        graph_paths = sorted((processed_dir / split_dir_name).glob("*.pt"))
        if not graph_paths:
            raise FileNotFoundError(f"No graph snapshots found for {split}: {processed_dir / split_dir_name}")
        split_rows = frame.loc[frame["split"].eq(split)].set_index("graph_file", drop=False)
        graph_names = {path.name for path in graph_paths}
        row_names = set(split_rows.index.astype(str))
        if graph_names != row_names:
            missing_rows = sorted(graph_names - row_names)
            extra_rows = sorted(row_names - graph_names)
            raise ValueError(
                f"Graph/causal row mismatch for {split}: missing_rows={missing_rows[:5]}, extra_rows={extra_rows[:5]}"
            )

        records: list[_PreparedRecord] = []
        for graph_path in graph_paths:
            graph = _load_graph(graph_path)
            _validate_graph(graph, graph_path, node_dim, edge_dim)
            row = split_rows.loc[graph_path.name]
            graph_time = pd.Timestamp(str(graph.window_start))
            row_time = pd.Timestamp(row["_parsed_time"])
            if graph_time != row_time:
                raise ValueError(
                    f"Timestamp mismatch for {split}/{graph_path.name}: graph={graph_time}, causal_row={row_time}"
                )
            records.append(
                _PreparedRecord(
                    graph=graph,
                    split=split,
                    graph_file=graph_path.name,
                    window_start=graph_time.isoformat(),
                    current_raw=row[causal_features].to_numpy(dtype=np.float32),
                )
            )
            expected_keys.add((split, graph_path.name))

        records.sort(key=lambda record: (pd.Timestamp(record.window_start), record.graph_file))
        pairs = 0
        expected_gap = pd.Timedelta(seconds=float(config["window_seconds"]))
        for current, following in zip(records, records[1:]):
            if pd.Timestamp(following.window_start) - pd.Timestamp(current.window_start) == expected_gap:
                current.next_raw = following.current_raw.copy()
                pairs += 1
        records_by_split[split] = records
        pair_counts[split] = pairs

    csv_keys = set(zip(frame["split"].astype(str), frame["graph_file"].astype(str)))
    if csv_keys != expected_keys:
        raise ValueError("Causal dataset and graph inventory do not have an exact one-to-one key match.")

    normalization = _fit_normalization(records_by_split["train"])
    datasets = {
        split: GraphCausalDataset(records, normalization)
        for split, records in records_by_split.items()
    }
    return Phase3DataBundle(
        datasets=datasets,
        causal_features=causal_features,
        normalization=normalization,
        node_dim=node_dim,
        edge_dim=edge_dim,
        split_counts={split: len(records) for split, records in records_by_split.items()},
        temporal_pair_counts=pair_counts,
    )
