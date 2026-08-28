"""Configuration and feature schema for Phase 2 causal discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


NODE_FEATURES = [
    "inbound_packets",
    "inbound_bytes_proxy",
    "outbound_packets",
    "outbound_bytes_proxy",
    "mean_packet_size_proxy",
    "mean_interarrival_time",
    "peer_count",
]

EDGE_AGGREGATE_FEATURES = [
    "packet_count",
    "byte_count_proxy",
    "mean_packet_size_proxy",
    "std_packet_size_proxy",
    "mean_interarrival_time",
    "std_interarrival_time",
]

CAUSAL_FEATURES = [
    "total_packets",
    "total_bytes_proxy",
    "packet_rate",
    "byte_rate_proxy",
    "mean_packet_size_proxy",
    "std_packet_size_proxy",
    "mean_interarrival_time",
    "std_interarrival_time",
    "active_nodes",
    "active_edges",
    "average_degree",
    "max_degree",
    "average_peer_count",
]

METADATA_COLUMNS = ["split", "graph_file", "window_start"]

CAUSAL_FEATURE_DESCRIPTIONS = {
    "total_packets": "Sum of Phase 1 edge packet_count aggregates in the graph window.",
    "total_bytes_proxy": "Sum of Phase 1 edge byte_count_proxy aggregates in the graph window.",
    "packet_rate": "total_packets divided by configured window_seconds.",
    "byte_rate_proxy": "total_bytes_proxy divided by configured window_seconds.",
    "mean_packet_size_proxy": "Packet-count weighted mean of Phase 1 edge mean_packet_size_proxy.",
    "std_packet_size_proxy": "Packet-count weighted mean of Phase 1 edge std_packet_size_proxy.",
    "mean_interarrival_time": "Packet-count weighted mean of Phase 1 edge mean_interarrival_time.",
    "std_interarrival_time": "Packet-count weighted mean of Phase 1 edge std_interarrival_time.",
    "active_nodes": "Number of nodes present in the PyG snapshot.",
    "active_edges": "Number of directed edges present in the PyG snapshot.",
    "average_degree": "Directed graph total degree average, computed as 2 * active_edges / active_nodes.",
    "max_degree": "Maximum directed total degree among local nodes in the snapshot.",
    "average_peer_count": "Mean inverse-transformed Phase 1 node peer_count aggregate.",
}


@dataclass(frozen=True)
class Phase2Paths:
    root: Path
    processed_dir: Path
    phase1_artifacts_dir: Path
    interim_dir: Path
    phase2_artifacts_dir: Path
    phase2_reports_dir: Path
    causal_dataset_csv: Path


def project_path(root: Path, configured: str) -> Path:
    return (root / configured).resolve()


def require_keys(config: dict[str, Any], keys: set[str], context: str) -> None:
    missing = keys - set(config)
    if missing:
        raise ValueError(f"{context} config missing keys: {sorted(missing)}")


def build_paths(root: Path, config: dict[str, Any]) -> Phase2Paths:
    require_keys(
        config,
        {
            "processed_dir",
            "phase1_artifacts_dir",
            "interim_dir",
            "phase2_artifacts_dir",
            "phase2_reports_dir",
            "causal_dataset_csv",
        },
        "Phase 2",
    )
    return Phase2Paths(
        root=root,
        processed_dir=project_path(root, config["processed_dir"]),
        phase1_artifacts_dir=project_path(root, config["phase1_artifacts_dir"]),
        interim_dir=project_path(root, config["interim_dir"]),
        phase2_artifacts_dir=project_path(root, config["phase2_artifacts_dir"]),
        phase2_reports_dir=project_path(root, config["phase2_reports_dir"]),
        causal_dataset_csv=project_path(root, config["causal_dataset_csv"]),
    )


def validate_phase2_config(config: dict[str, Any]) -> None:
    require_keys(
        config,
        {
            "processed_dir",
            "phase1_artifacts_dir",
            "interim_dir",
            "phase2_artifacts_dir",
            "phase2_reports_dir",
            "causal_dataset_csv",
            "random_seed",
            "window_seconds",
            "graph_splits",
            "dataset",
            "reports",
            "discovery",
            "constraints",
            "stability",
            "scm",
        },
        "Phase 2",
    )
    if config["window_seconds"] <= 0:
        raise ValueError("window_seconds must be positive.")
    discovery = config["discovery"]
    require_keys(discovery, {"method", "edge_weight_threshold", "notears"}, "discovery")
    if discovery["method"] != "notears":
        raise ValueError("Only discovery.method='notears' is implemented in Phase 2.")
    stability = config["stability"]
    require_keys(
        stability,
        {"enabled", "n_bootstraps", "subsample_fraction", "random_seeds", "edge_weight_threshold", "minimum_stability"},
        "stability",
    )
    if not 0 < stability["subsample_fraction"] <= 1:
        raise ValueError("stability.subsample_fraction must be in (0, 1].")
    if not 0 <= stability["minimum_stability"] <= 1:
        raise ValueError("stability.minimum_stability must be in [0, 1].")
