"""Reproducible Phase 1: Edge-IIoTset CSV to temporal PyG graph snapshots.

This module intentionally contains no GNN, causal discovery, SCM, RL, or
counterfactual code. Labels are audited but never placed in x or edge_attr.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np
import pandas as pd
import torch
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch_geometric.data import Data

from src.phase1_schema import CATEGORICAL, EDGE_FEATURE, EXPECTED_COLUMNS, feature_mapping

LOG = logging.getLogger("phase1")
SPLITS = ("train", "validation", "test")
PROGRESS_EVERY_CHUNKS = 10


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"source_csv", "raw_csv", "interim_csv", "processed_dir", "artifacts_dir", "reports_dir"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Config missing keys: {sorted(missing)}")
    if not 0 < config["train_fraction"] < 1 or not 0 < config["validation_fraction"] < 1:
        raise ValueError("Split fractions must be between zero and one.")
    if config["train_fraction"] + config["validation_fraction"] >= 1:
        raise ValueError("Train + validation fractions must be less than one.")
    return config


def project_path(root: Path, configured: str) -> Path:
    return (root / configured).resolve()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def parse_frame_time(values: pd.Series) -> pd.Series:
    """Parse both raw Edge-IIoTset timestamps and ISO timestamps written to interim CSV."""
    return pd.to_datetime(values.astype("string").str.strip(), format="mixed", errors="coerce")


def read_cleaned_chunks(cleaned_csv: Path, chunk_size: int):
    """Yield interim traffic with a guaranteed datetime timestamp column."""
    for chunk in pd.read_csv(cleaned_csv, chunksize=chunk_size, low_memory=False):
        chunk["frame.time"] = parse_frame_time(chunk["frame.time"])
        if chunk["frame.time"].isna().any():
            raise ValueError(f"Invalid timestamps found in cleaned CSV: {cleaned_csv}")
        yield chunk


def copy_raw_source(source: Path, raw_target: Path) -> None:
    raw_target.parent.mkdir(parents=True, exist_ok=True)
    if raw_target.exists():
        if file_digest(source) != file_digest(raw_target):
            raise FileExistsError(f"Raw target differs from source: {raw_target}")
        LOG.info("Raw CSV already present and unchanged: %s", raw_target)
        return
    LOG.info("Copying source CSV to immutable raw location: %s", raw_target)
    shutil.copy2(source, raw_target)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_schema(csv_path: Path) -> None:
    actual = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if actual != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(actual))
        unexpected = sorted(set(actual) - set(EXPECTED_COLUMNS))
        raise ValueError(f"Schema does not match verified Edge-IIoTset CSV. Missing={missing}; unexpected={unexpected}")


def numeric_columns() -> list[str]:
    return sorted(EDGE_FEATURE)


def infer_protocol(frame: pd.DataFrame) -> pd.Series:
    """Derive protocol family solely from verified packet fields; no source column is invented."""
    protocol = pd.Series("OTHER", index=frame.index, dtype="object")
    protocol = protocol.mask(pd.to_numeric(frame["arp.opcode"], errors="coerce").fillna(0).ne(0), "ARP")
    protocol = protocol.mask(pd.to_numeric(frame["icmp.checksum"], errors="coerce").fillna(0).ne(0), "ICMP")
    protocol = protocol.mask(pd.to_numeric(frame["tcp.dstport"], errors="coerce").fillna(0).ne(0), "TCP")
    protocol = protocol.mask(pd.to_numeric(frame["udp.port"], errors="coerce").fillna(0).ne(0), "UDP")
    protocol = protocol.mask(pd.to_numeric(frame["dns.qry.type"], errors="coerce").fillna(0).ne(0), "DNS")
    protocol = protocol.mask(pd.to_numeric(frame["http.content_length"], errors="coerce").fillna(0).ne(0), "HTTP")
    methods = frame["http.request.method"].fillna("").astype(str).str.strip()
    protocol = protocol.mask(~methods.isin({"", "0", "0.0", "nan", "None"}), "HTTP")
    protocol = protocol.mask(pd.to_numeric(frame["mbtcp.len"], errors="coerce").fillna(0).ne(0), "MBTCP")
    protocol = protocol.mask(pd.to_numeric(frame["mqtt.len"], errors="coerce").fillna(0).ne(0), "MQTT")
    return protocol


def audit_dataset(raw_csv: Path, chunk_size: int) -> dict:
    """Chunked, full-file audit so the 1GB CSV is not loaded at once."""
    started = time.perf_counter()
    LOG.info("Starting dataset audit: %s", raw_csv)
    rows = duplicates = 0
    missing = Counter()
    infinities = Counter()
    endpoints_src, endpoints_dst = set(), set()
    protocol_counts, label_counts = Counter(), Counter()
    timestamp_min = timestamp_max = None
    dtypes: dict[str, str] = {}
    numeric = numeric_columns()
    for chunk_number, chunk in enumerate(pd.read_csv(raw_csv, chunksize=chunk_size, low_memory=False), start=1):
        rows += len(chunk)
        dtypes.update({column: str(dtype) for column, dtype in chunk.dtypes.items()})
        missing.update(chunk.isna().sum().to_dict())
        for column in numeric:
            values = pd.to_numeric(chunk[column], errors="coerce").to_numpy(dtype=float, na_value=np.nan)
            infinities[column] += int(np.isinf(values).sum())
        duplicates += int(chunk.duplicated().sum())  # global removal is performed during cleaning
        endpoints_src.update(chunk["ip.src_host"].dropna().astype(str))
        endpoints_dst.update(chunk["ip.dst_host"].dropna().astype(str))
        protocol_counts.update(infer_protocol(chunk).value_counts().to_dict())
        label_counts.update(chunk["Attack_label"].fillna("<missing>").astype(str).value_counts().to_dict())
        times = parse_frame_time(chunk["frame.time"])
        valid = times.dropna()
        if not valid.empty:
            timestamp_min = valid.min() if timestamp_min is None else min(timestamp_min, valid.min())
            timestamp_max = valid.max() if timestamp_max is None else max(timestamp_max, valid.max())
        if chunk_number % PROGRESS_EVERY_CHUNKS == 0:
            LOG.info("Audit progress: %s chunks, %s rows scanned.", chunk_number, rows)
    LOG.info("Dataset audit complete: %s rows scanned in %.1fs.", rows, time.perf_counter() - started)
    return {
        "shape": [rows, len(EXPECTED_COLUMNS)], "columns": EXPECTED_COLUMNS, "dtypes_observed": dtypes,
        "missing_nan_counts": dict(missing), "infinity_counts_numeric_columns": dict(infinities),
        "duplicate_count_within_chunks": duplicates,
        "unique_source_endpoints": len(endpoints_src), "unique_destination_endpoints": len(endpoints_dst),
        "unique_endpoints": len(endpoints_src | endpoints_dst), "protocol_distribution": dict(protocol_counts),
        "attack_label_distribution": dict(label_counts), "timestamp_range": {"min": timestamp_min, "max": timestamp_max},
    }


def clean_csv(raw_csv: Path, output_csv: Path, chunk_size: int) -> dict:
    """Strictly remove any row with missing/invalid values and global exact duplicates."""
    started = time.perf_counter()
    LOG.info("Starting cleaning pass: %s", raw_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        output_csv.unlink()
    seen: set[int] = set()
    total = kept = nan_inf_removed = duplicate_removed = 0
    first = True
    numeric = numeric_columns()
    for chunk_number, chunk in enumerate(pd.read_csv(raw_csv, chunksize=chunk_size, low_memory=False), start=1):
        total += len(chunk)
        timestamps = parse_frame_time(chunk["frame.time"])
        invalid = chunk.isna().any(axis=1) | timestamps.isna()
        for column in numeric:
            values = pd.to_numeric(chunk[column], errors="coerce")
            invalid |= values.isna() | ~np.isfinite(values)
            chunk[column] = values
        nan_inf_removed += int(invalid.sum())
        clean = chunk.loc[~invalid].copy()
        clean["frame.time"] = timestamps.loc[~invalid]
        hashes = pd.util.hash_pandas_object(clean, index=False).astype("uint64")
        unique = ~(hashes.isin(seen) | hashes.duplicated(keep="first"))
        duplicate_removed += int((~unique).sum())
        seen.update(hashes.loc[unique].astype(int).tolist())
        clean = clean.loc[unique]
        clean.to_csv(output_csv, mode="w" if first else "a", index=False, header=first)
        first = False
        kept += len(clean)
        if chunk_number % PROGRESS_EVERY_CHUNKS == 0:
            LOG.info(
                "Cleaning progress: %s chunks, %s rows scanned, %s rows kept.",
                chunk_number, total, kept,
            )
    if kept == 0:
        raise ValueError("Cleaning removed every row. Inspect dataset_audit.json.")
    LOG.info(
        "Cleaning complete: %s rows kept from %s rows in %.1fs.",
        kept, total, time.perf_counter() - started,
    )
    return {"rows_before": total, "rows_after": kept, "nan_or_inf_rows_removed": nan_inf_removed,
            "exact_duplicates_removed": duplicate_removed}


def assign_strata(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, bins: int) -> pd.Series:
    elapsed = (frame["frame.time"] - start).dt.total_seconds()
    duration = max((end - start).total_seconds(), 1.0)
    time_bin = np.minimum((elapsed / duration * bins).astype(int), bins - 1)
    return time_bin.astype(str) + "|" + frame["Attack_label"].astype(str) + "|" + infer_protocol(frame)


def apportion(counts: Counter, desired: int) -> dict[str, int]:
    total = sum(counts.values())
    if desired >= total:
        return dict(counts)
    exact = {key: value * desired / total for key, value in counts.items()}
    quotas = {key: int(value) for key, value in exact.items()}
    remainder = desired - sum(quotas.values())
    for key in sorted(counts, key=lambda item: (exact[item] - quotas[item], item), reverse=True)[:remainder]:
        quotas[key] += 1
    return quotas


def sample_cleaned(cleaned_csv: Path, audit: dict, config: dict) -> pd.DataFrame:
    """Deterministic stratified sample across time, label, and derived protocol.

    The first pass counts strata. The second pass keeps only the best random
    priorities per stratum at chunk level. This avoids iterrows()/to_dict()
    over the full cleaned CSV, which was the main runtime bottleneck.
    """
    started = time.perf_counter()
    LOG.info("Starting deterministic sampling from cleaned traffic: %s", cleaned_csv)
    start, end = pd.Timestamp(audit["timestamp_range"]["min"]), pd.Timestamp(audit["timestamp_range"]["max"])
    counts = Counter()
    for chunk_number, chunk in enumerate(read_cleaned_chunks(cleaned_csv, config["chunk_size"]), start=1):
        counts.update(assign_strata(chunk, start, end, config["sampling_time_bins"]).value_counts().to_dict())
        if chunk_number % PROGRESS_EVERY_CHUNKS == 0:
            LOG.info("Sampling scan progress: %s chunks, %s rows counted.", chunk_number, sum(counts.values()))
    requested = config.get("sample_size")
    if requested is None or requested >= sum(counts.values()):
        LOG.info("Using all %s cleaned rows.", sum(counts.values()))
        return pd.concat(read_cleaned_chunks(cleaned_csv, config["chunk_size"]), ignore_index=True)
    quotas = apportion(counts, int(requested))
    # Each bucket contains only a small candidate DataFrame. Keeping complete
    # row dictionaries for every reservoir entry is substantially slower and
    # uses more Python-object memory than vectorized pandas operations.
    candidates: dict[str, list[pd.DataFrame]] = defaultdict(list)
    rng = np.random.default_rng(config["random_seed"])
    rows_seen = 0
    for chunk_number, chunk in enumerate(read_cleaned_chunks(cleaned_csv, config["chunk_size"]), start=1):
        rows_seen += len(chunk)
        work = chunk.copy()
        work["_stratum"] = assign_strata(work, start, end, config["sampling_time_bins"]).to_numpy()
        work["_priority"] = rng.random(len(work))
        for stratum, indices in work.groupby("_stratum", sort=False).groups.items():
            quota = quotas.get(stratum, 0)
            if quota <= 0:
                continue
            bucket = work.loc[indices]
            # Keep only the chunk's best candidates; later chunks are merged
            # and reduced to the exact global quota.
            candidates[stratum].append(bucket.nsmallest(min(quota, len(bucket)), "_priority"))
        if chunk_number % PROGRESS_EVERY_CHUNKS == 0:
            selected_so_far = sum(sum(len(part) for part in parts) for parts in candidates.values())
            LOG.info("Sampling reservoir progress: %s chunks, %s rows seen, %s rows selected.", chunk_number, rows_seen, selected_so_far)
    buckets = []
    for stratum, parts in candidates.items():
        bucket = pd.concat(parts, ignore_index=True)
        buckets.append(bucket.nsmallest(quotas[stratum], "_priority"))
    sample = pd.concat(buckets, ignore_index=True).drop(columns=["_stratum", "_priority"])
    sample = sample.sort_values("frame.time", kind="mergesort").reset_index(drop=True)
    LOG.info(
        "Selected deterministic sample of %s rows from %s cleaned rows in %.1fs.",
        len(sample), sum(counts.values()), time.perf_counter() - started,
    )
    return sample


def split_periods(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict[str, pd.Timestamp]]:
    frame = frame.sort_values("frame.time", kind="mergesort").reset_index(drop=True)
    train_end_idx = max(1, int(len(frame) * config["train_fraction"]))
    validation_end_idx = max(train_end_idx + 1, int(len(frame) * (config["train_fraction"] + config["validation_fraction"])))
    validation_end_idx = min(validation_end_idx, len(frame))
    train_end = frame.iloc[train_end_idx - 1]["frame.time"]
    validation_end = frame.iloc[validation_end_idx - 1]["frame.time"]
    later_than_train = frame.loc[frame["frame.time"].gt(train_end), "frame.time"]
    if later_than_train.empty:
        raise ValueError("Cannot create chronological splits: all sampled rows share one timestamp.")
    if validation_end <= train_end:
        validation_end = later_than_train.iloc[0]
    split = np.select(
        [frame["frame.time"].le(train_end), frame["frame.time"].le(validation_end)],
        ["train", "validation"],
        default="test",
    )
    frame["_split"] = split
    if not frame["_split"].eq("test").any():
        raise ValueError("Cannot create a test period after respecting timestamp boundaries.")
    boundaries = {"train_end": train_end, "validation_end": validation_end}
    return frame, boundaries


def prepare_record_features(frame: pd.DataFrame, config: dict):
    categorical = sorted(CATEGORICAL | {"_protocol"})
    numeric = sorted(EDGE_FEATURE)
    frame = frame.copy()
    frame["_protocol"] = infer_protocol(frame)
    eligible_categorical = [column for column in categorical if frame.loc[frame["_split"].eq("train"), column].nunique() <= config["max_categories"]]
    omitted = sorted(set(categorical) - set(eligible_categorical))

    # Edge-IIoTset nominal fields can contain values such as 0 alongside
    # textual values. Normalize only model categorical inputs so OneHotEncoder
    # receives one consistent type; endpoint IDs and labels remain separate.
    for column in eligible_categorical:
        frame[column] = frame[column].astype("string").fillna("<missing>")

    transformer = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), eligible_categorical),
    ], remainder="drop", verbose_feature_names_out=False)
    train = frame.loc[frame["_split"].eq("train")]
    transformer.fit(train[numeric + eligible_categorical])
    transformed = transformer.transform(frame[numeric + eligible_categorical]).astype(np.float32)
    return frame, transformed, transformer, numeric, eligible_categorical, omitted


def packet_size(frame: pd.DataFrame) -> pd.Series:
    available = ["tcp.len", "mqtt.len", "mbtcp.len", "http.content_length"]
    return frame[available].apply(pd.to_numeric, errors="coerce").fillna(0).max(axis=1)


def build_raw_graphs(frame: pd.DataFrame, record_features: np.ndarray, endpoint_map: dict[str, int], window_seconds: int):
    start = frame["frame.time"].min().floor("s")
    frame = frame.copy()
    frame["_window"] = ((frame["frame.time"] - start).dt.total_seconds() // window_seconds).astype(int)
    frame["_packet_size"] = packet_size(frame)
    frame["_row"] = np.arange(len(frame))
    graphs = []
    # Split is part of the grouping so a graph can never contain traffic from a
    # future validation/test period merely because a fixed window crosses a boundary.
    for (split, window), group in frame.groupby(["_split", "_window"], sort=True):
        group = group.sort_values("frame.time", kind="mergesort").copy()
        group["_iat"] = group.groupby(["ip.src_host", "ip.dst_host"])["frame.time"].diff().dt.total_seconds().fillna(0)
        graph = nx.DiGraph(window=int(window), split=str(split), start_time=start + pd.Timedelta(seconds=int(window) * window_seconds))
        involved = sorted(set(group["ip.src_host"].astype(str)) | set(group["ip.dst_host"].astype(str)), key=endpoint_map.__getitem__)

        # Compute node aggregates once per window. The previous implementation
        # filtered the complete window separately for every endpoint.
        inbound = group.groupby("ip.dst_host", sort=False).agg(
            inbound_packets=("_packet_size", "size"),
            inbound_bytes=("_packet_size", "sum"),
        )
        outbound = group.groupby("ip.src_host", sort=False).agg(
            outbound_packets=("_packet_size", "size"),
            outbound_bytes=("_packet_size", "sum"),
        )
        related_rows = pd.concat([
            group[["ip.src_host", "_packet_size", "_iat"]].rename(columns={"ip.src_host": "endpoint"}),
            group[["ip.dst_host", "_packet_size", "_iat"]].rename(columns={"ip.dst_host": "endpoint"}),
        ], ignore_index=True)
        related = related_rows.groupby("endpoint", sort=False).agg(
            mean_packet_size=("_packet_size", "mean"),
            mean_interarrival=("_iat", "mean"),
        )
        peer_rows = pd.concat([
            group[["ip.src_host", "ip.dst_host"]].rename(columns={"ip.src_host": "endpoint", "ip.dst_host": "peer"}),
            group[["ip.dst_host", "ip.src_host"]].rename(columns={"ip.dst_host": "endpoint", "ip.src_host": "peer"}),
        ], ignore_index=True)
        peer_counts = peer_rows.groupby("endpoint", sort=False)["peer"].nunique()
        for endpoint in involved:
            in_stats = inbound.loc[endpoint] if endpoint in inbound.index else None
            out_stats = outbound.loc[endpoint] if endpoint in outbound.index else None
            related_stats = related.loc[endpoint]
            node_raw = np.array([
                0 if in_stats is None else in_stats["inbound_packets"],
                0.0 if in_stats is None else in_stats["inbound_bytes"],
                0 if out_stats is None else out_stats["outbound_packets"],
                0.0 if out_stats is None else out_stats["outbound_bytes"],
                related_stats["mean_packet_size"], related_stats["mean_interarrival"],
                peer_counts.get(endpoint, 0),
            ], dtype=np.float32)
            graph.add_node(endpoint_map[endpoint], raw=node_raw)
        for (source, destination), edge in group.groupby(["ip.src_host", "ip.dst_host"], sort=True):
            rows = edge["_row"].to_numpy(dtype=int)
            transformed_mean = record_features[rows].mean(axis=0)
            protocol_counts = edge["_protocol"].value_counts().to_dict()
            raw = np.concatenate(([len(edge), edge["_packet_size"].sum(), edge["_packet_size"].mean(), edge["_packet_size"].std(ddof=0),
                                   edge["_iat"].mean(), edge["_iat"].std(ddof=0)], transformed_mean)).astype(np.float32)
            graph.add_edge(endpoint_map[str(source)], endpoint_map[str(destination)], raw=raw, protocol_counts=protocol_counts)
        graphs.append(graph)
    return graphs


def save_graphs(graphs: Iterable[nx.DiGraph], boundaries: dict[str, pd.Timestamp], output_dir: Path, overwrite: bool) -> tuple[list[dict], StandardScaler, StandardScaler]:
    if output_dir.exists() and any(output_dir.rglob("*.pt")) and not overwrite:
        raise FileExistsError(f"Output exists: {output_dir}. Set overwrite_outputs=true to replace it.")
    output_dir.mkdir(parents=True, exist_ok=True)
    node_scaler, edge_scaler = StandardScaler(), StandardScaler()
    graphs = list(graphs)
    def split_for(graph):
        return graph.graph["split"]
    train_nodes = np.vstack([attrs["raw"] for graph in graphs if split_for(graph) == "train" for _, attrs in graph.nodes(data=True)])
    train_edges = np.vstack([attrs["raw"] for graph in graphs if split_for(graph) == "train" for _, _, attrs in graph.edges(data=True)])
    node_scaler.fit(train_nodes)
    edge_scaler.fit(train_edges)
    statistics = []
    counters = Counter()
    for graph in graphs:
        split = split_for(graph)
        (output_dir / split).mkdir(parents=True, exist_ok=True)
        sequence = counters[split]
        counters[split] += 1
        node_ids = np.array(sorted(graph.nodes()), dtype=np.int64)
        local = {global_id: index for index, global_id in enumerate(node_ids)}
        node_raw = np.vstack([graph.nodes[node]["raw"] for node in node_ids])
        edges = list(graph.edges(data=True))
        edge_index = np.array([[local[source], local[target]] for source, target, _ in edges], dtype=np.int64).T
        edge_raw = np.vstack([attrs["raw"] for _, _, attrs in edges])
        x, edge_attr = node_scaler.transform(node_raw).astype(np.float32), edge_scaler.transform(edge_raw).astype(np.float32)
        validate_graph(node_ids, edge_index, x, edge_attr)
        data = Data(x=torch.from_numpy(x), edge_index=torch.from_numpy(edge_index), edge_attr=torch.from_numpy(edge_attr),
                    node_ids=torch.from_numpy(node_ids), window_start=graph.graph["start_time"].isoformat())
        torch.save(data, output_dir / split / f"graph_{sequence:03d}.pt")
        protocol_counts = Counter()
        for _, _, attrs in edges:
            protocol_counts.update(attrs["protocol_counts"])
        nodes, edge_count = graph.number_of_nodes(), graph.number_of_edges()
        statistics.append({"split": split, "graph": f"graph_{sequence:03d}.pt", "window_start": graph.graph["start_time"],
                           "nodes": nodes, "edges": edge_count, "average_degree": 2 * edge_count / nodes if nodes else 0,
                           "density": nx.density(graph), "isolated_nodes": len(list(nx.isolates(graph))),
                           "protocol_distribution": json.dumps(dict(sorted(protocol_counts.items())))})
    return statistics, node_scaler, edge_scaler


def validate_graph(node_ids: np.ndarray, edge_index: np.ndarray, x: np.ndarray, edge_attr: np.ndarray) -> None:
    if len(node_ids) == 0 or edge_index.shape[0] != 2:
        raise ValueError("Graph must have nodes and a [2, E] edge index.")
    if edge_index.size and (edge_index.min() < 0 or edge_index.max() >= len(node_ids)):
        raise ValueError("Edge index references a nonexistent local node.")
    if len(set(node_ids.tolist())) != len(node_ids):
        raise ValueError("Duplicate global node IDs in graph.")
    if not np.isfinite(x).all() or not np.isfinite(edge_attr).all():
        raise ValueError("Graph features contain NaN or infinity.")


def run(config_path: Path) -> None:
    root = Path.cwd()
    config = load_config(config_path)
    source, raw, interim = (project_path(root, config[key]) for key in ("source_csv", "raw_csv", "interim_csv"))
    processed, artifacts_dir, reports_dir = (project_path(root, config[key]) for key in ("processed_dir", "artifacts_dir", "reports_dir"))
    if not source.exists():
        raise FileNotFoundError(source)
    LOG.info("Loaded Phase 1 config: %s", config_path)
    copy_raw_source(source, raw)
    validate_schema(raw)
    LOG.info("Schema validation complete.")
    audit = audit_dataset(raw, config["chunk_size"])
    audit["feature_mapping"] = feature_mapping()
    audit["cleaning"] = clean_csv(raw, interim, config["chunk_size"])
    audit["duplicate_count_exact"] = audit["cleaning"]["exact_duplicates_removed"]
    sampled = sample_cleaned(interim, audit, config)
    LOG.info("Creating chronological train/validation/test split.")
    sampled, boundaries = split_periods(sampled, config)
    endpoints = sorted(set(sampled["ip.src_host"].astype(str)) | set(sampled["ip.dst_host"].astype(str)))
    endpoint_map = {endpoint: index for index, endpoint in enumerate(endpoints)}
    LOG.info("Prepared deterministic node mapping for %s endpoints.", len(endpoint_map))
    prepared, record_features, transformer, numeric, categorical, omitted = prepare_record_features(sampled, config)
    LOG.info("Building temporal directed graphs with %s-second windows.", config["window_seconds"])
    graphs = build_raw_graphs(prepared, record_features, endpoint_map, config["window_seconds"])
    LOG.info("Built %s temporal graph windows. Saving PyG snapshots.", len(graphs))
    graph_stats, node_scaler, edge_scaler = save_graphs(graphs, boundaries, processed, config["overwrite_outputs"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    with (artifacts_dir / "feature_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump({"source_column_roles": feature_mapping(), "record_numeric_features": numeric,
                   "record_categorical_features": categorical, "omitted_high_cardinality_categoricals": omitted,
                   "node_features": ["inbound_packets", "inbound_bytes_proxy", "outbound_packets", "outbound_bytes_proxy", "mean_packet_size_proxy", "mean_interarrival_time", "peer_count"],
                   "edge_aggregate_features": ["packet_count", "byte_count_proxy", "mean_packet_size_proxy", "std_packet_size_proxy", "mean_interarrival_time", "std_interarrival_time", "mean_preprocessed_record_features"],
                   "labels_excluded_from_inputs": ["Attack_label", "Attack_type"]}, handle, indent=2)
    with (artifacts_dir / "node_mapping.json").open("w", encoding="utf-8") as handle:
        json.dump(endpoint_map, handle, indent=2, sort_keys=True)
    dump({"record_preprocessor": transformer, "node_scaler": node_scaler, "edge_scaler": edge_scaler}, artifacts_dir / "scaler.pkl")
    with (artifacts_dir / "preprocessing_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump({"random_seed": config["random_seed"], "window_seconds": config["window_seconds"], "split_boundaries": boundaries,
                   "sample_size": len(sampled), "node_mapping_scope": "sampled endpoints, lexicographically sorted"}, handle, default=json_default, indent=2)
    with (artifacts_dir / "dataset_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, default=json_default, indent=2)
    pd.DataFrame(graph_stats).to_csv(reports_dir / "graph_statistics.csv", index=False)
    LOG.info("Phase 1 complete: %s snapshots written to %s", len(graph_stats), processed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1 Edge-IIoT temporal graph construction")
    parser.add_argument("--config", default="config/phase1_config.json", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config)


if __name__ == "__main__":
    main()
