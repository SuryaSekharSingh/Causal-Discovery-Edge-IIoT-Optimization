"""Standalone Phase 3 training and export pipeline for the edge-aware GNN."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import random
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from src.gnn_dataset import Phase3DataBundle, SPLITS, build_phase3_data
from src.gnn_model import EdgeAwareGATEncoder, Phase3GNN, load_phase3_encoder, masked_phase3_loss

LOG = logging.getLogger("phase3")


def _read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "processed_dir", "causal_dataset_csv", "phase2_artifacts_dir", "phase2_reports_dir",
        "phase3_artifacts_dir", "phase3_reports_dir", "random_seed", "window_seconds",
        "graph_splits", "expected_schema", "model", "training",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Phase 3 config is missing keys: {sorted(missing)}")
    if set(config["graph_splits"]) != set(SPLITS):
        raise ValueError(f"graph_splits must declare exactly {list(SPLITS)}.")
    model = config["model"]
    if int(model["hidden_dim"]) % int(model["attention_heads"]):
        raise ValueError("model.hidden_dim must be divisible by model.attention_heads.")
    if int(model["num_layers"]) < 1 or int(model["embedding_dim"]) < 1:
        raise ValueError("Model layer and embedding dimensions must be positive.")
    training = config["training"]
    if int(training["epochs"]) < 1 or int(training["batch_size"]) < 1:
        raise ValueError("Training epochs and batch_size must be positive.")
    if float(training["current_loss_weight"]) <= 0 or float(training["next_loss_weight"]) < 0:
        raise ValueError("Loss weights must be non-negative and current_loss_weight must be positive.")
    return config


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _prepare_output_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Phase 3 output directory is not empty: {path}. "
                "Set overwrite_outputs=true only when you intend to replace Phase 3 outputs."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Phase 3 requested CUDA, but CUDA is unavailable in this PyTorch installation.")
    if requested not in {"cpu", "cuda"}:
        raise ValueError("training.device must be cpu, cuda, or auto.")
    return torch.device(requested)


def _make_loader(
    dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


def _build_model(bundle: Phase3DataBundle, config: dict[str, Any]) -> Phase3GNN:
    model_config = config["model"]
    encoder = EdgeAwareGATEncoder(
        node_dim=bundle.node_dim,
        edge_dim=bundle.edge_dim,
        hidden_dim=int(model_config["hidden_dim"]),
        embedding_dim=int(model_config["embedding_dim"]),
        num_layers=int(model_config["num_layers"]),
        attention_heads=int(model_config["attention_heads"]),
        dropout=float(model_config["dropout"]),
        graph_size_mean=bundle.normalization.graph_size_mean,
        graph_size_scale=bundle.normalization.graph_size_scale,
    )
    return Phase3GNN(encoder, causal_dim=len(bundle.causal_features))


def _loss_for_batch(model: Phase3GNN, batch: Any, config: dict[str, Any]):
    outputs = model(batch)
    return masked_phase3_loss(
        outputs,
        batch.y_current,
        batch.y_next,
        batch.next_mask,
        current_weight=float(config["training"]["current_loss_weight"]),
        next_weight=float(config["training"]["next_loss_weight"]),
    )


def _run_epoch(
    model: Phase3GNN,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    current_sum = 0.0
    next_sum = 0.0
    graph_total = 0
    next_total = 0
    for batch in loader:
        batch = batch.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        total, current, following = _loss_for_batch(model, batch, config)
        if training:
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip"]))
            optimizer.step()
        graph_count = int(batch.num_graphs)
        valid_next = int(batch.next_mask.sum().detach().cpu())
        current_sum += float(current.detach()) * graph_count
        next_sum += float(following.detach()) * valid_next
        graph_total += graph_count
        next_total += valid_next
    if graph_total == 0:
        raise ValueError("Cannot run an epoch on an empty graph loader.")
    current_loss = current_sum / graph_total
    next_loss = next_sum / next_total if next_total else 0.0
    total_loss = (
        float(config["training"]["current_loss_weight"]) * current_loss
        + float(config["training"]["next_loss_weight"]) * next_loss
    )
    return {
        "loss": total_loss,
        "current_loss": current_loss,
        "next_loss": next_loss,
    }


def _train(
    model: Phase3GNN,
    loaders: dict[str, DataLoader],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[dict[str, float]], int, float, dict[str, torch.Tensor]]:
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, int(training["epochs"]) + 1):
        train_metrics = _run_epoch(model, loaders["train"], config, device, optimizer)
        with torch.no_grad():
            validation_metrics = _run_epoch(model, loaders["validation"], config, device, None)
        row = {"epoch": epoch}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"validation_{key}": value for key, value in validation_metrics.items()})
        history.append(row)
        if epoch == 1 or epoch % 10 == 0:
            LOG.info(
                "Epoch %s: train_loss=%.6f validation_loss=%.6f",
                epoch, train_metrics["loss"], validation_metrics["loss"],
            )
        improved = validation_metrics["loss"] < best_loss - float(training["minimum_improvement"])
        if improved:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            best_state = _cpu_state_dict(model)
            patience = 0
        else:
            patience += 1
        if patience >= int(training["early_stopping_patience"]):
            LOG.info("Early stopping after epoch %s; best epoch was %s.", epoch, best_epoch)
            break
    if best_state is None:
        raise RuntimeError("Training completed without producing a valid checkpoint.")
    model.load_state_dict(best_state)
    return history, best_epoch, best_loss, best_state


@torch.no_grad()
def _collect_predictions(model: Phase3GNN, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    collected: dict[str, list[Any]] = {
        "current_prediction": [], "next_prediction": [], "current_target": [], "next_target": [],
        "current_raw": [], "next_raw": [], "next_mask": [], "embedding": [],
        "graph_file": [], "window_start": [],
    }
    for batch in loader:
        graph_files = list(batch.graph_file)
        window_starts = list(batch.window_start)
        batch = batch.to(device)
        outputs = model(batch)
        for key in ("current_prediction", "next_prediction", "embedding"):
            collected[key].append(outputs[key].cpu())
        collected["current_target"].append(batch.y_current.cpu())
        collected["next_target"].append(batch.y_next.cpu())
        collected["current_raw"].append(batch.y_current_raw.cpu())
        collected["next_raw"].append(batch.y_next_raw.cpu())
        collected["next_mask"].append(batch.next_mask.cpu().reshape(-1))
        collected["graph_file"].extend(graph_files)
        collected["window_start"].extend(window_starts)
    for key in (
        "current_prediction", "next_prediction", "current_target", "next_target",
        "current_raw", "next_raw", "next_mask", "embedding",
    ):
        collected[key] = torch.cat(collected[key], dim=0)
    return collected


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    denominator = float(np.square(actual - actual.mean()).sum())
    if denominator <= 0:
        return None
    return float(1.0 - np.square(actual - predicted).sum() / denominator)


def _task_metrics(
    actual_raw: np.ndarray,
    predicted_raw: np.ndarray,
    actual_standardized: np.ndarray,
    predicted_standardized: np.ndarray,
    features: list[str],
) -> dict[str, Any]:
    per_feature: dict[str, Any] = {}
    for index, feature in enumerate(features):
        error = predicted_raw[:, index] - actual_raw[:, index]
        standardized_error = predicted_standardized[:, index] - actual_standardized[:, index]
        per_feature[feature] = {
            "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.square(error).mean())),
            "r2": _r2(actual_raw[:, index], predicted_raw[:, index]),
            "standardized_mse": float(np.square(standardized_error).mean()),
        }
    return {
        "sample_count": int(actual_raw.shape[0]),
        "macro_standardized_mse": float(
            np.square(predicted_standardized - actual_standardized).mean()
        ),
        "per_feature": per_feature,
    }


def _evaluate_split(
    collected: dict[str, Any],
    bundle: Phase3DataBundle,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mean = bundle.normalization.causal_mean
    scale = bundle.normalization.causal_scale
    current_pred_std = collected["current_prediction"].numpy()
    next_pred_std = collected["next_prediction"].numpy()
    current_target_std = collected["current_target"].numpy()
    next_target_std = collected["next_target"].numpy()
    current_raw = collected["current_raw"].numpy()
    current_pred_raw = current_pred_std * scale + mean
    mask = collected["next_mask"].numpy().astype(bool)
    next_raw = collected["next_raw"].numpy()[mask]
    next_pred_std_masked = next_pred_std[mask]
    next_target_std_masked = next_target_std[mask]
    next_pred_raw = next_pred_std_masked * scale + mean

    current_metrics = _task_metrics(
        current_raw, current_pred_raw, current_target_std, current_pred_std, bundle.causal_features
    )
    next_metrics = _task_metrics(
        next_raw, next_pred_raw, next_target_std_masked, next_pred_std_masked, bundle.causal_features
    )
    baselines = {
        "current_train_mean_standardized_mse": float(np.square(current_target_std).mean()),
        "next_train_mean_standardized_mse": float(np.square(next_target_std_masked).mean()),
        "next_persistence_standardized_mse": float(
            np.square(current_target_std[mask] - next_target_std_masked).mean()
        ),
    }
    rows: list[dict[str, Any]] = []
    for task, actual, predicted, selected in (
        ("current", current_raw, current_pred_raw, np.ones(len(current_raw), dtype=bool)),
        ("next", next_raw, next_pred_raw, mask),
    ):
        source_indices = np.flatnonzero(selected)
        for local_index, source_index in enumerate(source_indices):
            for feature_index, feature in enumerate(bundle.causal_features):
                rows.append({
                    "graph_file": collected["graph_file"][source_index],
                    "window_start": collected["window_start"][source_index],
                    "task": task,
                    "feature": feature,
                    "actual": float(actual[local_index, feature_index]),
                    "predicted": float(predicted[local_index, feature_index]),
                })
    return {"current": current_metrics, "next": next_metrics, "baselines": baselines}, rows


def _embedding_diagnostics(embeddings: torch.Tensor) -> dict[str, Any]:
    values = embeddings.numpy().astype(np.float64)
    standard_deviation = np.std(values, axis=0, ddof=0)
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = np.square(singular_values)
    probabilities = variance / variance.sum() if variance.sum() > 0 else np.zeros_like(variance)
    positive = probabilities[probabilities > 0]
    effective_rank = float(np.exp(-(positive * np.log(positive)).sum())) if positive.size else 0.0
    return {
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "minimum_dimension_std": float(standard_deviation.min()),
        "maximum_dimension_std": float(standard_deviation.max()),
        "collapsed_dimensions": int((standard_deviation <= 1e-6).sum()),
        "all_dimensions_collapsed": bool(np.all(standard_deviation <= 1e-6)),
        "effective_rank": effective_rank,
        "mean_l2_norm": float(np.linalg.norm(values, axis=1).mean()),
        "std_l2_norm": float(np.linalg.norm(values, axis=1).std()),
    }


def _model_config(bundle: Phase3DataBundle, config: dict[str, Any]) -> dict[str, Any]:
    model = config["model"]
    return {
        "architecture": "bidirectional_edge_aware_gatv2",
        "node_dim": bundle.node_dim,
        "edge_dim": bundle.edge_dim,
        "hidden_dim": int(model["hidden_dim"]),
        "embedding_dim": int(model["embedding_dim"]),
        "num_layers": int(model["num_layers"]),
        "attention_heads": int(model["attention_heads"]),
        "dropout": float(model["dropout"]),
    }


def _save_checkpoints(
    model: Phase3GNN,
    bundle: Phase3DataBundle,
    config: dict[str, Any],
    best_epoch: int,
    best_loss: float,
    artifacts_dir: Path,
) -> str:
    normalization = bundle.normalization.to_dict(bundle.causal_features)
    common = {
        "checkpoint_version": 1,
        "model_config": _model_config(bundle, config),
        "causal_features": bundle.causal_features,
        "normalization": normalization,
        "best_epoch": best_epoch,
        "validation_loss": best_loss,
    }
    torch.save(
        {
            **common,
            "model_state_dict": _cpu_state_dict(model),
            "encoder_state_dict": _cpu_state_dict(model.encoder),
        },
        artifacts_dir / "best_model.pt",
    )
    torch.save(
        {**common, "encoder_state_dict": _cpu_state_dict(model.encoder), "frozen_by_default": True},
        artifacts_dir / "encoder.pt",
    )
    _write_json(artifacts_dir / "normalization.json", normalization)
    return _sha256(artifacts_dir / "encoder.pt")


def _save_embeddings(
    collected_by_split: dict[str, dict[str, Any]],
    diagnostics: dict[str, Any],
    bundle: Phase3DataBundle,
    encoder_digest: str,
    artifacts_dir: Path,
) -> None:
    output_dir = artifacts_dir / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, collected in collected_by_split.items():
        embeddings = collected["embedding"].to(torch.float32)
        torch.save(
            {
                "format_version": 1,
                "split": split,
                "embeddings": embeddings,
                "graph_files": collected["graph_file"],
                "window_start": collected["window_start"],
                "embedding_dim": int(embeddings.shape[1]),
                "encoder_sha256": encoder_digest,
                "causal_feature_order": bundle.causal_features,
            },
            output_dir / f"{split}.pt",
        )
        if diagnostics[split]["all_dimensions_collapsed"]:
            raise RuntimeError(f"Exported {split} embeddings are completely collapsed.")


def _write_plots(history: pd.DataFrame, metrics: dict[str, Any], reports_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        LOG.warning("matplotlib is unavailable; Phase 3 PNG reports were skipped.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["validation_loss"], label="validation")
    axes[0].set_title("Combined loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["train_current_loss"], label="current")
    axes[1].plot(history["epoch"], history["train_next_loss"], label="next")
    axes[1].set_title("Training objectives")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "training_curves.png", dpi=180)
    plt.close(fig)

    features = list(metrics["validation"]["current"]["per_feature"])
    positions = np.arange(len(features))
    width = 0.2
    fig, ax = plt.subplots(figsize=(13, 5))
    series = [
        ("validation current", metrics["validation"]["current"]),
        ("validation next", metrics["validation"]["next"]),
        ("test current", metrics["test"]["current"]),
        ("test next", metrics["test"]["next"]),
    ]
    for index, (label, task_metrics) in enumerate(series):
        values = [task_metrics["per_feature"][feature]["standardized_mse"] for feature in features]
        ax.bar(positions + (index - 1.5) * width, values, width, label=label)
    ax.set_xticks(positions, features, rotation=45, ha="right")
    ax.set_ylabel("Standardized MSE")
    ax.set_title("Causal-target prediction quality")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(reports_dir / "prediction_quality.png", dpi=180)
    plt.close(fig)


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "torch-geometric", "numpy", "pandas", "scikit-learn", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _graph_inventory_digest(root: Path, config: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    processed = (root / config["processed_dir"]).resolve()
    for split in SPLITS:
        for path in sorted((processed / config["graph_splits"][split]).glob("*.pt")):
            digest.update(f"{split}/{path.name}:{path.stat().st_size}\n".encode("utf-8"))
    return digest.hexdigest()


def _audit_outputs(
    artifacts_dir: Path,
    reports_dir: Path,
    bundle: Phase3DataBundle,
    loaders: dict[str, DataLoader],
    device: torch.device,
) -> None:
    required = [
        artifacts_dir / "best_model.pt",
        artifacts_dir / "encoder.pt",
        artifacts_dir / "normalization.json",
        artifacts_dir / "model_metadata.json",
        reports_dir / "training_history.csv",
        reports_dir / "evaluation_metrics.json",
        reports_dir / "prediction_quality.csv",
        reports_dir / "embedding_diagnostics.json",
        reports_dir / "run_manifest.json",
    ] + [artifacts_dir / "embeddings" / f"{split}.pt" for split in SPLITS]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Phase 3 output audit found missing or empty files: {missing}")

    reloaded = load_phase3_encoder(artifacts_dir / "encoder.pt", device=device)
    for split in SPLITS:
        payload = torch.load(
            artifacts_dir / "embeddings" / f"{split}.pt", map_location="cpu", weights_only=True
        )
        embeddings = payload["embeddings"]
        if tuple(embeddings.shape) != (bundle.split_counts[split], reloaded.embedding_dim):
            raise RuntimeError(f"Unexpected exported embedding shape for {split}: {tuple(embeddings.shape)}")
        if not torch.isfinite(embeddings).all():
            raise RuntimeError(f"Exported {split} embeddings contain non-finite values.")
        if len(payload["graph_files"]) != len(set(payload["graph_files"])):
            raise RuntimeError(f"Exported {split} embeddings contain duplicate graph filenames.")
        if payload["window_start"] != sorted(payload["window_start"]):
            raise RuntimeError(f"Exported {split} embeddings are not chronological.")
        reproduced = []
        with torch.no_grad():
            for batch in loaders[split]:
                reproduced.append(reloaded(batch.to(device)).cpu())
        reproduced_tensor = torch.cat(reproduced, dim=0)
        difference = float(torch.max(torch.abs(reproduced_tensor - embeddings)))
        if difference > 1e-6:
            raise RuntimeError(f"Reloaded encoder mismatch for {split}: max_abs_difference={difference}")
    LOG.info("ALL_PHASE3_CHECKS_PASSED")


def run(config_path: Path, validate_only: bool = False) -> None:
    root = Path.cwd()
    config = _read_config(config_path)
    seed = int(config["random_seed"])
    _seed_everything(seed)
    LOG.info("Validating and joining existing Phase 1 graphs with Phase 2 causal targets.")
    bundle = build_phase3_data(root, config)
    LOG.info(
        "Validated graph counts %s and contiguous temporal pairs %s.",
        bundle.split_counts, bundle.temporal_pair_counts,
    )
    if validate_only:
        LOG.info("Phase 3 input validation complete; no outputs were written.")
        return

    artifacts_dir = (root / config["phase3_artifacts_dir"]).resolve()
    reports_dir = (root / config["phase3_reports_dir"]).resolve()
    overwrite = bool(config.get("overwrite_outputs", False))
    _prepare_output_directory(artifacts_dir, overwrite)
    _prepare_output_directory(reports_dir, overwrite)
    matplotlib_cache = Path(tempfile.gettempdir()) / "edge_iiot_phase3_matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    training = config["training"]
    device = _resolve_device(str(training["device"]))
    loaders = {
        split: _make_loader(
            bundle.datasets[split],
            batch_size=int(training["batch_size"]),
            shuffle=split == "train",
            seed=seed,
            num_workers=int(training["num_workers"]),
        )
        for split in SPLITS
    }
    evaluation_loaders = {
        split: _make_loader(
            bundle.datasets[split],
            batch_size=int(training["batch_size"]),
            shuffle=False,
            seed=seed,
            num_workers=int(training["num_workers"]),
        )
        for split in SPLITS
    }
    model = _build_model(bundle, config).to(device)
    LOG.info("Training bidirectional edge-aware GATv2 on %s.", device)
    history, best_epoch, best_loss, _ = _train(model, loaders, config, device)
    encoder_digest = _save_checkpoints(model, bundle, config, best_epoch, best_loss, artifacts_dir)

    # The test split is evaluated only after validation has selected and locked the checkpoint.
    collected_by_split: dict[str, dict[str, Any]] = {}
    metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {}
    for split in SPLITS:
        collected = _collect_predictions(model, evaluation_loaders[split], device)
        collected_by_split[split] = collected
        split_metrics, rows = _evaluate_split(collected, bundle)
        metrics[split] = split_metrics
        for row in rows:
            row["split"] = split
        prediction_rows.extend(rows)
        diagnostics[split] = _embedding_diagnostics(collected["embedding"])

    validation_model_mse = metrics["validation"]["current"]["macro_standardized_mse"]
    validation_baseline_mse = metrics["validation"]["baselines"]["current_train_mean_standardized_mse"]
    if validation_model_mse >= validation_baseline_mse:
        raise RuntimeError(
            "Current-state validation prediction did not beat the training-mean baseline: "
            f"model={validation_model_mse:.6f}, baseline={validation_baseline_mse:.6f}"
        )

    _save_embeddings(collected_by_split, diagnostics, bundle, encoder_digest, artifacts_dir)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(reports_dir / "training_history.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(reports_dir / "prediction_quality.csv", index=False)
    _write_json(reports_dir / "evaluation_metrics.json", metrics)
    _write_json(reports_dir / "embedding_diagnostics.json", diagnostics)

    metadata = {
        "phase": 3,
        "purpose": "standalone causally supervised graph representation learning",
        "model": _model_config(bundle, config),
        "causal_features": bundle.causal_features,
        "split_counts": bundle.split_counts,
        "temporal_pair_counts": bundle.temporal_pair_counts,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "encoder_sha256": encoder_digest,
        "frozen_for_phase4_by_default": True,
        "test_set_usage": "evaluated only after validation checkpoint selection",
        "unsupported_claims": ["latency prediction", "energy prediction", "action effects", "counterfactual safety"],
    }
    _write_json(artifacts_dir / "model_metadata.json", metadata)
    manifest = {
        "config": config,
        "package_versions": _package_versions(),
        "inputs": {
            "causal_dataset_sha256": _sha256((root / config["causal_dataset_csv"]).resolve()),
            "refined_dag_sha256": _sha256((root / config["phase2_artifacts_dir"] / "refined_dag.json").resolve()),
            "variable_decisions_sha256": _sha256(
                (root / config["phase2_reports_dir"] / "causal_variable_decisions.json").resolve()
            ),
            "graph_inventory_sha256": _graph_inventory_digest(root, config),
        },
        "outputs": {"encoder_sha256": encoder_digest},
    }
    _write_json(reports_dir / "run_manifest.json", manifest)
    if config.get("reports", {}).get("generate_plots", True):
        _write_plots(history_frame, metrics, reports_dir)
    _audit_outputs(artifacts_dir, reports_dir, bundle, evaluation_loaders, device)
    LOG.info("Phase 3 complete. Encoder and embeddings written to %s", artifacts_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 edge-aware GNN training and embedding export")
    parser.add_argument("--config", default="config/phase3_config.json", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config, validate_only=args.validate_only)


if __name__ == "__main__":
    main()
