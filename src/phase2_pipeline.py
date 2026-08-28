"""Phase 2 pipeline: graph snapshots to candidate DAG and reusable SCM."""

from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.causal_dataset import build_causal_dataset, write_causal_feature_mapping
from src.causal_discovery import discover_candidate_dag, draw_dag, write_dag_json
from src.causal_refinement import apply_constraints, apply_stability_filter, write_refinement_log
from src.causal_scm import LinearSCM, save_scm
from src.causal_validation import adjacency_to_dag, constant_features, validate_dag, validate_no_constant_features, validate_numeric_frame
from src.phase2_schema import CAUSAL_FEATURES, build_paths, validate_phase2_config

LOG = logging.getLogger("phase2")


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    validate_phase2_config(config)
    return config


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def select_features(train_frame: pd.DataFrame, config: dict[str, Any], paths) -> list[str]:
    validate_numeric_frame(train_frame)
    removed: list[dict[str, Any]] = []
    features = list(CAUSAL_FEATURES)
    constants = constant_features(train_frame, features)
    for feature in constants:
        removed.append({"feature": feature, "reason": "constant_on_training_period"})
    features = [feature for feature in features if feature not in constants]

    threshold = config["dataset"]["correlation_redundancy_threshold"]
    corr = train_frame[features].corr(method="pearson").abs()
    redundant: set[str] = set()
    for i, source in enumerate(features):
        for target in features[i + 1 :]:
            if source not in redundant and target not in redundant and corr.loc[source, target] >= threshold:
                redundant.add(target)
                removed.append({"feature": target, "reason": "redundant_high_pearson_correlation", "kept_feature": source, "correlation": float(corr.loc[source, target])})
    features = [feature for feature in features if feature not in redundant]
    if len(features) < 2:
        raise ValueError("Fewer than two non-constant/non-redundant causal variables remain.")
    validate_no_constant_features(train_frame, features)
    _write_json(paths.phase2_reports_dir / "causal_variable_decisions.json", {"used_features": features, "removed_features": removed})
    return features


def write_exploratory_reports(dataset: pd.DataFrame, train_frame: pd.DataFrame, features: list[str], paths) -> None:
    paths.phase2_reports_dir.mkdir(parents=True, exist_ok=True)
    train_frame[features].describe().T.to_csv(paths.phase2_reports_dir / "summary_statistics.csv")
    train_frame[features].corr(method="pearson").to_csv(paths.phase2_reports_dir / "pearson_correlation.csv")
    train_frame[features].corr(method="spearman").to_csv(paths.phase2_reports_dir / "spearman_correlation.csv")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        LOG.warning(
            "matplotlib is not installed, so Phase 2 CSV/JSON/SCM outputs will be generated without PNG plots. "
            "Install plotting support with `python -m pip install -r requirements.txt`."
        )
        return

    for method, filename in [("pearson", "pearson_correlation.png"), ("spearman", "spearman_correlation.png")]:
        corr = train_frame[features].corr(method=method)
        fig, ax = plt.subplots(figsize=(10, 8))
        image = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(features)), features, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(features)), features, fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{method.title()} correlation")
        fig.tight_layout()
        fig.savefig(paths.phase2_reports_dir / filename, dpi=180)
        plt.close(fig)

    axes = train_frame[features].hist(figsize=(12, 10), bins=24)
    for ax in np.ravel(axes):
        ax.tick_params(labelsize=7)
    plt.tight_layout()
    plt.savefig(paths.phase2_reports_dir / "variable_distributions.png", dpi=180)
    plt.close()

    timeline = dataset.copy()
    timeline["window_start"] = pd.to_datetime(timeline["window_start"], errors="coerce")
    fig, axes = plt.subplots(len(features), 1, figsize=(12, max(8, len(features) * 1.6)), sharex=True)
    if len(features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, features):
        for split, split_frame in timeline.groupby("split", sort=False):
            ax.plot(split_frame["window_start"], split_frame[feature], label=split, linewidth=0.9)
        ax.set_ylabel(feature, fontsize=7)
        ax.tick_params(labelsize=7)
    axes[0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(paths.phase2_reports_dir / "temporal_plots.png", dpi=180)
    plt.close(fig)


def run_stability(train_frame: pd.DataFrame, features: list[str], config: dict[str, Any], paths) -> dict[str, Any]:
    stability_config = config["stability"]
    counts: dict[str, int] = {}
    runs: list[dict[str, Any]] = []
    n = len(train_frame)
    sample_size = max(2, int(round(n * stability_config["subsample_fraction"])))
    seeds = stability_config["random_seeds"][: stability_config["n_bootstraps"]]
    for seed in seeds:
        sample = train_frame.sample(n=sample_size, replace=True, random_state=int(seed))
        local_config = json.loads(json.dumps(config))
        local_config["discovery"]["edge_weight_threshold"] = stability_config["edge_weight_threshold"]
        adjacency, _ = discover_candidate_dag(sample, features, local_config)
        selected = []
        for i, source in enumerate(features):
            for j, target in enumerate(features):
                if i != j and adjacency[i, j] != 0:
                    key = f"{source}->{target}"
                    counts[key] = counts.get(key, 0) + 1
                    selected.append(key)
        runs.append({"seed": int(seed), "selected_edges": selected})
    total = max(len(seeds), 1)
    edge_rows = []
    for key, count in sorted(counts.items()):
        source, target = key.split("->", 1)
        edge_rows.append({"source": source, "target": target, "selection_count": count, "selection_frequency": count / total, "stable": count / total >= stability_config["minimum_stability"]})
    payload = {"n_bootstraps": total, "minimum_stability": stability_config["minimum_stability"], "edges": {f"{row['source']}->{row['target']}": row for row in edge_rows}, "runs": runs}
    _write_json(paths.phase2_artifacts_dir / "edge_stability.json", payload)
    pd.DataFrame(edge_rows, columns=["source", "target", "selection_count", "selection_frequency", "stable"]).to_csv(paths.phase2_reports_dir / "edge_stability.csv", index=False)
    return payload


def run(config_path: Path) -> None:
    root = Path.cwd()
    config = load_config(config_path)
    paths = build_paths(root, config)
    np.random.seed(int(config["random_seed"]))
    paths.phase2_artifacts_dir.mkdir(parents=True, exist_ok=True)
    paths.phase2_reports_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = Path(tempfile.gettempdir()) / "edge_iiot_matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    LOG.info("Building Phase 2 causal dataset from existing graph snapshots.")
    dataset = build_causal_dataset(config, paths)
    write_causal_feature_mapping(paths, config)
    validate_numeric_frame(dataset)

    train_frame = dataset.loc[dataset["split"].eq("train")].reset_index(drop=True)
    validation_frame = dataset.loc[dataset["split"].eq("validation")].reset_index(drop=True)
    if len(train_frame) < config["dataset"]["minimum_rows_per_split"]["train"]:
        raise ValueError("Training causal dataset is too small for discovery.")
    if len(validation_frame) < config["dataset"]["minimum_rows_per_split"]["validation"]:
        raise ValueError("Validation causal dataset is missing or too small for SCM validation.")

    features = select_features(train_frame, config, paths)
    if config.get("reports", {}).get("generate_plots", True):
        write_exploratory_reports(dataset, train_frame, features, paths)
    else:
        paths.phase2_reports_dir.mkdir(parents=True, exist_ok=True)
        train_frame[features].describe().T.to_csv(paths.phase2_reports_dir / "summary_statistics.csv")
        train_frame[features].corr(method="pearson").to_csv(paths.phase2_reports_dir / "pearson_correlation.csv")
        train_frame[features].corr(method="spearman").to_csv(paths.phase2_reports_dir / "spearman_correlation.csv")

    LOG.info("Running NOTEARS on training-period causal data.")
    candidate_adjacency, candidate_dag = discover_candidate_dag(train_frame, features, config)
    write_dag_json(paths.phase2_artifacts_dir / "candidate_dag.json", candidate_dag, {"type": "candidate causal DAG", "discovery_method": "notears", "features": features})
    if config.get("reports", {}).get("generate_plots", True):
        try:
            draw_dag(paths.phase2_reports_dir / "candidate_dag.png", candidate_dag, "Candidate causal DAG")
        except ModuleNotFoundError:
            LOG.warning("matplotlib is not installed; skipped candidate DAG PNG.")

    stability = {"edges": {}}
    if config["stability"]["enabled"]:
        LOG.info("Running NOTEARS stability analysis on training bootstraps.")
        stability = run_stability(train_frame, features, config, paths)

    constrained_adjacency, refinement_log = apply_constraints(candidate_adjacency, features, config["constraints"])
    if config["stability"]["enabled"]:
        constrained_adjacency, stability_log = apply_stability_filter(constrained_adjacency, features, stability, config["stability"]["minimum_stability"])
        refinement_log.extend(stability_log)
    refined_dag = adjacency_to_dag(constrained_adjacency, features)
    validate_dag(refined_dag, features)
    write_dag_json(paths.phase2_artifacts_dir / "refined_dag.json", refined_dag, {"type": "domain-refined stable candidate causal DAG", "features": features})
    if config.get("reports", {}).get("generate_plots", True):
        try:
            draw_dag(paths.phase2_reports_dir / "final_dag.png", refined_dag, "Final refined candidate causal DAG")
        except ModuleNotFoundError:
            LOG.warning("matplotlib is not installed; skipped final DAG PNG.")
    write_refinement_log(paths.phase2_reports_dir / "causal_refinement_log.json", refinement_log)

    LOG.info("Fitting interpretable linear SCM.")
    scm = LinearSCM(refined_dag, features).fit(train_frame)
    metadata = scm.metadata(train_frame, validation_frame)
    metadata.update({"config": config, "causal_dataset": str(paths.causal_dataset_csv), "test_set_usage": "untouched"})
    save_scm(scm, paths.phase2_artifacts_dir / "scm.pkl", paths.phase2_artifacts_dir / "scm_metadata.json", metadata)
    LOG.info("Phase 2 complete. Artifacts written to %s", paths.phase2_artifacts_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 causal discovery and SCM construction")
    parser.add_argument("--config", default="config/phase2_config.json", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config)


if __name__ == "__main__":
    main()
