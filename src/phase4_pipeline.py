"""Standalone Phase 4 DQN training, evaluation, and transition generation."""

from __future__ import annotations

import argparse
import copy
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
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from src.rl_agent import (
    FixedStartEvaluationCallback,
    build_dqn,
    evaluate_agent,
    load_phase4_agent,
    select_evaluation_starts,
)
from src.rl_environment import (
    EdgeIIoTRoutingEnv,
    EnvironmentCalibration,
    Phase4SplitData,
    calibrate_environment,
    file_sha256,
    load_phase4_split,
)

LOG = logging.getLogger("phase4")
SPLITS = ("train", "validation", "test")


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {
        "causal_dataset_csv", "phase3_artifacts_dir", "phase4_artifacts_dir",
        "phase4_reports_dir", "transition_dataset_csv", "random_seed", "window_seconds",
        "environment", "dqn", "transition_generation",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"Phase 4 config is missing keys: {sorted(missing)}")
    if float(config["window_seconds"]) <= 0:
        raise ValueError("window_seconds must be positive.")
    dqn = config["dqn"]
    for key in ("total_timesteps", "buffer_size", "learning_starts", "batch_size", "evaluation_frequency"):
        if int(dqn[key]) <= 0:
            raise ValueError(f"dqn.{key} must be positive.")
    if int(dqn["learning_starts"]) >= int(dqn["total_timesteps"]):
        raise ValueError("dqn.learning_starts must be smaller than dqn.total_timesteps.")
    return config


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def _prepare_directory(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Phase 4 output directory is not empty: {path}. "
                "Set overwrite_outputs=true only when replacing Phase 4 outputs intentionally."
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


def _make_env(
    data: Phase4SplitData,
    calibration: EnvironmentCalibration,
    config: dict[str, Any],
    starts: list[int] | None = None,
) -> EdgeIIoTRoutingEnv:
    return EdgeIIoTRoutingEnv(data, calibration, config, fixed_starts=starts)


def _causal_features(root: Path, config: dict[str, Any]) -> list[str]:
    metadata_path = (root / config["phase3_artifacts_dir"] / "model_metadata.json").resolve()
    with metadata_path.open(encoding="utf-8") as handle:
        features = json.load(handle).get("causal_features")
    if not isinstance(features, list) or len(features) != 9:
        raise ValueError("Phase 3 metadata must declare exactly nine causal features.")
    return [str(feature) for feature in features]


def _fixed_starts(
    data_by_split: dict[str, Phase4SplitData],
    calibration: EnvironmentCalibration,
    config: dict[str, Any],
) -> dict[str, list[int]]:
    count = int(config["dqn"]["evaluation_start_count"])
    return {
        split: select_evaluation_starts(_make_env(data, calibration, config), count)
        for split, data in data_by_split.items()
    }


def _evaluate_policies(
    agent,
    data: Phase4SplitData,
    calibration: EnvironmentCalibration,
    config: dict[str, Any],
    starts: list[int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    aggregates: dict[str, Any] = {}
    for policy in ("dqn", "random", "round_robin", "least_queue"):
        env = _make_env(data, calibration, config)
        rows, aggregate = evaluate_agent(
            agent if policy == "dqn" else None,
            env,
            starts,
            policy=policy,
            seed=seed,
        )
        all_rows.extend(rows)
        aggregates[policy] = aggregate
    return all_rows, aggregates


def _transition_row(
    env: EdgeIIoTRoutingEnv,
    action: int,
    outcome,
    causal_features: list[str],
    episode: int,
    step: int,
    horizon: int,
) -> dict[str, Any]:
    row = env.current_row
    source_index = env.current_index
    would_truncate = bool(step + 1 >= horizon or not env.valid_transition[source_index + 1])
    result: dict[str, Any] = {
        "state_id": f"{env.data.split}_{episode:04d}_{step:03d}",
        "split": env.data.split,
        "rollout_episode": episode,
        "step": step,
        "graph_file": str(row["graph_file"]),
        "window_start": row["window_start"].isoformat(),
        "next_graph_file": str(env.data.frame.iloc[source_index + 1]["graph_file"]),
        "action": action,
        "action_name": outcome.action_name,
        "allocation_server_a": float(outcome.allocation_fractions[0]),
        "allocation_server_b": float(outcome.allocation_fractions[1]),
        "allocation_server_c": float(outcome.allocation_fractions[2]),
        "pre_queue_server_a": float(outcome.pre_queue[0]),
        "pre_queue_server_b": float(outcome.pre_queue[1]),
        "pre_queue_server_c": float(outcome.pre_queue[2]),
        "post_queue_server_a": float(outcome.post_queue[0]),
        "post_queue_server_b": float(outcome.post_queue[1]),
        "post_queue_server_c": float(outcome.post_queue[2]),
        "demand_packets": outcome.demand_packets,
        "throughput_packets": outcome.throughput_packets,
        "pdr": outcome.pdr,
        "dropped_packets": outcome.dropped_packets,
        "drop_ratio": outcome.drop_ratio,
        "latency_proxy_ms": outcome.latency_proxy_ms,
        "energy_proxy": outcome.energy_proxy,
        "task_completion": outcome.task_completion,
        "reward": outcome.reward,
        "would_truncate": would_truncate,
    }
    result.update({feature: float(row[feature]) for feature in causal_features})
    return result


def generate_balanced_transitions(
    data_by_split: dict[str, Phase4SplitData],
    calibration: EnvironmentCalibration,
    config: dict[str, Any],
    causal_features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    seed = int(config["random_seed"])
    generation = config["transition_generation"]
    horizon = int(generation["episode_horizon"])
    rows: list[dict[str, Any]] = []
    states_by_split: dict[str, int] = {}
    for split_index, split in enumerate(SPLITS):
        env = _make_env(data_by_split[split], calibration, config)
        episode_count = int(generation["episodes"][split])
        rng = np.random.default_rng(seed + 1000 * (split_index + 1))
        state_count = 0
        for episode in range(episode_count):
            episode_seed = seed + 1000 * (split_index + 1) + episode
            env.reset(seed=episode_seed)
            done = False
            step = 0
            while not done and step < horizon:
                for action in range(env.action_space.n):
                    outcome = env.simulate_action(action)
                    rows.append(_transition_row(env, action, outcome, causal_features, episode, step, horizon))
                behavior_action = int(rng.integers(0, env.action_space.n))
                _, _, terminated, truncated, _ = env.step(behavior_action)
                done = bool(terminated or truncated)
                step += 1
                state_count += 1
        states_by_split[split] = state_count
    frame = pd.DataFrame(rows)
    group_sizes = frame.groupby("state_id").size()
    action_coverage = frame.groupby("state_id")["action"].nunique()
    if not group_sizes.eq(6).all() or not action_coverage.eq(6).all():
        raise RuntimeError("Balanced transition generation did not produce every action exactly once per state.")
    numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise RuntimeError("Balanced transition dataset contains non-finite values.")
    audit = {
        "rows": len(frame),
        "decision_states": int(frame["state_id"].nunique()),
        "states_by_split": states_by_split,
        "rows_by_split": frame.groupby("split").size().astype(int).to_dict(),
        "actions_per_state": 6,
        "complete_action_coverage": True,
        "phase5_training_policy": "fit action-aware models on training rows only",
    }
    return frame, audit


def _write_plots(
    callback_history: pd.DataFrame,
    evaluation: dict[str, Any],
    reports_dir: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        LOG.warning("matplotlib is unavailable; Phase 4 PNG reports were skipped.")
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(callback_history["timesteps"], callback_history["mean_cumulative_reward"], marker="o")
    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean validation cumulative reward")
    ax.set_title("DQN validation reward")
    fig.tight_layout()
    fig.savefig(reports_dir / "reward_curve.png", dpi=180)
    plt.close(fig)

    policies = ["dqn", "random", "round_robin", "least_queue"]
    metrics = ["mean_step_reward", "mean_pdr", "mean_task_completion"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4.5))
    for ax, metric in zip(axes, metrics):
        validation_values = [evaluation["validation"][policy][metric] for policy in policies]
        test_values = [evaluation["test"][policy][metric] for policy in policies]
        positions = np.arange(len(policies))
        ax.bar(positions - 0.18, validation_values, 0.36, label="validation")
        ax.bar(positions + 0.18, test_values, 0.36, label="test")
        ax.set_xticks(positions, policies, rotation=30, ha="right")
        ax.set_title(metric.replace("mean_", "").replace("_", " "))
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(reports_dir / "policy_comparison.png", dpi=180)
    plt.close(fig)

    metric_names = ["mean_latency_proxy_ms", "mean_energy_proxy", "mean_dropped_packets", "mean_throughput_packets"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.ravel(), metric_names):
        values = [evaluation["test"][policy][metric] for policy in policies]
        ax.bar(policies, values)
        ax.set_title(metric.replace("mean_", "").replace("_", " "))
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(reports_dir / "network_metrics.png", dpi=180)
    plt.close(fig)


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("torch", "gymnasium", "stable-baselines3", "tensorboard", "numpy", "pandas", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _audit_outputs(
    artifacts_dir: Path,
    reports_dir: Path,
    transition_path: Path,
    validation_data: Phase4SplitData,
    calibration: EnvironmentCalibration,
    config: dict[str, Any],
) -> None:
    required = [
        artifacts_dir / "best_model.zip",
        artifacts_dir / "final_model.zip",
        artifacts_dir / "environment_calibration.json",
        artifacts_dir / "model_metadata.json",
        artifacts_dir / "evaluation_start_states.json",
        reports_dir / "training.monitor.csv",
        reports_dir / "validation_history.csv",
        reports_dir / "evaluation_episodes.csv",
        reports_dir / "evaluation_metrics.json",
        reports_dir / "action_distribution.csv",
        reports_dir / "transition_dataset_audit.json",
        reports_dir / "run_manifest.json",
        transition_path,
    ]
    missing = [str(path) for path in required if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Phase 4 output audit found missing or empty files: {missing}")
    event_files = list((reports_dir / "tensorboard").rglob("events.out.tfevents.*"))
    if not event_files or any(path.stat().st_size == 0 for path in event_files):
        raise RuntimeError("Phase 4 TensorBoard event output is missing or empty.")
    transitions = pd.read_csv(transition_path)
    if not transitions.groupby("state_id").size().eq(6).all():
        raise RuntimeError("Transition audit found an incomplete six-action state.")
    if not transitions.groupby("state_id")["action"].nunique().eq(6).all():
        raise RuntimeError("Transition audit found duplicate/missing actions for a state.")
    env = _make_env(validation_data, calibration, config)
    model = load_phase4_agent(artifacts_dir / "best_model.zip", env=env, device=str(config["dqn"]["device"]))
    observation, _ = env.reset(seed=int(config["random_seed"]))
    action, _ = model.predict(observation, deterministic=True)
    if not env.action_space.contains(int(np.asarray(action).item())):
        raise RuntimeError("Reloaded DQN produced an invalid action.")
    LOG.info("ALL_PHASE4_CHECKS_PASSED")


def _validate_inputs(root: Path, config: dict[str, Any]) -> tuple[dict[str, Phase4SplitData], EnvironmentCalibration]:
    data = {split: load_phase4_split(root, config, split) for split in SPLITS}
    calibration = calibrate_environment(data["train"], config)
    for split in SPLITS:
        env = _make_env(data[split], calibration, config)
        observation, _ = env.reset(seed=int(config["random_seed"]))
        if observation.shape != (67,) or not env.observation_space.contains(observation):
            raise RuntimeError(f"Invalid initial observation for {split}.")
    check_env(_make_env(data["train"], calibration, config), warn=True)
    return data, calibration


def run_smoke_test(root: Path, config: dict[str, Any]) -> None:
    data, calibration = _validate_inputs(root, config)
    smoke_config = copy.deepcopy(config)
    smoke_config["dqn"].update({
        "total_timesteps": 1000,
        "learning_starts": 100,
        "buffer_size": 2000,
        "batch_size": 32,
    })
    with tempfile.TemporaryDirectory(prefix="phase4_smoke_") as directory:
        temporary = Path(directory)
        env = Monitor(_make_env(data["train"], calibration, smoke_config))
        agent = build_dqn(env, smoke_config, temporary / "tensorboard", int(config["random_seed"]))
        agent.learn(total_timesteps=1000, tb_log_name="smoke")
        checkpoint = temporary / "smoke_model"
        agent.save(str(checkpoint))
        loaded = load_phase4_agent(checkpoint.with_suffix(".zip"), env=env, device=str(config["dqn"]["device"]))
        observation, _ = env.reset(seed=int(config["random_seed"]))
        action, _ = loaded.predict(observation, deterministic=True)
        if not env.action_space.contains(int(np.asarray(action).item())):
            raise RuntimeError("Phase 4 smoke model produced an invalid action.")
    LOG.info("PHASE4_SMOKE_TEST_PASSED")


def run(config_path: Path, validate_only: bool = False, smoke_test: bool = False) -> None:
    root = Path.cwd()
    config = _load_config(config_path)
    seed = int(config["random_seed"])
    _seed_everything(seed)
    LOG.info("Validating Phase 3 embeddings and Phase 2 causal inputs for Phase 4.")
    if smoke_test:
        run_smoke_test(root, config)
        return
    if validate_only:
        data, calibration = _validate_inputs(root, config)
        LOG.info(
            "Phase 4 validation complete: counts=%s capacities=%s; no outputs written.",
            {split: value.size for split, value in data.items()},
            calibration.service_capacity.tolist(),
        )
        return

    artifacts_dir = (root / config["phase4_artifacts_dir"]).resolve()
    reports_dir = (root / config["phase4_reports_dir"]).resolve()
    transition_path = (root / config["transition_dataset_csv"]).resolve()
    overwrite = bool(config.get("overwrite_outputs", False))
    _prepare_directory(artifacts_dir, overwrite)
    _prepare_directory(reports_dir, overwrite)
    if transition_path.exists() and not overwrite:
        raise FileExistsError(
            f"Phase 4 transition dataset already exists: {transition_path}. Set overwrite_outputs=true to replace it."
        )
    transition_path.parent.mkdir(parents=True, exist_ok=True)
    matplotlib_cache = Path(tempfile.gettempdir()) / "edge_iiot_phase4_matplotlib"
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))

    # Only train and validation are loaded before validation chooses the checkpoint.
    train_data = load_phase4_split(root, config, "train")
    validation_data = load_phase4_split(root, config, "validation")
    calibration = calibrate_environment(train_data, config)
    _write_json(artifacts_dir / "environment_calibration.json", calibration.to_dict())
    check_env(_make_env(train_data, calibration, config), warn=True)
    pretest_data = {"train": train_data, "validation": validation_data}
    starts = _fixed_starts(pretest_data, calibration, config)

    train_env = Monitor(
        _make_env(train_data, calibration, config),
        filename=str(reports_dir / "training.monitor.csv"),
    )
    validation_factory = lambda: _make_env(validation_data, calibration, config)
    callback = FixedStartEvaluationCallback(
        validation_factory,
        starts["validation"],
        int(config["dqn"]["evaluation_frequency"]),
        artifacts_dir / "best_model",
        seed,
    )
    agent = build_dqn(train_env, config, reports_dir / "tensorboard", seed)
    LOG.info("Training DQN for %s timesteps.", config["dqn"]["total_timesteps"])
    agent.learn(
        total_timesteps=int(config["dqn"]["total_timesteps"]),
        callback=callback,
        tb_log_name="phase4_dqn",
    )
    agent.save(str(artifacts_dir / "final_model"))
    if not (artifacts_dir / "best_model.zip").exists():
        raise RuntimeError("Validation callback did not produce a best DQN checkpoint.")
    pd.DataFrame(callback.history).to_csv(reports_dir / "validation_history.csv", index=False)

    best_agent = load_phase4_agent(
        artifacts_dir / "best_model.zip",
        env=_make_env(validation_data, calibration, config),
        device=str(config["dqn"]["device"]),
    )

    # Test data enters the workflow only after the best checkpoint is fixed.
    test_data = load_phase4_split(root, config, "test")
    data_by_split = {"train": train_data, "validation": validation_data, "test": test_data}
    starts.update(_fixed_starts({"test": test_data}, calibration, config))
    _write_json(artifacts_dir / "evaluation_start_states.json", starts)

    evaluation_rows: list[dict[str, Any]] = []
    evaluation: dict[str, Any] = {}
    for split in ("validation", "test"):
        rows, aggregate = _evaluate_policies(
            best_agent, data_by_split[split], calibration, config, starts[split], seed
        )
        evaluation_rows.extend(rows)
        evaluation[split] = aggregate
    if (
        evaluation["validation"]["dqn"]["mean_cumulative_reward"]
        <= evaluation["validation"]["random"]["mean_cumulative_reward"]
    ):
        raise RuntimeError(
            "DQN validation reward did not exceed the paired random baseline: "
            f"dqn={evaluation['validation']['dqn']['mean_cumulative_reward']:.6f}, "
            f"random={evaluation['validation']['random']['mean_cumulative_reward']:.6f}"
        )
    pd.DataFrame(evaluation_rows).to_csv(reports_dir / "evaluation_episodes.csv", index=False)
    _write_json(reports_dir / "evaluation_metrics.json", evaluation)
    action_rows = []
    for split, policies in evaluation.items():
        for policy, metrics in policies.items():
            for action, (count, frequency) in enumerate(zip(metrics["action_counts"], metrics["action_frequencies"])):
                action_rows.append({
                    "split": split, "policy": policy, "action": action,
                    "action_name": config["environment"]["actions"][action]["name"],
                    "count": count, "frequency": frequency,
                })
    pd.DataFrame(action_rows).to_csv(reports_dir / "action_distribution.csv", index=False)

    causal_features = _causal_features(root, config)
    transition_frame, transition_audit = generate_balanced_transitions(
        data_by_split, calibration, config, causal_features
    )
    transition_frame.to_csv(transition_path, index=False)
    _write_json(reports_dir / "transition_dataset_audit.json", transition_audit)

    callback_frame = pd.DataFrame(callback.history)
    if config.get("reports", {}).get("generate_plots", True):
        _write_plots(callback_frame, evaluation, reports_dir)
    model_metadata = {
        "phase": 4,
        "algorithm": "DQN",
        "observation": {"shape": [67], "gnn_embedding": 64, "normalized_server_queues": 3},
        "action_space": {"type": "Discrete", "size": 6, "actions": config["environment"]["actions"]},
        "training_timesteps": int(config["dqn"]["total_timesteps"]),
        "best_validation_mean_reward": callback.best_mean_reward,
        "phase3_encoder_sha256": train_data.encoder_sha256,
        "gnn_frozen": True,
        "test_set_usage": "loaded only after best validation checkpoint selection",
        "metric_status": {
            "latency_proxy_ms": "synthetic; not measured latency",
            "energy_proxy": "synthetic; not measured energy consumption",
        },
        "phase5_transition_dataset": str(transition_path),
    }
    _write_json(artifacts_dir / "model_metadata.json", model_metadata)
    manifest = {
        "config": config,
        "package_versions": _package_versions(),
        "inputs": {
            "causal_dataset_sha256": file_sha256((root / config["causal_dataset_csv"]).resolve()),
            "phase3_encoder_sha256": train_data.encoder_sha256,
            "train_embeddings_sha256": file_sha256(
                (root / config["phase3_artifacts_dir"] / "embeddings" / "train.pt").resolve()
            ),
            "validation_embeddings_sha256": file_sha256(
                (root / config["phase3_artifacts_dir"] / "embeddings" / "validation.pt").resolve()
            ),
            "test_embeddings_sha256": file_sha256(
                (root / config["phase3_artifacts_dir"] / "embeddings" / "test.pt").resolve()
            ),
        },
        "outputs": {
            "best_model_sha256": file_sha256(artifacts_dir / "best_model.zip"),
            "transition_dataset_sha256": file_sha256(transition_path),
        },
    }
    _write_json(reports_dir / "run_manifest.json", manifest)
    _audit_outputs(artifacts_dir, reports_dir, transition_path, validation_data, calibration, config)
    LOG.info("Phase 4 complete. DQN and reports written under %s and %s", artifacts_dir, reports_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 standalone DQN routing pipeline")
    parser.add_argument("--config", default="config/phase4_config.json", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.validate_only and args.smoke_test:
        parser.error("--validate-only and --smoke-test are mutually exclusive.")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config, validate_only=args.validate_only, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
