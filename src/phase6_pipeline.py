"""Standalone Phase 6 factorial evaluation of timing-corrected routing policies."""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import random
import tempfile
import time

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.monitor import Monitor

from src.counterfactual_controller import CounterfactualController, DecisionState, METRICS, load_phase5_scm
from src.phase4_pipeline import _causal_features
from src.rl_agent import FixedStartEvaluationCallback, build_dqn, load_phase4_agent
from src.rl_environment import EdgeIIoTRoutingEnv, calibrate_environment, file_sha256, load_phase4_split

LOG = logging.getLogger("phase6")
REPRESENTATIONS = ("gnn", "causal", "minimal")
METRIC_FIELDS = ("reward", "throughput_packets", "pdr", "dropped_packets", "drop_ratio",
                 "latency_proxy_ms", "energy_proxy", "task_completion")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


class AblationObservation(gym.ObservationWrapper):
    """Replace the frozen GNN with previous-window causal variables or no context."""

    def __init__(self, env: EdgeIIoTRoutingEnv, representation: str, features: list[str], normalizer: dict):
        super().__init__(env)
        if representation not in ("causal", "minimal") or not env.timing_v2:
            raise ValueError("Ablation requires a timing-v2 environment and known representation")
        self.representation = representation
        self.features = features
        self.normalizer = normalizer
        count = len(features) if representation == "causal" else 0
        self.observation_space = spaces.Box(
            low=np.r_[np.full(count, -10.), 0., np.zeros(3)].astype(np.float32),
            high=np.r_[np.full(count, 10.), 10., np.ones(3)].astype(np.float32),
            dtype=np.float32,
        )

    @property
    def data(self):
        return self.env.data

    def observation(self, observation: np.ndarray) -> np.ndarray:
        if self.env.done:
            result = np.zeros(self.observation_space.shape, dtype=np.float32)
            result[-3:] = observation[-3:]
            return result
        context = np.empty(0, dtype=np.float64)
        if self.representation == "causal":
            raw = self.env.observation_row[self.features].to_numpy(dtype=np.float64)
            context = np.clip((raw - self.normalizer["mean"]) / self.normalizer["std"], -10, 10)
        result = np.r_[context, observation[64:]].astype(np.float32)
        if not self.observation_space.contains(result):
            raise RuntimeError("Ablation observation outside declared bounds")
        return result


def fit_normalizer(train, config: dict, features: list[str], calibration) -> dict:
    env = EdgeIIoTRoutingEnv(train, calibration, config)
    previous_indices = np.asarray(env.valid_starts, dtype=int) - 1
    values = train.frame.iloc[previous_indices][features].to_numpy(dtype=np.float64)
    mean, std = values.mean(axis=0), values.std(axis=0)
    std[std < 1e-12] = 1.0
    return {"features": features, "mean": mean.tolist(), "std": std.tolist(),
            "fitted_on_split": "train", "training_pairs": len(previous_indices)}


def make_env(data, calibration, phase4_config, representation: str, normalizer: dict):
    base = EdgeIIoTRoutingEnv(data, calibration, phase4_config)
    if representation == "gnn":
        return base
    return AblationObservation(base, representation, normalizer["features"], normalizer)


def base_env(env):
    return env.unwrapped


def decision_blocks(env: EdgeIIoTRoutingEnv) -> list[int]:
    """Cover every legal decision once with disjoint, horizon-limited episodes."""
    starts = []
    previous = -2
    span = 0
    for index in env.valid_starts:
        if index != previous + 1 or span >= env.episode_horizon:
            starts.append(index)
            span = 0
        span += 1
        previous = index
    return starts


def scenario_multipliers(env: EdgeIIoTRoutingEnv, start: int, multiplier: float,
                         onset: int, duration: int) -> tuple[np.ndarray, bool]:
    values = np.ones(env.data.size, dtype=np.float64)
    eligible = []
    for offset in range(env.episode_horizon):
        index = start + offset
        if index >= env.data.size or not env.valid_decision[index] or (offset and index != eligible[-1] + 1):
            break
        eligible.append(index)
    reached = len(eligible) > onset
    if reached:
        values[eligible[onset:onset + duration]] = multiplier
    return values, reached


def validate_inputs(root: Path, config: dict, include_test: bool):
    phase4_path = root / config["phase4_config"]
    phase5_path = root / config["phase5_config"]
    phase4 = read_json(phase4_path)
    phase5 = read_json(phase5_path)
    if phase4.get("timing_schema_version") != 2 or phase5.get("phase4_config") != config["phase4_config"]:
        raise ValueError("Phase 6 requires matching timing-v2 Phase 4/5 configurations")
    a4 = root / phase4["phase4_artifacts_dir"]
    r4 = root / phase4["phase4_reports_dir"]
    a5 = root / phase5["artifacts_dir"]
    r5 = root / phase5["reports_dir"]
    manifest4 = read_json(r4 / "run_manifest.json")
    manifest5 = read_json(r5 / "run_manifest.json")
    if manifest4["config"] != phase4 or manifest5["phase4_config"] != phase4:
        raise ValueError("Saved Phase 4/5 configurations disagree")
    if file_sha256(a4 / "best_model.zip") != manifest4["outputs"]["best_model_sha256"]:
        raise ValueError("Phase 4 DQN hash mismatch")
    if read_json(a4 / "model_metadata.json")["observation"]["shape"] != [68]:
        raise ValueError("Phase 4 DQN has a legacy observation shape")
    if read_json(a5 / "completion_status.json")["status"] != "complete":
        raise ValueError("Phase 5 did not complete")
    for key, digest in manifest5["inputs"].items():
        if file_sha256(root / key) != digest:
            raise ValueError(f"Phase 5 input hash mismatch: {key}")
    for group, folder in (("artifacts", a5), ("reports", r5)):
        for name, digest in manifest5["outputs"][group].items():
            if file_sha256(folder / name) != digest:
                raise ValueError(f"Phase 5 output hash mismatch: {name}")
    scm = load_phase5_scm(a5)
    if scm.schema_version != 2 or scm.environment != phase4["environment"]:
        raise ValueError("Phase 5 SCM is not aligned to timing-v2 Phase 4")
    controller = CounterfactualController(scm, read_json(a5 / "safety_config.json"))
    splits = ("train", "validation", "test") if include_test else ("train", "validation")
    data = {split: load_phase4_split(root, phase4, split) for split in splits}
    calibration = calibrate_environment(data["train"], phase4)
    if scm.calibration != calibration.to_dict():
        raise ValueError("Phase 5 SCM calibration differs from training-only calibration")
    features = _causal_features(root, phase4)
    normalizer = fit_normalizer(data["train"], phase4, features, calibration)
    model = load_phase4_agent(a4 / "best_model.zip", device="cpu")
    if model.observation_space.shape != (68,) or model.action_space.n != 6:
        raise ValueError("Corrected DQN dimensions differ")
    return phase4, phase5, data, calibration, controller, normalizer, model


def train_ablation(data, calibration, phase4, representation: str, normalizer: dict,
                   seed: int, steps: int, output: Path, reports: Path):
    torch.set_num_threads(1)
    settings = copy.deepcopy(phase4)
    settings["dqn"]["total_timesteps"] = steps
    if steps <= settings["dqn"]["learning_starts"]:
        settings["dqn"]["learning_starts"] = max(20, steps // 10)
    settings["dqn"]["evaluation_frequency"] = min(settings["dqn"]["evaluation_frequency"], max(100, steps // 4))
    train = Monitor(make_env(data["train"], calibration, settings, representation, normalizer),
                    filename=str(reports / f"{representation}_{seed}.monitor.csv"))
    # Use the same deterministic fixed-start selection as the corrected Phase 4 run.
    valid_starts = list(EdgeIIoTRoutingEnv(data["validation"], calibration, settings).valid_starts)
    count = int(settings["dqn"]["evaluation_start_count"])
    selected = [valid_starts[i] for i in np.linspace(0, len(valid_starts) - 1, min(count, len(valid_starts)), dtype=int)]
    callback = FixedStartEvaluationCallback(
        lambda: make_env(data["validation"], calibration, settings, representation, normalizer),
        selected, settings["dqn"]["evaluation_frequency"], output / f"{representation}_{seed}", seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    agent = build_dqn(train, settings, reports / "tensorboard", seed)
    agent.learn(total_timesteps=steps, callback=callback, tb_log_name=f"{representation}_{seed}")
    pd.DataFrame(callback.history).assign(representation=representation, seed=seed).to_csv(
        reports / f"{representation}_{seed}_validation.csv", index=False)
    path = output / f"{representation}_{seed}.zip"
    if not path.is_file():
        raise RuntimeError(f"No validation checkpoint produced: {path}")
    return load_phase4_agent(path, device="cpu")


def train_worker(root: Path, phase4: dict, representation: str, seed: int,
                 steps: int, artifacts: Path, reports: Path) -> str:
    """Isolated seed job; never reads validation/test outside checkpoint selection."""
    train = load_phase4_split(root, phase4, "train")
    validation = load_phase4_split(root, phase4, "validation")
    calibration = calibrate_environment(train, phase4)
    features = _causal_features(root, phase4)
    normalizer = fit_normalizer(train, phase4, features, calibration)
    train_ablation({"train": train, "validation": validation}, calibration, phase4,
                   representation, normalizer, seed, steps, artifacts, reports)
    return str(artifacts / f"{representation}_{seed}.zip")


def run_episode(data, calibration, phase4, representation, normalizer, agent, controller,
                start: int, seed: int, policy: str, multiplier: float, onset: int, duration: int):
    env = make_env(data, calibration, phase4, representation, normalizer)
    base = base_env(env)
    factors, shift_reached = scenario_multipliers(base, start, multiplier, onset, duration)
    base.demand_multiplier = factors
    observation, _ = env.reset(seed=seed, options={"start_index": start})
    rows = []
    explanations = []
    step = 0
    while True:
        source_index = base.current_index
        state = DecisionState(base.announced_demand, tuple(base.queues), data.split,
            str(base.current_row.graph_file), base.current_row.window_start.isoformat(),
            controller.scm.calibration_id, schema_version=2,
            observation_graph_file=str(base.observation_row.graph_file),
            observation_window_start=base.observation_row.window_start.isoformat())
        feasible = False
        fallback = False
        query_seconds = 0.0
        if policy in ("dqn", "screened_dqn"):
            predicted_action, _ = agent.predict(observation, deterministic=True)
            action = int(np.asarray(predicted_action).item())
        elif policy == "least_queue":
            action = int(np.argmin(observation[-3:]))
        else:
            action = int(np.random.default_rng(seed + step).integers(0, 6))
        q_values = None
        if policy == "screened_dqn":
            agent.policy.set_training_mode(False)
            with torch.no_grad():
                tensor, _ = agent.policy.obs_to_tensor(observation)
                q_values = agent.q_net(tensor).cpu().numpy()[0]
            decision = controller.select_action(base._observation(), state, q_values)
            if action != decision.proposed_action:
                raise RuntimeError("DQN predict disagrees with online Q ranking")
            action = decision.executed_action
            feasible = decision.safety_feasible
            fallback = not feasible
            query_seconds = decision.query_seconds
            explanation = decision.to_dict()
            explanation.update({"split": data.split, "start_index": start, "step": step,
                                "observation_graph_file": state.observation_graph_file,
                                "demand_graph_file": state.graph_file, "demand_packets": state.demand_packets})
            explanations.append(explanation)
            predicted = decision.outcomes[action]
        else:
            outcomes = controller.scm.evaluate_actions(state)
            feasible = any(not controller.assess(outcome)[0] for outcome in outcomes)
            predicted = outcomes[action]
        before_queues = base.queues.copy()
        observation, reward, terminated, truncated, info = env.step(action)
        for metric in METRICS:
            if not np.isclose(getattr(predicted, metric), info[metric], atol=1e-6, rtol=1e-8):
                raise RuntimeError(f"Exact prediction mismatch: {metric}")
        if not np.allclose(predicted.post_queue, info["post_queue"], atol=1e-6, rtol=1e-8):
            raise RuntimeError("Exact post-action queue mismatch")
        if policy == "screened_dqn":
            explanation["realized"] = {metric: float(info[metric]) for metric in METRICS}
            explanation["realized_post_queue"] = info["post_queue"].tolist()
        reasons = controller.assess(predicted)[0]
        if policy == "screened_dqn" and feasible and reasons:
            raise RuntimeError("Screened policy violated a feasible state")
        rows.append({"split": data.split, "start_index": start, "step": step,
                     "observation_graph_file": state.observation_graph_file,
                     "demand_graph_file": state.graph_file, "window_start": state.window_start,
                     "demand_multiplier": float(factors[source_index]),
                     "demand_packets": state.demand_packets, "action": action,
                     "pre_queue_total": float(before_queues.sum()), "post_queue_total": float(base.queues.sum()),
                     "safety_feasible": feasible, "violated": bool(reasons), "no_safe_action": fallback,
                     "query_seconds": query_seconds, **{metric: float(info[metric]) for metric in METRIC_FIELDS}})
        step += 1
        if terminated or truncated:
            break
    return rows, explanations, shift_reached


def summarize(steps: pd.DataFrame, episodes: pd.DataFrame) -> dict:
    result = {}
    keys = ["split", "scenario", "representation", "policy", "seed"]
    for identity, group in steps.groupby(keys, sort=True):
        paired = episodes.loc[(episodes[keys] == pd.Series(identity, index=keys)).all(axis=1)]
        item = {metric: float(group[metric].mean()) for metric in METRIC_FIELDS}
        item.update({"steps": len(group), "episodes": len(paired),
                     "mean_episode_reward": float(paired.cumulative_reward.mean()),
                     "feasible_fraction": float(group.safety_feasible.mean()),
                     "feasible_violation_count": int((group.safety_feasible & group.violated).sum()),
                     "no_safe_action_fraction": float(group.no_safe_action.mean()),
                     "mean_query_seconds": float(group.query_seconds.mean()),
                     "action_counts": np.bincount(group.action.astype(int), minlength=6).tolist()})
        result["/".join(map(str, identity))] = item
    return result


def bootstrap_differences(episodes: pd.DataFrame, resamples: int) -> dict:
    rng = np.random.default_rng(42)
    results = {}
    for identity, part in episodes.groupby(["split", "scenario", "representation", "seed"]):
        base = part.loc[part.policy.eq("dqn")].set_index("start_index")
        screened = part.loc[part.policy.eq("screened_dqn")].set_index("start_index")
        if not base.index.equals(screened.index):
            raise RuntimeError("Paired episode starts differ")
        diffs = (screened.cumulative_reward - base.cumulative_reward).to_numpy()
        draws = rng.choice(diffs, size=(resamples, len(diffs)), replace=True).mean(axis=1)
        results["/".join(map(str, identity))] = {
            "mean_screened_minus_unscreened_reward": float(diffs.mean()),
            "descriptive_interval_95": np.quantile(draws, [.025, .975]).tolist(),
            "episodes": len(diffs), "interpretation": "descriptive; one shared traffic stream"}
    return results


def shift_recovery(rows: list[dict], onset: int, duration: int) -> dict:
    """First post-surge window with recovered service and no excess queue."""
    if len(rows) <= onset:
        return {"shift_reached": False, "recovery_steps": None, "recovery_censored": True}
    reference = rows[onset]["pre_queue_total"]
    for row in rows[onset + duration:]:
        if row["pdr"] >= .99 and row["post_queue_total"] <= reference + 1e-6:
            return {"shift_reached": True, "recovery_steps": row["step"] - onset - duration + 1,
                    "recovery_censored": False}
    return {"shift_reached": True, "recovery_steps": None, "recovery_censored": True}


def boundary_audit(root: Path, phase4: dict) -> dict:
    metadata = read_json(root / "artifacts/preprocessing_metadata.json")
    frame = pd.read_csv(root / phase4["causal_dataset_csv"])
    overlaps = frame.groupby("window_start")["split"].nunique()
    shared = overlaps.loc[overlaps.gt(1)].index.tolist()
    return {"split_boundaries": metadata["split_boundaries"],
            "shared_window_starts": shared,
            "status": "unchanged_by_user_choice; not fully window-isolated"}


def audit_outputs(root: Path, artifacts: Path, reports: Path) -> None:
    manifest = read_json(reports / "run_manifest.json")
    required_reports = ("evaluation_episodes.csv", "evaluation_steps.csv", "shift_trajectories.csv",
                        "shift_summary.csv", "policy_metrics.json", "paired_intervals.json",
                        "boundary_audit.json", "action_distribution.csv", "explanations.jsonl",
                        "review_report.md", "plot_status.json")
    for name in required_reports:
        if not (reports / name).is_file() or (reports / name).stat().st_size == 0:
            raise ValueError(f"Missing or empty Phase 6 report: {name}")
    if not (artifacts / "causal_normalization.json").is_file():
        raise ValueError("Missing train-only causal normalizer")
    for name, digest in manifest["inputs"].items():
        if file_sha256(root / name) != digest:
            raise ValueError(f"Changed input: {name}")
    for name, digest in manifest["outputs"].items():
        path = Path(name) if Path(name).is_absolute() else root / name
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Missing or changed output: {name}")
    episodes = pd.read_csv(reports / "evaluation_episodes.csv")
    steps = pd.read_csv(reports / "evaluation_steps.csv")
    if episodes.empty or steps.empty or (steps.safety_feasible & steps.violated & steps.policy.eq("screened_dqn")).any():
        raise ValueError("Invalid Phase 6 evaluation records")
    if not steps.groupby(["split", "scenario", "representation", "seed", "policy", "demand_graph_file"]).size().eq(1).all():
        raise ValueError("Primary evaluation overlaps a demand graph")
    if not manifest["smoke_test"]:
        phase4 = manifest["phase4_config"]
        train = load_phase4_split(root, phase4, "train")
        calibration = calibrate_environment(train, phase4)
        for split in ("validation", "test"):
            data = load_phase4_split(root, phase4, split)
            expected = set(data.frame.iloc[EdgeIIoTRoutingEnv(data, calibration, phase4).valid_starts].graph_file)
            normal = steps.loc[steps.split.eq(split) & steps.scenario.eq("normal")]
            for _, group in normal.groupby(["representation", "seed", "policy"]):
                if set(group.demand_graph_file) != expected:
                    raise ValueError(f"Incomplete {split} decision-window coverage")
    if not read_json(reports / "boundary_audit.json")["shared_window_starts"]:
        raise ValueError("Known shared boundary windows missing from report")
    explanations = 0
    with (reports / "explanations.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if len(record["outcomes"]) != 6 or sorted(outcome["action"] for outcome in record["outcomes"]) != list(range(6)):
                raise ValueError("Incomplete six-action explanation")
            chosen = record["outcomes"][record["executed_action"]]
            if any(not np.isclose(chosen[metric], record["realized"][metric], atol=1e-6, rtol=1e-8)
                   for metric in METRICS):
                raise ValueError("Explanation prediction disagrees with realized outcome")
            explanations += 1
    if explanations != int(steps.policy.eq("screened_dqn").sum()):
        raise ValueError("Screened decision/explanation count mismatch")
    plot = read_json(reports / "plot_status.json")
    if plot["status"] == "complete":
        from PIL import Image
        for name in plot["files"]:
            with Image.open(reports / name) as image:
                image.verify()
    LOG.info("ALL_PHASE6_CHECKS_PASSED")


def execute(root: Path, config: dict, artifacts: Path, reports: Path, smoke: bool) -> None:
    phase4, phase5, data, calibration, controller, normalizer, gnn_42 = validate_inputs(root, config, include_test=False)
    write_json(artifacts / "causal_normalization.json", normalizer)
    seeds = [42] if smoke else [int(seed) for seed in config["training_seeds"]]
    steps_budget = 1000 if smoke else int(config["training_timesteps"])
    agents = {}
    jobs = [(representation, seed) for representation in REPRESENTATIONS for seed in seeds
            if not (representation == "gnn" and seed == 42 and not smoke)]
    if not smoke and 42 in seeds:
        agents[("gnn", 42)] = gnn_42
    workers = min(2 if smoke else int(config.get("max_training_workers", 4)), len(jobs))
    if workers == 1:
        for representation, seed in jobs:
            LOG.info("Training Phase 6 %s DQN seed=%s steps=%s", representation, seed, steps_budget)
            agents[(representation, seed)] = train_ablation(data, calibration, phase4, representation,
                normalizer, seed, steps_budget, artifacts, reports)
    else:
        LOG.info("Training %s Phase 6 DQNs with %s isolated workers.", len(jobs), workers)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(train_worker, root, phase4, representation, seed,
                        steps_budget, artifacts, reports): (representation, seed)
                       for representation, seed in jobs}
            for future in as_completed(futures):
                representation, seed = futures[future]
                path = future.result()
                agents[(representation, seed)] = load_phase4_agent(path, device="cpu")
                LOG.info("Finished Phase 6 %s DQN seed=%s", representation, seed)
    # Test enters only after all policy checkpoints have been fixed on validation.
    test = load_phase4_split(root, phase4, "test")
    data["test"] = test
    scenarios = [("normal", 1.)] + [(f"surge_{factor:g}x", float(factor)) for factor in config["shift_multipliers"]]
    if smoke:
        scenarios = scenarios[:2]
    step_records = []
    episode_records = []
    explanations = []
    for split in ("validation", "test"):
        source = data[split]
        starts = decision_blocks(EdgeIIoTRoutingEnv(source, calibration, phase4))
        if smoke:
            starts = starts[:1]
        for scenario, multiplier in scenarios:
            for representation in REPRESENTATIONS:
                for seed in seeds:
                    agent = agents[(representation, seed)]
                    policies = ("dqn", "screened_dqn", "random", "least_queue") if representation == "gnn" else ("dqn", "screened_dqn")
                    for policy in policies:
                        for episode, start in enumerate(starts):
                            rows, details, reached = run_episode(source, calibration, phase4, representation,
                                normalizer, agent, controller, start, seed + episode, policy, multiplier,
                                int(config["shift_onset_step"]), int(config["shift_duration_steps"]))
                            for row in rows:
                                row.update({"scenario": scenario, "representation": representation,
                                            "policy": policy, "seed": seed, "episode": episode, "shift_reached": reached})
                            step_records.extend(rows)
                            for detail in details:
                                detail.update({"scenario": scenario, "representation": representation,
                                               "seed": seed, "episode": episode})
                            explanations.extend(details)
                            episode_records.append({"split": split, "scenario": scenario,
                                "representation": representation, "policy": policy, "seed": seed,
                                "episode": episode, "start_index": start, "steps": len(rows),
                                "shift_reached": reached if multiplier != 1 else False,
                                **(shift_recovery(rows, int(config["shift_onset_step"]),
                                    int(config["shift_duration_steps"])) if multiplier != 1 else
                                    {"recovery_steps": None, "recovery_censored": False}),
                                "cumulative_reward": float(sum(row["reward"] for row in rows))})
    step_frame = pd.DataFrame(step_records)
    episode_frame = pd.DataFrame(episode_records)
    step_frame.to_csv(reports / "evaluation_steps.csv", index=False)
    episode_frame.to_csv(reports / "evaluation_episodes.csv", index=False)
    step_frame.loc[step_frame.scenario.ne("normal")].to_csv(reports / "shift_trajectories.csv", index=False)
    with (reports / "explanations.jsonl").open("w", encoding="utf-8") as handle:
        for detail in explanations:
            handle.write(json.dumps(detail, allow_nan=False) + "\n")
    write_json(reports / "policy_metrics.json", summarize(step_frame, episode_frame))
    write_json(reports / "paired_intervals.json", bootstrap_differences(episode_frame, int(config["bootstrap_resamples"])))
    write_json(reports / "boundary_audit.json", boundary_audit(root, phase4))
    shifted = step_frame.loc[step_frame.scenario.ne("normal") & step_frame.shift_reached]
    post_onset = shifted.loc[shifted.step.ge(int(config["shift_onset_step"]))]
    shift_metrics = post_onset.groupby(["split", "scenario", "representation", "policy", "seed"])[
        ["reward", "pdr", "drop_ratio", "post_queue_total", "latency_proxy_ms", "energy_proxy"]
    ].mean().reset_index()
    shift_metrics.to_csv(reports / "shift_summary.csv", index=False)
    pd.DataFrame(step_frame.groupby(["split", "scenario", "representation", "policy", "seed", "action"]).size()
                 .rename("count").reset_index()).to_csv(reports / "action_distribution.csv", index=False)
    review = ["# Phase 6 Review", "", "## Observed simulator results", "",
              "| Representation | Policy | Mean reward/action | Immediate service | Latency proxy (ms) | Energy proxy |",
              "|---|---|---:|---:|---:|---:|"]
    test_normal = step_frame.loc[step_frame.split.eq("test") & step_frame.scenario.eq("normal")]
    for (representation, policy), part in test_normal.groupby(["representation", "policy"], sort=True):
        review.append(f"| {representation} | {policy} | {part.reward.mean():.4f} | {part.pdr.mean():.4f} | "
                      f"{part.latency_proxy_ms.mean():.1f} | {part.energy_proxy.mean():.3f} |")
    review += ["", "Values aggregate all configured training seeds on one shared test stream.",
              "See paired_intervals.json for descriptive paired intervals and shift_summary.csv for post-onset surges.",
              "", "## Simulator assumptions",
              "The current packet batch is announced before routing. Historical graph embeddings are not modified during synthetic demand surges.",
              "Latency and energy are simulator proxies. PDR means immediate service of current arrivals.",
              "", "## Unsupported claims",
              "No real-network safety, measured latency or energy gain, topology attack detection, or Phase 2 DAG action effect is established.",
              "The two Phase 1 split boundaries remain inside shared graph windows."]
    (reports / "review_report.md").write_text("\n".join(review) + "\n", encoding="utf-8")
    plot_status = {"status": "disabled", "files": []}
    if config["generate_plots"]:
        try:
            import matplotlib.pyplot as plt
            test_normal = step_frame.loc[step_frame.split.eq("test") & step_frame.scenario.eq("normal")]
            chart = test_normal.groupby(["representation", "policy"])["reward"].mean().sort_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            chart.plot.bar(ax=ax)
            ax.set_ylabel("Mean simulator reward per action")
            fig.tight_layout()
            fig.savefig(reports / "policy_comparison.png", dpi=160)
            plt.close(fig)
            plot_status = {"status": "complete", "files": ["policy_comparison.png"]}
        except Exception as error:
            LOG.warning("Optional Phase 6 plotting failed: %s", error)
            plot_status = {"status": "warning", "error": str(error), "files": []}
    write_json(reports / "plot_status.json", plot_status)
    inputs = [root / config["phase4_config"], root / config["phase5_config"],
              root / "config/phase6_config.json",
              root / "src/rl_environment.py", root / "src/counterfactual_controller.py",
              root / phase4["phase4_artifacts_dir"] / "best_model.zip",
              root / phase4["phase4_reports_dir"] / "run_manifest.json",
              root / phase5["artifacts_dir"] / "exact_scm.json",
              root / phase5["reports_dir"] / "run_manifest.json",
              root / phase4["causal_dataset_csv"]]
    inputs += [root / phase4["phase3_artifacts_dir"] / "embeddings" / f"{split}.pt" for split in ("train", "validation", "test")]
    outputs = [path for folder in (artifacts, reports) for path in folder.rglob("*") if path.is_file() and path.name != "run_manifest.json"]
    write_json(reports / "run_manifest.json", {"config": config,
        "timing_schema_version": 2, "phase4_config": phase4,
        "smoke_test": smoke,
        "torch_cpu_threads": torch.get_num_threads(),
        "inputs": {str(path.relative_to(root)): file_sha256(path) for path in inputs},
        "outputs": {str(path.relative_to(root)) if path.is_relative_to(root) else str(path): file_sha256(path) for path in outputs},
        "test_usage": "after all validation checkpoint selections",
        "uncertainty": "descriptive paired episode bootstrap; common traffic stream"})
    audit_outputs(root, artifacts, reports)


def run(config_path: Path, validate_only=False, smoke_test=False, audit_only=False) -> None:
    root = Path.cwd().resolve()
    config = read_json(config_path)
    artifacts = (root / config["artifacts_dir"]).resolve()
    reports = (root / config["reports_dir"]).resolve()
    if artifacts != (root / "artifacts/phase6_v3").resolve() or reports != (root / "reports/phase6_v3").resolve():
        raise ValueError("Phase 6 output paths must be confined to Phase 6 directories")
    if audit_only:
        audit_outputs(root, artifacts, reports)
        return
    if validate_only:
        validate_inputs(root, config, include_test=True)
        LOG.info("PHASE6_INPUT_CHECKS_PASSED")
        return
    if smoke_test:
        with tempfile.TemporaryDirectory(prefix="phase6_smoke_",dir=root) as temporary:
            a, r = Path(temporary) / "artifacts", Path(temporary) / "reports"
            a.mkdir(); r.mkdir()
            execute(root, config, a, r, smoke=True)
        LOG.info("PHASE6_SMOKE_TEST_PASSED")
        return
    for directory in (artifacts, reports):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing non-empty Phase 6 output: {directory}")
        directory.mkdir(parents=True, exist_ok=True)
    execute(root, config, artifacts, reports, smoke=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/phase6_config.json"))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--smoke-test", action="store_true")
    modes.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.config, args.validate_only, args.smoke_test, args.audit_only)


if __name__ == "__main__":
    main()
