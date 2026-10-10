"""Gymnasium environment for simulated Edge-IIoT routing decisions."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from gymnasium import spaces


@dataclass(frozen=True)
class EnvironmentCalibration:
    service_capacity: np.ndarray
    buffer_capacity: np.ndarray
    base_latency_ms: np.ndarray
    idle_energy: np.ndarray
    dynamic_energy: np.ndarray
    latency_budget_ms: float
    pdr_completion_threshold: float
    window_seconds: float
    fitted_on_split: str = "train"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "service_capacity", "buffer_capacity", "base_latency_ms",
            "idle_energy", "dynamic_energy",
        ):
            payload[key] = payload[key].tolist()
        payload["server_names"] = ["server_a_performance", "server_b_balanced", "server_c_energy"]
        payload["metric_status"] = {
            "latency_proxy_ms": "synthetic queueing proxy; not measured latency",
            "energy_proxy": "synthetic energy proxy; not measured power consumption",
        }
        return payload


@dataclass(frozen=True)
class RoutingOutcome:
    action: int
    action_name: str
    allocation_fractions: np.ndarray
    demand_packets: float
    pre_queue: np.ndarray
    post_queue: np.ndarray
    arrivals: np.ndarray
    served_old: np.ndarray
    served_current: np.ndarray
    throughput_packets: float
    pdr: float
    dropped_packets: float
    drop_ratio: float
    latency_proxy_ms: float
    energy_proxy: float
    task_completion: float
    throughput_score: float
    latency_score: float
    energy_score: float
    reward: float

    def info(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "action_name": self.action_name,
            "allocation_fractions": self.allocation_fractions.copy(),
            "demand_packets": self.demand_packets,
            "pre_queue": self.pre_queue.copy(),
            "post_queue": self.post_queue.copy(),
            "arrivals": self.arrivals.copy(),
            "served_old": self.served_old.copy(),
            "served_current": self.served_current.copy(),
            "throughput_packets": self.throughput_packets,
            "pdr": self.pdr,
            "dropped_packets": self.dropped_packets,
            "drop_ratio": self.drop_ratio,
            "latency_proxy_ms": self.latency_proxy_ms,
            "energy_proxy": self.energy_proxy,
            "task_completion": self.task_completion,
            "throughput_score": self.throughput_score,
            "latency_score": self.latency_score,
            "energy_score": self.energy_score,
            "reward": self.reward,
        }


@dataclass
class Phase4SplitData:
    split: str
    embeddings: np.ndarray
    frame: pd.DataFrame
    encoder_sha256: str

    @property
    def size(self) -> int:
        return len(self.frame)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing required artifact: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_phase4_split(root: Path, config: dict[str, Any], split: str) -> Phase4SplitData:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unsupported split: {split}")
    phase3_dir = (root / config["phase3_artifacts_dir"]).resolve()
    encoder_path = phase3_dir / "encoder.pt"
    metadata_path = phase3_dir / "model_metadata.json"
    embedding_path = phase3_dir / "embeddings" / f"{split}.pt"
    causal_path = (root / config["causal_dataset_csv"]).resolve()
    for path in (encoder_path, metadata_path, embedding_path, causal_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing Phase 4 input: {path}")

    metadata = _read_json(metadata_path)
    encoder_digest = file_sha256(encoder_path)
    if metadata.get("encoder_sha256") != encoder_digest:
        raise ValueError("Current Phase 3 encoder does not match model_metadata.json.")
    payload = torch.load(embedding_path, map_location="cpu", weights_only=True)
    if payload.get("split") != split or payload.get("encoder_sha256") != encoder_digest:
        raise ValueError(f"Embedding metadata/hash mismatch for {split}.")
    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 2 or embeddings.shape[1] != 64:
        raise ValueError(f"Phase 3 {split} embeddings must have shape [N, 64].")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"Phase 3 {split} embeddings contain non-finite values.")

    causal = pd.read_csv(causal_path)
    causal = causal.loc[causal["split"].eq(split)].copy()
    if causal.duplicated(["split", "graph_file"]).any():
        raise ValueError(f"Phase 2 causal dataset contains duplicate keys for {split}.")
    by_graph = causal.set_index("graph_file", drop=False)
    graph_files = list(payload.get("graph_files", []))
    timestamps = list(payload.get("window_start", []))
    if len(graph_files) != len(embeddings) or len(timestamps) != len(embeddings):
        raise ValueError(f"Embedding identity lengths do not match tensor rows for {split}.")
    if len(set(graph_files)) != len(graph_files) or set(graph_files) != set(by_graph.index):
        raise ValueError(f"Embeddings and causal rows do not have a one-to-one key match for {split}.")
    ordered = by_graph.loc[graph_files].reset_index(drop=True)
    ordered_times = pd.to_datetime(ordered["window_start"], errors="coerce").reset_index(drop=True)
    embedding_times = pd.to_datetime(pd.Series(timestamps), errors="coerce").reset_index(drop=True)
    if ordered_times.isna().any() or embedding_times.isna().any() or not ordered_times.equals(embedding_times):
        raise ValueError(f"Embedding and causal timestamps do not match for {split}.")
    if not ordered_times.is_monotonic_increasing:
        raise ValueError(f"Phase 4 {split} rows are not chronological.")
    ordered["window_start"] = ordered_times
    numeric_columns = [column for column in ordered.columns if column not in {"split", "graph_file", "window_start"}]
    if not np.isfinite(ordered[numeric_columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Phase 2 causal inputs contain non-finite values for {split}.")
    return Phase4SplitData(
        split=split,
        embeddings=embeddings.numpy().astype(np.float32, copy=False),
        frame=ordered,
        encoder_sha256=encoder_digest,
    )


def calibrate_environment(train_data: Phase4SplitData, config: dict[str, Any]) -> EnvironmentCalibration:
    if train_data.split != "train":
        raise ValueError("Environment calibration must use the training split only.")
    env_config = config["environment"]
    demand = train_data.frame["total_packets"].to_numpy(dtype=np.float64)
    quantiles = np.asarray(env_config["capacity_quantiles"], dtype=np.float64)
    if quantiles.shape != (3,) or np.any((quantiles <= 0) | (quantiles > 1)):
        raise ValueError("capacity_quantiles must contain three values in (0, 1].")
    capacity = np.maximum(np.quantile(demand, quantiles).astype(np.float64), 1.0)
    multiplier = float(env_config["buffer_capacity_multiplier"])
    if multiplier <= 0:
        raise ValueError("buffer_capacity_multiplier must be positive.")
    calibration = EnvironmentCalibration(
        service_capacity=capacity,
        buffer_capacity=capacity * multiplier,
        base_latency_ms=np.asarray(env_config["base_latency_ms"], dtype=np.float64),
        idle_energy=np.asarray(env_config["idle_energy"], dtype=np.float64),
        dynamic_energy=np.asarray(env_config["dynamic_energy"], dtype=np.float64),
        latency_budget_ms=float(env_config["latency_budget_ms"]),
        pdr_completion_threshold=float(env_config["pdr_completion_threshold"]),
        window_seconds=float(config["window_seconds"]),
    )
    for name in ("base_latency_ms", "idle_energy", "dynamic_energy"):
        values = getattr(calibration, name)
        if values.shape != (3,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"{name} must contain three finite non-negative values.")
    return calibration


class EdgeIIoTRoutingEnv(gym.Env):
    """Discrete routing environment driven by historical exogenous traffic."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        split_data: Phase4SplitData,
        calibration: EnvironmentCalibration,
        config: dict[str, Any],
        fixed_starts: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.data = split_data
        self.calibration = calibration
        self.config = config
        env_config = config["environment"]
        self.embedding_clip = float(env_config["embedding_clip"])
        self.episode_horizon = int(env_config["episode_horizon"])
        self.actions = env_config["actions"]
        self.action_names = [str(action["name"]) for action in self.actions]
        self.action_fractions = np.asarray([action["fractions"] for action in self.actions], dtype=np.float64)
        if self.action_fractions.shape != (6, 3):
            raise ValueError("Phase 4 requires exactly six routing profiles over three servers.")
        if not np.allclose(self.action_fractions.sum(axis=1), 1.0, atol=1e-8) or np.any(self.action_fractions < 0):
            raise ValueError("Every routing profile must contain non-negative fractions summing to one.")
        self.reward_weights = {key: float(value) for key, value in env_config["reward_weights"].items()}
        if set(self.reward_weights) != {"throughput", "pdr", "completion", "latency", "energy", "drops"}:
            raise ValueError("reward_weights must define throughput, pdr, completion, latency, energy, and drops.")
        self.action_space = spaces.Discrete(6)
        low = np.concatenate([np.full(64, -self.embedding_clip), np.zeros(3)]).astype(np.float32)
        high = np.concatenate([np.full(64, self.embedding_clip), np.ones(3)]).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        gaps = self.data.frame["window_start"].shift(-1) - self.data.frame["window_start"]
        self.valid_transition = gaps.eq(pd.Timedelta(seconds=self.calibration.window_seconds)).to_numpy(dtype=bool)
        self.valid_starts = np.flatnonzero(self.valid_transition).astype(int).tolist()
        if not self.valid_starts:
            raise ValueError(f"Split {self.data.split} has no contiguous Phase 4 transitions.")
        if fixed_starts is not None:
            invalid = sorted(set(fixed_starts) - set(self.valid_starts))
            if invalid:
                raise ValueError(f"Fixed starts are not valid transition indices: {invalid}")
        self.fixed_starts = list(fixed_starts) if fixed_starts is not None else None
        self._fixed_cursor = 0
        self.current_index = self.valid_starts[0]
        self.queues = np.zeros(3, dtype=np.float64)
        self.episode_steps = 0

    @property
    def current_row(self) -> pd.Series:
        return self.data.frame.iloc[self.current_index]

    def _observation(self) -> np.ndarray:
        embedding = np.clip(self.data.embeddings[self.current_index], -self.embedding_clip, self.embedding_clip)
        queue_ratio = np.divide(
            self.queues, self.calibration.buffer_capacity,
            out=np.zeros(3, dtype=np.float64), where=self.calibration.buffer_capacity > 0,
        )
        observation = np.concatenate([embedding, np.clip(queue_ratio, 0.0, 1.0)]).astype(np.float32)
        if not self.observation_space.contains(observation):
            raise RuntimeError("Generated observation is outside the declared observation space.")
        return observation

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        if "start_index" in options:
            start = int(options["start_index"])
            if start not in self.valid_starts:
                raise ValueError(f"start_index {start} is not a valid contiguous transition.")
        elif self.fixed_starts:
            start = self.fixed_starts[self._fixed_cursor % len(self.fixed_starts)]
            self._fixed_cursor += 1
        else:
            start = int(self.np_random.choice(self.valid_starts))
        initial_queues = np.asarray(options.get("queues", np.zeros(3)), dtype=np.float64)
        if initial_queues.shape != (3,) or not np.isfinite(initial_queues).all():
            raise ValueError("Initial queues must contain three finite values.")
        if np.any(initial_queues < 0) or np.any(initial_queues > self.calibration.buffer_capacity):
            raise ValueError("Initial queues must be within configured server buffers.")
        self.current_index = start
        self.queues = initial_queues.copy()
        self.episode_steps = 0
        return self._observation(), {
            "split": self.data.split,
            "graph_file": str(self.current_row["graph_file"]),
            "window_start": self.current_row["window_start"].isoformat(),
            "start_index": start,
        }

    def simulate_action(self, action: int) -> RoutingOutcome:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid routing action: {action}")
        calibration = self.calibration
        fractions = self.action_fractions[int(action)]
        demand = max(float(self.current_row["total_packets"]), 0.0)
        pre_queue = self.queues.astype(np.float64, copy=True)
        served_old = np.minimum(pre_queue, calibration.service_capacity)
        residual_queue = pre_queue - served_old
        remaining_capacity = calibration.service_capacity - served_old
        arrivals = fractions * demand
        served_current = np.minimum(arrivals, remaining_capacity)
        unserved_current = arrivals - served_current
        available_buffer = np.maximum(calibration.buffer_capacity - residual_queue, 0.0)
        enqueued = np.minimum(unserved_current, available_buffer)
        dropped = np.maximum(unserved_current - enqueued, 0.0)
        post_queue = np.minimum(residual_queue + enqueued, calibration.buffer_capacity)

        throughput_by_server = served_old + served_current
        throughput = float(throughput_by_server.sum())
        if demand > 0:
            pdr = float(np.clip(served_current.sum() / demand, 0.0, 1.0))
            drop_ratio = float(np.clip(dropped.sum() / demand, 0.0, 1.0))
            allocation_weight = arrivals / demand
        else:
            pdr, drop_ratio = 1.0, 0.0
            allocation_weight = fractions
        waiting_ms = (
            np.divide(
                pre_queue, calibration.service_capacity,
                out=np.zeros(3, dtype=np.float64), where=calibration.service_capacity > 0,
            )
            * calibration.window_seconds * 1000.0
        )
        latency = float(
            np.dot(allocation_weight, calibration.base_latency_ms + waiting_ms)
            + (1.0 - pdr) * 2.0 * calibration.latency_budget_ms
        )
        utilization = np.divide(
            throughput_by_server, calibration.service_capacity,
            out=np.zeros(3, dtype=np.float64), where=calibration.service_capacity > 0,
        )
        active = (pre_queue + arrivals) > 0
        energy = float(np.sum(active * (calibration.idle_energy + calibration.dynamic_energy * utilization**2)))
        completion = float(pdr >= calibration.pdr_completion_threshold and latency <= calibration.latency_budget_ms)

        throughput_score = float(np.clip(throughput / calibration.service_capacity.sum(), 0.0, 1.0))
        latency_score = float(np.clip(latency / calibration.latency_budget_ms, 0.0, 1.0))
        max_energy = float(np.sum(calibration.idle_energy + calibration.dynamic_energy))
        energy_score = float(np.clip(energy / max_energy, 0.0, 1.0))
        reward = (
            self.reward_weights["throughput"] * throughput_score
            + self.reward_weights["pdr"] * pdr
            + self.reward_weights["completion"] * completion
            + self.reward_weights["latency"] * latency_score
            + self.reward_weights["energy"] * energy_score
            + self.reward_weights["drops"] * drop_ratio
        )
        reward = float(np.clip(reward, -1.0, 1.0))
        values = np.concatenate([
            post_queue, arrivals, served_old, served_current,
            np.array([throughput, pdr, dropped.sum(), drop_ratio, latency, energy, completion, reward]),
        ])
        if not np.isfinite(values).all():
            raise RuntimeError("Simulator produced a non-finite outcome.")
        return RoutingOutcome(
            action=int(action), action_name=self.action_names[int(action)],
            allocation_fractions=fractions.copy(), demand_packets=demand,
            pre_queue=pre_queue, post_queue=post_queue, arrivals=arrivals,
            served_old=served_old, served_current=served_current,
            throughput_packets=throughput, pdr=pdr, dropped_packets=float(dropped.sum()),
            drop_ratio=drop_ratio, latency_proxy_ms=latency, energy_proxy=energy,
            task_completion=completion, throughput_score=throughput_score,
            latency_score=latency_score, energy_score=energy_score, reward=reward,
        )

    def step(self, action: int):
        source_index = self.current_index
        source_row = self.current_row
        outcome = self.simulate_action(int(action))
        self.queues = outcome.post_queue.copy()
        self.episode_steps += 1
        if not self.valid_transition[source_index]:
            raise RuntimeError("Environment attempted to step across a non-contiguous timestamp.")
        self.current_index = source_index + 1
        horizon_reached = self.episode_steps >= self.episode_horizon
        segment_ended = not self.valid_transition[self.current_index]
        truncated = bool(horizon_reached or segment_ended)
        info = outcome.info()
        info.update({
            "split": self.data.split,
            "graph_file": str(source_row["graph_file"]),
            "window_start": source_row["window_start"].isoformat(),
            "next_graph_file": str(self.current_row["graph_file"]),
            "next_window_start": self.current_row["window_start"].isoformat(),
            "source_index": source_index,
            "next_index": self.current_index,
            "episode_step": self.episode_steps,
            "horizon_reached": horizon_reached,
            "segment_ended": segment_ended,
        })
        return self._observation(), outcome.reward, False, truncated, info
