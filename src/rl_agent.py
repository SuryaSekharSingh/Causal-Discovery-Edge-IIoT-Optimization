"""Stable Baselines3 DQN utilities for Phase 4."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from src.rl_environment import EdgeIIoTRoutingEnv


def select_evaluation_starts(env: EdgeIIoTRoutingEnv, count: int) -> list[int]:
    if count < 1:
        raise ValueError("evaluation_start_count must be positive.")
    valid = env.valid_starts
    if len(valid) <= count:
        return list(valid)
    positions = np.linspace(0, len(valid) - 1, num=count, dtype=int)
    return [valid[int(position)] for position in positions]


def build_dqn(
    env: Any,
    config: dict[str, Any],
    tensorboard_log: Path | None,
    seed: int,
) -> DQN:
    dqn = config["dqn"]
    return DQN(
        "MlpPolicy",
        env,
        learning_rate=float(dqn["learning_rate"]),
        buffer_size=int(dqn["buffer_size"]),
        learning_starts=int(dqn["learning_starts"]),
        batch_size=int(dqn["batch_size"]),
        gamma=float(dqn["gamma"]),
        train_freq=int(dqn["train_freq"]),
        gradient_steps=int(dqn["gradient_steps"]),
        target_update_interval=int(dqn["target_update_interval"]),
        exploration_fraction=float(dqn["exploration_fraction"]),
        exploration_final_eps=float(dqn["exploration_final_eps"]),
        policy_kwargs={"net_arch": [int(width) for width in dqn["net_arch"]]},
        tensorboard_log=str(tensorboard_log) if tensorboard_log is not None else None,
        seed=int(seed),
        device=str(dqn["device"]),
        verbose=0,
    )


def _choose_action(
    policy: str,
    observation: np.ndarray,
    step: int,
    rng: np.random.Generator,
    agent: DQN | None,
) -> int:
    if policy == "dqn":
        if agent is None:
            raise ValueError("A DQN agent is required for policy='dqn'.")
        action, _ = agent.predict(observation, deterministic=True)
        return int(np.asarray(action).item())
    if policy == "random":
        return int(rng.integers(0, 6))
    if policy == "round_robin":
        return int(step % 6)
    if policy == "least_queue":
        return int(np.argmin(observation[-3:]))
    raise ValueError(f"Unsupported evaluation policy: {policy}")


def evaluate_agent(
    agent: DQN | None,
    env: EdgeIIoTRoutingEnv,
    starts: list[int],
    policy: str = "dqn",
    seed: int = 42,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not starts:
        raise ValueError("At least one fixed evaluation start is required.")
    rng = np.random.default_rng(seed)
    episode_rows: list[dict[str, Any]] = []
    action_counts = np.zeros(6, dtype=int)
    metric_names = (
        "throughput_packets", "pdr", "dropped_packets", "drop_ratio",
        "latency_proxy_ms", "energy_proxy", "task_completion",
    )
    total_steps = 0
    metric_totals = {name: 0.0 for name in metric_names}
    for episode, start in enumerate(starts):
        observation, _ = env.reset(seed=seed + episode, options={"start_index": int(start)})
        done = False
        cumulative_reward = 0.0
        steps = 0
        episode_metrics = {name: 0.0 for name in metric_names}
        while not done:
            action = _choose_action(policy, observation, steps, rng, agent)
            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            cumulative_reward += float(reward)
            steps += 1
            action_counts[action] += 1
            for name in metric_names:
                value = float(info[name])
                episode_metrics[name] += value
                metric_totals[name] += value
        total_steps += steps
        row: dict[str, Any] = {
            "policy": policy,
            "split": env.data.split,
            "episode": episode,
            "start_index": int(start),
            "start_graph_file": str(env.data.frame.iloc[start]["graph_file"]),
            "steps": steps,
            "cumulative_reward": cumulative_reward,
            "mean_step_reward": cumulative_reward / max(steps, 1),
        }
        row.update({f"mean_{name}": total / max(steps, 1) for name, total in episode_metrics.items()})
        episode_rows.append(row)
    rewards = np.asarray([row["cumulative_reward"] for row in episode_rows], dtype=float)
    step_rewards = np.asarray([row["mean_step_reward"] for row in episode_rows], dtype=float)
    aggregate: dict[str, Any] = {
        "policy": policy,
        "split": env.data.split,
        "episodes": len(episode_rows),
        "total_steps": total_steps,
        "mean_cumulative_reward": float(rewards.mean()),
        "std_cumulative_reward": float(rewards.std(ddof=0)),
        "mean_step_reward": float(step_rewards.mean()),
        "action_counts": action_counts.tolist(),
        "action_frequencies": (action_counts / max(action_counts.sum(), 1)).tolist(),
    }
    aggregate.update({f"mean_{name}": total / max(total_steps, 1) for name, total in metric_totals.items()})
    return episode_rows, aggregate


class FixedStartEvaluationCallback(BaseCallback):
    """Evaluate on the same validation starts and save only improved checkpoints."""

    def __init__(
        self,
        env_factory: Callable[[], EdgeIIoTRoutingEnv],
        starts: list[int],
        eval_freq: int,
        best_model_path: Path,
        seed: int,
    ) -> None:
        super().__init__(verbose=0)
        self.env_factory = env_factory
        self.starts = starts
        self.eval_freq = int(eval_freq)
        self.best_model_path = best_model_path
        self.seed = int(seed)
        self.best_mean_reward = -np.inf
        self.history: list[dict[str, Any]] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq:
            return True
        env = self.env_factory()
        _, aggregate = evaluate_agent(self.model, env, self.starts, policy="dqn", seed=self.seed)
        mean_reward = float(aggregate["mean_cumulative_reward"])
        improved = mean_reward > self.best_mean_reward
        self.history.append({
            "timesteps": int(self.num_timesteps),
            "mean_cumulative_reward": mean_reward,
            "mean_step_reward": float(aggregate["mean_step_reward"]),
            "improved": improved,
        })
        if improved:
            self.best_mean_reward = mean_reward
            self.model.save(str(self.best_model_path))
        return True


def load_phase4_agent(
    path: str | Path,
    env: Any | None = None,
    device: str = "cpu",
) -> DQN:
    return DQN.load(str(path), env=env, device=device)
