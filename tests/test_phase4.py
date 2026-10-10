from __future__ import annotations

import json
import unittest
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

from src.phase4_pipeline import generate_balanced_transitions
from src.rl_environment import EdgeIIoTRoutingEnv, EnvironmentCalibration, Phase4SplitData


ROOT = Path(__file__).resolve().parents[1]


def phase4_config() -> dict:
    with (ROOT / "config" / "phase4_config.json").open(encoding="utf-8") as handle:
        return json.load(handle)


def split_data(split: str = "train", gap: bool = False) -> Phase4SplitData:
    starts = pd.to_datetime([
        "2021-01-01T00:00:00",
        "2021-01-01T00:01:00",
        "2021-01-01T00:03:00" if gap else "2021-01-01T00:02:00",
        "2021-01-01T00:04:00" if gap else "2021-01-01T00:03:00",
    ])
    frame = pd.DataFrame({
        "split": split,
        "graph_file": [f"graph_{index:03d}.pt" for index in range(4)],
        "window_start": starts,
        "total_packets": [12.0, 8.0, 20.0, 5.0],
        "total_bytes_proxy": [120.0, 80.0, 200.0, 50.0],
        "mean_packet_size_proxy": [10.0] * 4,
        "std_packet_size_proxy": [1.0] * 4,
        "mean_interarrival_time": [1.0] * 4,
        "std_interarrival_time": [0.2] * 4,
        "active_nodes": [3.0] * 4,
        "average_degree": [2.0] * 4,
        "average_peer_count": [1.0] * 4,
    })
    embeddings = np.arange(4 * 64, dtype=np.float32).reshape(4, 64) / 100.0
    return Phase4SplitData(split=split, embeddings=embeddings, frame=frame, encoder_sha256="test")


def calibration() -> EnvironmentCalibration:
    return EnvironmentCalibration(
        service_capacity=np.array([10.0, 8.0, 5.0]),
        buffer_capacity=np.array([20.0, 16.0, 10.0]),
        base_latency_ms=np.array([5.0, 10.0, 20.0]),
        idle_energy=np.array([1.0, 0.65, 0.35]),
        dynamic_energy=np.array([2.5, 1.4, 0.7]),
        latency_budget_ms=15000.0,
        pdr_completion_threshold=0.99,
        window_seconds=60.0,
    )


class Phase4EnvironmentTests(unittest.TestCase):
    def test_gym_contract_and_observation_bounds(self) -> None:
        env = EdgeIIoTRoutingEnv(split_data(), calibration(), phase4_config())
        check_env(env, warn=True)
        observation, _ = env.reset(seed=42, options={"start_index": 0})
        self.assertEqual(observation.shape, (67,))
        self.assertTrue(env.observation_space.contains(observation))
        self.assertEqual(env.action_space.n, 6)

    def test_hand_calculated_queue_equations(self) -> None:
        env = EdgeIIoTRoutingEnv(split_data(), calibration(), phase4_config())
        env.reset(options={"start_index": 0, "queues": [2.0, 0.0, 0.0]})
        outcome = env.simulate_action(0)
        np.testing.assert_allclose(outcome.served_old, [2.0, 0.0, 0.0])
        np.testing.assert_allclose(outcome.served_current, [8.0, 0.0, 0.0])
        np.testing.assert_allclose(outcome.post_queue, [4.0, 0.0, 0.0])
        self.assertAlmostEqual(outcome.throughput_packets, 10.0)
        self.assertAlmostEqual(outcome.pdr, 8.0 / 12.0)
        self.assertEqual(outcome.dropped_packets, 0.0)
        self.assertGreaterEqual(outcome.reward, -1.0)
        self.assertLessEqual(outcome.reward, 1.0)

    def test_simulation_is_side_effect_free_and_matches_step(self) -> None:
        env = EdgeIIoTRoutingEnv(split_data(), calibration(), phase4_config())
        env.reset(options={"start_index": 0, "queues": [3.0, 2.0, 1.0]})
        before_queue = env.queues.copy()
        before_index = env.current_index
        simulated = env.simulate_action(4)
        np.testing.assert_allclose(env.queues, before_queue)
        self.assertEqual(env.current_index, before_index)
        _, reward, _, _, info = env.step(4)
        self.assertAlmostEqual(reward, simulated.reward)
        np.testing.assert_allclose(info["post_queue"], simulated.post_queue)
        np.testing.assert_allclose(env.queues, simulated.post_queue)

    def test_gap_truncation_and_no_crossing(self) -> None:
        env = EdgeIIoTRoutingEnv(split_data(gap=True), calibration(), phase4_config())
        self.assertEqual(env.valid_starts, [0, 2])
        env.reset(options={"start_index": 0})
        _, _, terminated, truncated, info = env.step(3)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertTrue(info["segment_ended"])
        self.assertEqual(info["next_index"], 1)

    def test_seeded_resets_are_reproducible(self) -> None:
        left = EdgeIIoTRoutingEnv(split_data(), calibration(), phase4_config())
        right = EdgeIIoTRoutingEnv(split_data(), calibration(), phase4_config())
        left_observation, left_info = left.reset(seed=123)
        right_observation, right_info = right.reset(seed=123)
        np.testing.assert_array_equal(left_observation, right_observation)
        self.assertEqual(left_info["start_index"], right_info["start_index"])

    def test_balanced_transition_generation(self) -> None:
        config = phase4_config()
        config["transition_generation"] = {
            "episodes": {"train": 2, "validation": 2, "test": 2},
            "episode_horizon": 2,
        }
        datasets = {split: split_data(split) for split in ("train", "validation", "test")}
        features = [
            "total_packets", "total_bytes_proxy", "mean_packet_size_proxy",
            "std_packet_size_proxy", "mean_interarrival_time", "std_interarrival_time",
            "active_nodes", "average_degree", "average_peer_count",
        ]
        frame, audit = generate_balanced_transitions(datasets, calibration(), config, features)
        self.assertTrue(frame.groupby("state_id").size().eq(6).all())
        self.assertTrue(frame.groupby("state_id")["action"].nunique().eq(6).all())
        self.assertTrue(np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy()).all())
        self.assertTrue(audit["complete_action_coverage"])


class OneStepActionEnv(gym.Env):
    def __init__(self) -> None:
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        reward = 1.0 if int(action) == 1 else -1.0
        return np.zeros(2, dtype=np.float32), reward, True, False, {}


class Phase4LearningTests(unittest.TestCase):
    def test_dqn_learns_known_best_action(self) -> None:
        model = DQN(
            "MlpPolicy", OneStepActionEnv(), learning_rate=1e-3, buffer_size=1000,
            learning_starts=50, batch_size=32, train_freq=1, gradient_steps=1,
            target_update_interval=100, exploration_fraction=0.2,
            exploration_final_eps=0.02, policy_kwargs={"net_arch": [32]},
            seed=42, device="cpu", verbose=0,
        )
        model.learn(total_timesteps=800)
        action, _ = model.predict(np.zeros(2, dtype=np.float32), deterministic=True)
        self.assertEqual(int(action), 1)


if __name__ == "__main__":
    unittest.main()
