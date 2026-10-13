"""Regression tests for the announced-batch, previous-graph decision contract."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from stable_baselines3.common.env_checker import check_env

from src.counterfactual_controller import ActionAwareSCM, CounterfactualController, DecisionState
from src.rl_environment import EdgeIIoTRoutingEnv, EnvironmentCalibration, Phase4SplitData
from src.rl_agent import build_dqn


ROOT = Path(__file__).resolve().parents[1]


def fixture(gap=False):
    config = json.loads((ROOT / "config/phase4_v2_config.json").read_text())
    times = pd.to_datetime(["2021-01-01T00:00:00", "2021-01-01T00:01:00",
                            "2021-01-01T00:03:00" if gap else "2021-01-01T00:02:00"])
    frame = pd.DataFrame({"graph_file": ["a.pt", "b.pt", "c.pt"], "window_start": times,
                          "total_packets": [2.0, 7.0, 11.0]})
    embeddings = np.zeros((3, 64), dtype=np.float32)
    embeddings[:, 0] = [1, 2, 3]
    data = Phase4SplitData("train", embeddings, frame, "fixture")
    calibration = EnvironmentCalibration(np.array([10., 8., 5.]), np.array([20., 16., 10.]),
        np.array([5., 10., 20.]), np.array([1., .65, .35]), np.array([2.5, 1.4, .7]),
        15000., .99, 60., demand_log_scale=float(np.log1p(10)))
    return EdgeIIoTRoutingEnv(data, calibration, config)


class TimingV2Tests(unittest.TestCase):
    def test_previous_graph_and_announced_demand(self):
        env = fixture()
        observation, info = env.reset(options={"start_index": 1})
        self.assertEqual(observation.shape, (68,))
        self.assertEqual(info["observation_graph_file"], "a.pt")
        self.assertEqual(info["graph_file"], "b.pt")
        self.assertAlmostEqual(observation[0], 1)
        self.assertAlmostEqual(observation[64], np.log1p(7) / np.log1p(10))
        original = observation.copy()
        env.data.embeddings[1, 0] = 99
        np.testing.assert_array_equal(env._observation(), original)
        env.demand_multiplier[1] = 2
        self.assertGreater(env._observation()[64], original[64])
        self.assertEqual(env.simulate_action(0).demand_packets, 14)

    def test_last_window_executes_then_rejects_more_actions(self):
        env = fixture()
        env.reset(options={"start_index": 2})
        observation, _, terminated, truncated, info = env.step(0)
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertIsNone(info["next_graph_file"])
        self.assertTrue(env.observation_space.contains(observation))
        with self.assertRaises(RuntimeError):
            env.step(0)
        with self.assertRaises(RuntimeError):
            env.simulate_action(0)

    def test_gap_never_uses_cross_segment_history(self):
        env = fixture(gap=True)
        self.assertEqual(env.valid_starts, [1])
        env.reset(options={"start_index": 1})
        _, _, _, truncated, info = env.step(3)
        self.assertTrue(truncated)
        self.assertTrue(info["segment_ended"])

    def test_exact_scm_and_gym_checker(self):
        env = fixture()
        check_env(env, warn=True)
        env.reset(options={"start_index": 1})
        scm = ActionAwareSCM(env.calibration.to_dict(), env.config["environment"])
        state = DecisionState(env.announced_demand, tuple(env.queues), "train", "b.pt",
            env.current_row.window_start.isoformat(), scm.calibration_id,
            schema_version=2, observation_graph_file="a.pt",
            observation_window_start=env.observation_row.window_start.isoformat())
        outcome = scm.intervene(state, 4)
        realized = env.simulate_action(4)
        self.assertAlmostEqual(outcome.reward, realized.reward)
        controller = CounterfactualController(scm, json.loads((ROOT / "config/phase5_v2_config.json").read_text())["safety"])
        decision = controller.select_action(env._observation(), state, np.arange(6, dtype=float))
        self.assertEqual(len(decision.outcomes), 6)

    def test_truncations_do_not_bootstrap_from_placeholder(self):
        env = fixture()
        model = build_dqn(env, env.config, None, 42)
        self.assertFalse(model.replay_buffer.handle_timeout_termination)


if __name__ == "__main__":
    unittest.main()
