"""Phase 6 ablation and evaluation schedule checks."""
import unittest
import json
from pathlib import Path

import numpy as np

from src.phase6_pipeline import AblationObservation, decision_blocks, fit_normalizer, scenario_multipliers
from src.rl_environment import EdgeIIoTRoutingEnv, calibrate_environment, load_phase4_split
from tests.test_timing_v2 import fixture


class Phase6Tests(unittest.TestCase):
    def test_ablation_context_precedes_announced_batch(self):
        env = fixture()
        env.data.frame["context"] = [10., 20., 30.]
        normalizer = fit_normalizer(env.data, env.config, ["context"], env.calibration)
        wrapped = AblationObservation(env, "causal", ["context"], normalizer)
        observation, _ = wrapped.reset(options={"start_index": 1})
        self.assertEqual(observation.shape, (5,))
        self.assertAlmostEqual(observation[0], -1.)
        self.assertAlmostEqual(observation[1], np.log1p(7) / np.log1p(10))
        env.data.frame.loc[1, "context"] = 999.
        np.testing.assert_array_equal(wrapped.observation(env._observation()), observation)
        minimal = AblationObservation(fixture(), "minimal", [], normalizer)
        self.assertEqual(minimal.reset(options={"start_index": 1})[0].shape, (4,))

    def test_nonoverlap_and_shift_schedule(self):
        env = fixture()
        self.assertEqual(decision_blocks(env), [1])
        factors, reached = scenario_multipliers(env, 1, 1.5, 1, 1)
        self.assertTrue(reached)
        np.testing.assert_array_equal(factors, [1, 1, 1.5])
        factors, reached = scenario_multipliers(env, 2, 2., 8, 16)
        self.assertFalse(reached)
        np.testing.assert_array_equal(factors, [1, 1, 1])

    def test_saved_split_pair_counts_and_disjoint_blocks(self):
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / "config/phase4_v2_config.json").read_text())
        train = load_phase4_split(root, config, "train")
        calibration = calibrate_environment(train, config)
        for split, expected in (("train", 537), ("validation", 112), ("test", 143)):
            data = train if split == "train" else load_phase4_split(root, config, split)
            env = EdgeIIoTRoutingEnv(data, calibration, config)
            self.assertEqual(len(env.valid_starts), expected)
            starts = decision_blocks(env)
            covered = []
            for start in starts:
                index = start
                for _ in range(env.episode_horizon):
                    if index >= data.size or not env.valid_decision[index]:
                        break
                    covered.append(index)
                    index += 1
            self.assertEqual(covered, env.valid_starts)


if __name__ == "__main__":
    unittest.main()
