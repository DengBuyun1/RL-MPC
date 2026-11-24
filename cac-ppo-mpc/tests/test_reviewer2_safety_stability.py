import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "results" / "finall" / "tools" / "analyze_safety_stability.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_safety_stability", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SafetyStabilityAnalysisTests(unittest.TestCase):
    def test_episode_durations_use_five_minute_samples(self):
        module = load_module()
        values = np.array([100, 65, 60, 75, 68, 90], dtype=float)
        self.assertEqual(module.episode_durations(values < 70, 5.0), [10.0, 5.0])

    def test_high_recovery_tracks_until_target_band(self):
        module = load_module()
        values = np.array([180, 260, 240, 210, 175, 255, 230], dtype=float)
        durations, censored = module.recovery_durations(
            values, trigger=lambda x: x > 250, recovered=lambda x: x <= 180,
            sample_minutes=5.0,
        )
        self.assertEqual(durations, [15.0])
        self.assertEqual(censored, 1)

    def test_action_regularity_is_duration_normalized(self):
        module = load_module()
        metrics = module.action_regularity(np.array([1.0, 2.0, 1.0, 1.0]), 5.0)
        self.assertAlmostEqual(metrics["basal_total_variation_per_hour"], 8.0)
        self.assertEqual(metrics["basal_direction_reversals"], 1)


if __name__ == "__main__":
    unittest.main()
