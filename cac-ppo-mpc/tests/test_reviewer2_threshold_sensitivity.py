import importlib.util
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "results" / "finall" / "tools" / "analyze_threshold_sensitivity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_threshold_sensitivity", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def frame(mode, offset):
    rows = []
    for patient in ("adolescent#001", "adolescent#002"):
        for repeat in range(2):
            rows.append({
                "patient": patient,
                "mode": mode,
                "repeat": repeat,
                "seed": 1000 + repeat + int(patient[-3:]),
                "terminated_early": False,
                "cgm_tir": 70.0 + offset,
                "cgm_tbr70": 1.0,
                "cgm_tbr54": 0.0,
                "cgm_tar180": 29.0 - offset,
                "cgm_tar250": 10.0 - offset,
                "cgm_mean_bg": 170.0 - offset,
                "total_insulin_u": 50.0 + offset,
            })
    return pd.DataFrame(rows)


class ThresholdSensitivityAnalysisTests(unittest.TestCase):
    def test_patient_is_the_statistical_unit(self):
        module = load_module()
        report, setting_summary, paired = module.analyze_frames(
            frame("ppo_shield", 0.0),
            {"current": frame("switch_nogate_shield", 2.0)},
            patients=["adolescent#001", "adolescent#002"],
            repeats=2,
        )
        self.assertEqual(report["statistical_unit"], "patient_mean_over_repeats")
        self.assertEqual(len(paired), 2)
        observed = setting_summary.set_index("setting").loc["current"]
        self.assertAlmostEqual(observed["delta_cgm_tir"], 2.0)
        self.assertAlmostEqual(observed["delta_cgm_tar180"], -2.0)


if __name__ == "__main__":
    unittest.main()
