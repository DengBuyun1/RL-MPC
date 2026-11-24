import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "finall"
    / "tools"
    / "analyze_guard140_screen.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_guard140_screen", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_frame(tir_delta=1.0, tbr_delta=0.1, switch_only_termination=False):
    rows = []
    for patient_index in range(1, 11):
        patient = f"adolescent#{patient_index:03d}"
        for repeat in range(3):
            seed = 104400 + 1000 * repeat + patient_index
            rows.extend(
                [
                    {
                        "patient": patient,
                        "mode": "ppo_shield",
                        "repeat": repeat,
                        "seed": seed,
                        "cgm_tir": 70.0,
                        "cgm_tbr70": 0.5,
                        "cgm_tbr54": 0.0,
                        "cgm_tar180": 29.5,
                        "cgm_tar250": 10.0,
                        "cgm_mean_bg": 170.0,
                        "terminated_early": False,
                    },
                    {
                        "patient": patient,
                        "mode": "switch_nogate_shield",
                        "repeat": repeat,
                        "seed": seed,
                        "cgm_tir": 70.0 + tir_delta,
                        "cgm_tbr70": 0.5 + tbr_delta,
                        "cgm_tbr54": 0.0,
                        "cgm_tar180": 29.5 - tir_delta - tbr_delta,
                        "cgm_tar250": 9.0,
                        "cgm_mean_bg": 165.0,
                        "terminated_early": (
                            switch_only_termination
                            and patient_index == 1
                            and repeat == 0
                        ),
                    },
                ]
            )
    return pd.DataFrame(rows)


class Guard140ScreenAnalysisTests(unittest.TestCase):
    def test_accepts_candidate_that_meets_all_frozen_criteria(self):
        report, paired = MODULE.analyze_frame(make_frame())
        self.assertEqual(report["decision"], "ADVANCE")
        self.assertEqual(len(paired), 10)
        self.assertTrue(all(item["passed"] for item in report["criteria"]))

    def test_rejects_excess_patient_level_tbr_even_if_mean_is_small(self):
        frame = make_frame(tbr_delta=0.0)
        mask = (
            (frame["patient"] == "adolescent#001")
            & (frame["mode"] == "switch_nogate_shield")
        )
        frame.loc[mask, "cgm_tbr70"] += 0.6
        report, _ = MODULE.analyze_frame(frame)
        self.assertEqual(report["decision"], "REJECT")
        criterion = next(
            item for item in report["criteria"]
            if item["name"] == "max_patient_cgm_tbr70_increase"
        )
        self.assertFalse(criterion["passed"])

    def test_rejects_switch_only_early_termination(self):
        report, _ = MODULE.analyze_frame(
            make_frame(switch_only_termination=True)
        )
        self.assertEqual(report["decision"], "REJECT")
        self.assertEqual(report["switch_only_early_terminations"], 1)

    def test_rejects_incomplete_matrix(self):
        with self.assertRaisesRegex(ValueError, "expected 60 rows"):
            MODULE.analyze_frame(make_frame().iloc[:-1])

    def test_cli_writes_auditable_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            make_frame().to_csv(run_dir / "summary.csv", index=False)
            report = MODULE.analyze_directory(run_dir)
            self.assertEqual(report["decision"], "ADVANCE")
            self.assertTrue((run_dir / "guard140_screen_decision.json").is_file())
            self.assertTrue((run_dir / "guard140_patient_paired.csv").is_file())


if __name__ == "__main__":
    unittest.main()
