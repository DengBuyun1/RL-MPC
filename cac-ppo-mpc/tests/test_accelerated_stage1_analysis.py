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
    / "analyze_accelerated_stage1.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_accelerated_stage1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_frame(tir_delta, tbr_delta, patient_tbr=None):
    rows = []
    patient_tbr = patient_tbr or {}
    for patient_index in (2, 4, 9, 10):
        patient = f"adolescent#{patient_index:03d}"
        for repeat in range(2):
            seed = 107400 + repeat * 1000 + patient_index
            for mode in ("ppo_shield", "switch_nogate_shield"):
                switch = mode == "switch_nogate_shield"
                local_tbr = patient_tbr.get(patient, tbr_delta)
                rows.append(
                    {
                        "patient": patient,
                        "mode": mode,
                        "repeat": repeat,
                        "seed": seed,
                        "cgm_tir": 70.0 + (tir_delta if switch else 0.0),
                        "cgm_tbr70": 0.5 + (local_tbr if switch else 0.0),
                        "cgm_tbr54": 0.0,
                        "cgm_tar180": 29.5
                        - (tir_delta + local_tbr if switch else 0.0),
                        "cgm_tar250": 10.0 - (0.5 if switch else 0.0),
                        "cgm_mean_bg": 170.0 - (2.0 if switch else 0.0),
                        "total_insulin_u": 30.0 + (0.5 if switch else 0.0),
                        "terminated_early": False,
                    }
                )
    return pd.DataFrame(rows)


class AcceleratedStage1AnalysisTests(unittest.TestCase):
    def test_selects_safer_passing_candidate(self):
        frames = {
            "h120_only": make_frame(0.8, 0.15),
            "h120_taper": make_frame(0.7, 0.05),
        }
        report, _ = MODULE.analyze_frames(frames)
        self.assertEqual(report["decision"], "PROMOTE_H120_TAPER")
        self.assertTrue(report["ppo_reproducible_across_candidates"])

    def test_rejects_candidate_with_one_unsafe_patient(self):
        frames = {
            "h120_only": make_frame(
                0.8, 0.0, {"adolescent#004": 0.75}
            ),
            "h120_taper": make_frame(
                0.7, 0.0, {"adolescent#009": 0.80}
            ),
        }
        report, _ = MODULE.analyze_frames(frames)
        self.assertEqual(report["decision"], "REJECT_ALL")

    def test_detects_ppo_reproducibility_failure(self):
        first = make_frame(0.8, 0.1)
        second = make_frame(0.7, 0.05)
        mask = second["mode"] == "ppo_shield"
        second.loc[mask, "cgm_tir"] += 0.01
        report, _ = MODULE.analyze_frames(
            {"h120_only": first, "h120_taper": second}
        )
        self.assertEqual(report["decision"], "REJECT_ALL")
        self.assertFalse(report["ppo_reproducible_across_candidates"])

    def test_writes_combined_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name, frame in {
                "h120_only": make_frame(0.8, 0.15),
                "h120_taper": make_frame(0.7, 0.05),
            }.items():
                directory = root / name
                directory.mkdir()
                frame.to_csv(directory / "summary.csv", index=False)
            report = MODULE.analyze_directory(root)
            self.assertEqual(report["decision"], "PROMOTE_H120_TAPER")
            self.assertTrue((root / "stage1_decision.json").is_file())
            self.assertTrue((root / "stage1_patient_paired.csv").is_file())


if __name__ == "__main__":
    unittest.main()
