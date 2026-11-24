from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.analyze_paper_experiment_suite import (
    aggregate,
    paired_tests,
    patient_means,
    statistical_design_audit,
    validate_suite_completeness,
    write_partial_diagnostics,
)


class PatientLevelAnalysisTests(unittest.TestCase):
    def synthetic_rows(self):
        rows = []
        modes = ["ppo", "ppo_shield", "mpc", "mpc_shield", "switch", "switch_shield"]
        for patient_index in range(1, 6):
            for repeat in range(5):
                for mode_index, mode in enumerate(modes):
                    rows.append(
                        {
                            "experiment": "primary",
                            "split_label": "validation",
                            "role": "primary",
                            "meal_announcement": "unannounced",
                            "patient": f"adult#{patient_index:03d}",
                            "mode": mode,
                            "repeat": repeat,
                            "tir": 70.0 + patient_index + mode_index * 0.1,
                            "tbr70": 3.0,
                            "tbr54": 1.0,
                            "tar180": 20.0,
                            "tar250": 5.0,
                            "mean_bg": 150.0,
                            "total_insulin_u": 30.0,
                        }
                    )
        return pd.DataFrame(rows)

    def test_repeats_collapse_to_five_patient_units(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            patients = patient_means(self.synthetic_rows(), out_dir)
            overall = aggregate(patients, out_dir)
            tests = paired_tests(patients, out_dir)
            self.assertTrue((overall["n_patients"] == 5).all())
            self.assertTrue((tests["n_pairs"] == 5).all())

    def test_missing_manifest_run_fails_completeness(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            out_dir = suite / "analysis"
            out_dir.mkdir()
            exp_dir = suite / "primary"
            exp_dir.mkdir()
            (exp_dir / "completion.json").write_text(
                json.dumps({"complete": False, "expected_runs": 2, "completed_runs": 1}),
                encoding="utf-8",
            )
            manifest = {
                "primary": {
                    "patients": "adult#001,adult#002",
                    "modes": "switch_shield",
                    "repeats": 1,
                }
            }
            df = pd.DataFrame(
                [
                    {
                        "experiment": "primary",
                        "patient": "adult#001",
                        "mode": "switch_shield",
                        "repeat": 0,
                        "tir": 80.0,
                        "tbr70": 2.0,
                        "tbr54": 1.0,
                        "tar180": 18.0,
                        "tar250": 4.0,
                        "nonfinite_steps": 0,
                        "cgm_nonfinite_steps": 0,
                        "raw_steps": 12,
                        "completed_steps": 12,
                        "expected_steps": 12,
                        "terminated_early": False,
                        "terminal_event": False,
                        "claim_valid": True,
                    }
                ]
            )
            issues = validate_suite_completeness(suite, manifest, df, out_dir)
            self.assertTrue(issues)
            report = pd.read_csv(out_dir / "suite_completeness.csv")
            self.assertEqual(int(report.loc[0, "missing_rows"]), 1)
            self.assertFalse(bool(report.loc[0, "complete"]))

    def test_statistical_audit_flags_five_patient_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            patients = patient_means(self.synthetic_rows(), out_dir)
            audit = statistical_design_audit(patients, out_dir, "primary")
            self.assertEqual(audit["n_independent_patients"], 5)
            self.assertAlmostEqual(audit["minimum_two_sided_exact_p"], 0.0625)
            self.assertEqual(audit["status"], "LIMITED")
            self.assertFalse(audit["claim_gate_passed"])
            self.assertTrue(any("validation" in item for item in audit["warnings"]))

    def test_terminal_at_expected_horizon_is_still_invalid(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            out_dir = suite / "analysis"
            exp_dir = suite / "primary"
            out_dir.mkdir()
            exp_dir.mkdir()
            fingerprint = "locked-v3"
            (exp_dir / "completion.json").write_text(
                json.dumps(
                    {
                        "execution_complete": True,
                        "expected_runs": 1,
                        "completed_runs": 1,
                        "invalid_runs": 1,
                        "claim_valid": False,
                        "protocol_fingerprint": fingerprint,
                    }
                ),
                encoding="utf-8",
            )
            (exp_dir / "config.json").write_text(
                json.dumps(
                    {
                        "protocol_version": "causal_fair_v3",
                        "protocol_fingerprint": fingerprint,
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "primary": {
                    "patients": "adult#001",
                    "modes": "switch_shield",
                    "repeats": 1,
                    "protocol_fingerprint": fingerprint,
                }
            }
            row = {
                "experiment": "primary",
                "patient": "adult#001",
                "mode": "switch_shield",
                "repeat": 0,
                "seed": 1,
                "tir": 80.0,
                "tbr70": 2.0,
                "tbr54": 1.0,
                "tar180": 18.0,
                "tar250": 4.0,
                "nonfinite_steps": 0,
                "cgm_nonfinite_steps": 0,
                "raw_steps": 12,
                "completed_steps": 12,
                "expected_steps": 12,
                "terminated_early": False,
                "terminal_event": True,
                "termination_reason": "bg_below_10",
                "terminal_bg": 9.0,
                "claim_valid": False,
            }
            df = pd.DataFrame([row])
            issues = validate_suite_completeness(suite, manifest, df, out_dir)
            self.assertTrue(issues)
            counts = write_partial_diagnostics(df, manifest, out_dir)
            self.assertEqual(counts["terminal_runs"], 1)
            self.assertEqual(counts["invalid_runs"], 1)
            validity = pd.read_csv(out_dir / "diagnostic_run_validity.csv")
            self.assertTrue(bool(validity.loc[0, "censored"]))
            failures = pd.read_csv(out_dir / "diagnostic_failure_counts.csv")
            self.assertEqual(failures.loc[0, "termination_reason"], "bg_below_10")
            self.assertEqual(int(failures.loc[0, "expected_runs"]), 1)

    def test_censored_diagnostic_arm_is_complete_but_not_claim_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            out_dir = suite / "analysis"
            exp_dir = suite / "raw_diagnostic"
            out_dir.mkdir()
            exp_dir.mkdir()
            fingerprint = "locked-v3-diagnostic"
            (exp_dir / "completion.json").write_text(
                json.dumps(
                    {
                        "execution_complete": True,
                        "expected_runs": 1,
                        "completed_runs": 1,
                        "invalid_runs": 1,
                        "claim_valid": False,
                        "protocol_fingerprint": fingerprint,
                    }
                ),
                encoding="utf-8",
            )
            (exp_dir / "config.json").write_text(
                json.dumps(
                    {
                        "protocol_version": "causal_fair_v3",
                        "protocol_fingerprint": fingerprint,
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "raw_diagnostic": {
                    "patients": "adult#001",
                    "modes": "mpc",
                    "repeats": 1,
                    "role": "diagnostic",
                    "protocol_fingerprint": fingerprint,
                }
            }
            df = pd.DataFrame(
                [
                    {
                        "experiment": "raw_diagnostic",
                        "patient": "adult#001",
                        "mode": "mpc",
                        "repeat": 0,
                        "tir": 20.0,
                        "tbr70": 0.0,
                        "tbr54": 0.0,
                        "tar180": 80.0,
                        "tar250": 70.0,
                        "nonfinite_steps": 0,
                        "cgm_nonfinite_steps": 0,
                        "raw_steps": 8,
                        "completed_steps": 8,
                        "expected_steps": 12,
                        "terminated_early": True,
                        "terminal_event": True,
                        "claim_valid": False,
                    }
                ]
            )
            issues = validate_suite_completeness(suite, manifest, df, out_dir)
            self.assertEqual(issues, [])
            report = pd.read_csv(out_dir / "suite_completeness.csv")
            self.assertTrue(bool(report.loc[0, "complete"]))
            self.assertFalse(bool(report.loc[0, "claim_required"]))

    def test_v4_requires_bound_provenance_and_zero_mpc_authorization_violations(self):
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            out_dir = suite / "analysis"
            exp_dir = suite / "compat"
            out_dir.mkdir()
            exp_dir.mkdir()
            fingerprint = "locked-v4"
            hashes = {
                "checkpoint_sha256": "checkpoint",
                "arx_param_sha256": "arx",
                "source_tree_sha256": "source",
                "requirements_lock_sha256": "lock",
                "runtime_fingerprint": "runtime",
                "compatibility_calibration_sha256": "calibration",
                "compatibility_reference_arx_sha256": "arx",
            }
            (exp_dir / "completion.json").write_text(
                json.dumps(
                    {
                        "execution_complete": True,
                        "expected_runs": 1,
                        "completed_runs": 1,
                        "invalid_runs": 0,
                        "claim_valid": True,
                        "protocol_fingerprint": fingerprint,
                    }
                ),
                encoding="utf-8",
            )
            (exp_dir / "config.json").write_text(
                json.dumps(
                    {
                        "protocol_version": "causal_model_compat_v4",
                        "protocol_fingerprint": fingerprint,
                        "model_condition": "nominal",
                        "checkpoint_sha256": hashes["checkpoint_sha256"],
                        "arx_param_sha256": hashes["arx_param_sha256"],
                        "source_tree_sha256": hashes["source_tree_sha256"],
                        "requirements_lock_sha256": hashes[
                            "requirements_lock_sha256"
                        ],
                        "runtime_fingerprint_actual": hashes[
                            "runtime_fingerprint"
                        ],
                        "compatibility_calibration_sha256": hashes[
                            "compatibility_calibration_sha256"
                        ],
                        "compatibility_reference_arx_sha256": hashes[
                            "compatibility_reference_arx_sha256"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            manifest = {
                "compat": {
                    "protocol_version": "causal_model_compat_v4",
                    "patients": "adult#006",
                    "modes": "switch_shield",
                    "repeats": 1,
                    "role": "primary",
                    "model_condition": "nominal",
                    "protocol_fingerprint": fingerprint,
                    **hashes,
                }
            }
            row = {
                "experiment": "compat",
                "patient": "adult#006",
                "mode": "switch_shield",
                "repeat": 0,
                "tir": 80.0,
                "tbr70": 2.0,
                "tbr54": 1.0,
                "tar180": 18.0,
                "tar250": 4.0,
                "nonfinite_steps": 0,
                "cgm_nonfinite_steps": 0,
                "raw_steps": 12,
                "completed_steps": 12,
                "expected_steps": 12,
                "terminated_early": False,
                "terminal_event": False,
                "claim_valid": True,
                "model_residual_valid_rate": 0.9,
                "model_residual_consistency_max_mgdl": 0.0,
                "model_npe_ewma_mean": 1.0,
                "model_compatibility_score_mean": 0.4,
                "model_compatible_rate": 0.5,
                "mpc_model_authorized_rate": 0.5,
                "mpc_authorization_violations": 0,
                "compatibility_gate_enabled": True,
                "compatibility_calibration_id": "compat-test",
            }
            valid_df = pd.DataFrame([row])
            self.assertEqual(
                validate_suite_completeness(
                    suite, manifest, valid_df, out_dir
                ),
                [],
            )
            invalid_df = valid_df.copy()
            invalid_df.loc[0, "mpc_authorization_violations"] = 1
            self.assertTrue(
                validate_suite_completeness(
                    suite, manifest, invalid_df, out_dir
                )
            )
            missing_id = valid_df.copy()
            missing_id.loc[0, "compatibility_calibration_id"] = ""
            self.assertTrue(
                validate_suite_completeness(
                    suite, manifest, missing_id, out_dir
                )
            )

    def test_claim_gate_requires_six_patients_and_locked_test(self):
        rows = self.synthetic_rows()
        sixth = rows[rows["patient"].eq("adult#005")].copy()
        sixth["patient"] = "adult#006"
        rows = pd.concat([rows, sixth], ignore_index=True)
        rows["split_label"] = "test"
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            patients = patient_means(rows, out_dir)
            audit = statistical_design_audit(patients, out_dir, "primary")
            self.assertEqual(audit["n_independent_patients"], 6)
            self.assertTrue(audit["claim_gate_passed"])


if __name__ == "__main__":
    unittest.main()
