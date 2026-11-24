from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_paper_experiment_suite import (
    CALIBRATION_MODES,
    DIAGNOSTIC_MODES,
    FAIR_MODES,
    MISMATCH_MODES,
    build_experiments,
    experiment_is_complete,
)


class SuiteMatrixTests(unittest.TestCase):
    def make_args(self):
        return SimpleNamespace(
            scope="paper",
            checkpoint_unannounced="runs/unann/model_final.pt",
            checkpoint_announced="runs/ann/model_final.pt",
            arx_param_file=(
                "mpc_assets/output/arx_rls_training_30/"
                "population_theta_adult001_005.json"
            ),
            mild_arx_param_file=(
                "mpc_assets/output/arx_rls_training_30/"
                "population_theta_adult_child_blend_050.json"
            ),
            severe_arx_param_file=(
                "mpc_assets/output/arx_rls_training_30/"
                "population_theta_child001_005.json"
            ),
            compatibility_calibration_file=(
                "results/paper_compat_v4_calibration/analysis/"
                "compatibility_calibration.json"
            ),
            requirements_lock="requirements-lock-server-py310-cu126.txt",
            device="cpu",
            smoke_hours=3.0,
            smoke_start_time="2018-01-01 08:00",
            calibration_hours=24.0,
            calibration_repeats=3,
            calibration_seed=71000,
            hours=72.0,
            repeats=5,
            patients=",".join(
                f"adult#{idx:03d}" for idx in range(6, 11)
            ),
            start_time="2018-01-01 00:00",
            split_label="validation",
            seed=61000,
            include_ablations=False,
            include_generalization=False,
            include_announced=False,
            generalization_hours=72.0,
            generalization_repeats=3,
        )

    def test_paper_scope_is_the_prespecified_275_row_matrix(self):
        experiments = build_experiments(self.make_args())
        self.assertEqual(len(experiments), 3)
        by_condition = {item.model_condition: item for item in experiments}
        self.assertEqual(set(by_condition), {
            "nominal", "mild_mismatch", "severe_mismatch"
        })
        self.assertEqual(by_condition["nominal"].modes, FAIR_MODES)
        self.assertEqual(by_condition["nominal"].expected_rows, 100)
        self.assertEqual(by_condition["mild_mismatch"].modes, MISMATCH_MODES)
        self.assertEqual(by_condition["mild_mismatch"].expected_rows, 75)
        self.assertEqual(by_condition["severe_mismatch"].modes, FAIR_MODES)
        self.assertEqual(by_condition["severe_mismatch"].expected_rows, 100)
        self.assertEqual(sum(item.expected_rows for item in experiments), 275)
        self.assertFalse(
            any(item.role == "diagnostic" for item in experiments)
        )
        self.assertEqual(DIAGNOSTIC_MODES, "ppo,mpc,switch_nogate")

    def test_patient_list_whitespace_is_normalized(self):
        args = self.make_args()
        args.patients = "adult#006, adult#007,adult#008 , adult#009,adult#010"
        experiments = build_experiments(args)
        self.assertTrue(
            all(
                experiment.patients
                == "adult#006,adult#007,adult#008,adult#009,adult#010"
                for experiment in experiments
            )
        )

    def test_gate_and_nogate_are_explicit_and_nonduplicated(self):
        experiments = build_experiments(self.make_args())
        nominal_modes = experiments[0].modes.split(",")
        self.assertIn("switch_shield", nominal_modes)
        self.assertIn("switch_nogate_shield", nominal_modes)
        self.assertEqual(len(nominal_modes), len(set(nominal_modes)))
        self.assertTrue(experiments[0].gate_enabled)
        command = experiments[0].command(Path("suite"), "cpu")
        self.assertIn("--enable-switch-compatibility-gate", command)
        self.assertIn("--compatibility-calibration-file", command)
        self.assertIn("--compatibility-reference-arx-sha256", command)

    def test_calibration_is_development_only_and_has_15_rows(self):
        args = self.make_args()
        args.scope = "calibration"
        experiments = build_experiments(args)
        self.assertEqual(len(experiments), 1)
        experiment = experiments[0]
        self.assertEqual(experiment.role, "calibration")
        self.assertEqual(experiment.split_label, "development")
        self.assertEqual(experiment.patients, ",".join(
            f"adult#{idx:03d}" for idx in range(1, 6)
        ))
        self.assertEqual(experiment.modes, CALIBRATION_MODES)
        self.assertEqual(experiment.expected_rows, 15)
        self.assertIsNone(experiment.compatibility_calibration_file)
        self.assertFalse(experiment.gate_enabled)

    def test_optional_legacy_ablations_are_not_in_default_matrix(self):
        args = self.make_args()
        self.assertFalse(
            any(item.role == "ablation" for item in build_experiments(args))
        )
        args.include_ablations = True
        experiments = build_experiments(args)
        ablations = [item for item in experiments if item.role == "ablation"]
        self.assertEqual(len(ablations), 5)
        self.assertTrue(all(item.expected_rows == 25 for item in ablations))
        self.assertEqual(len({tuple(item.extra_args) for item in ablations}), 5)

    def test_resume_rejects_invalid_or_stale_completion_marker(self):
        experiment = build_experiments(self.make_args())[0]
        with tempfile.TemporaryDirectory() as temporary:
            suite = Path(temporary)
            output = suite / experiment.name
            output.mkdir()
            marker = {
                "complete": True,
                "execution_complete": True,
                "expected_runs": experiment.expected_rows,
                "completed_runs": experiment.expected_rows,
                "invalid_runs": 0,
                "claim_valid": True,
                "protocol_fingerprint": experiment.protocol_fingerprint,
            }
            config = {
                "protocol_version": "causal_model_compat_v4",
                "protocol_fingerprint": experiment.protocol_fingerprint,
                "checkpoint_sha256": experiment.checkpoint_sha256,
                "arx_param_sha256": experiment.arx_param_sha256,
                "source_tree_sha256": experiment.source_tree_sha256,
                "requirements_lock_sha256": experiment.requirements_lock_sha256,
                "runtime_fingerprint_actual": experiment.runtime_fingerprint,
                "compatibility_calibration_sha256": (
                    experiment.compatibility_calibration_sha256
                ),
                "compatibility_reference_arx_sha256": (
                    experiment.compatibility_reference_arx_sha256
                ),
                "model_condition": experiment.model_condition,
                "arx_prior_label": experiment.arx_prior_label,
            }
            (output / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            rows = []
            for patient in experiment.patients.split(","):
                patient_index = int(patient.split("#")[-1])
                for mode in experiment.modes.split(","):
                    for repeat in range(experiment.repeats):
                        rows.append(
                            {
                                "patient": patient,
                                "mode": mode,
                                "repeat": repeat,
                                "seed": (
                                    experiment.seed
                                    + 1000 * repeat
                                    + patient_index
                                ),
                                "claim_valid": True,
                            }
                        )
            with (output / "summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (output / "completion.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            self.assertTrue(experiment_is_complete(suite, experiment))

            with (output / "summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows[:-1])
            self.assertFalse(experiment_is_complete(suite, experiment))
            with (output / "summary.csv").open(
                "w", encoding="utf-8", newline=""
            ) as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            config["protocol_fingerprint"] = "stale"
            (output / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )
            self.assertFalse(experiment_is_complete(suite, experiment))
            config["protocol_fingerprint"] = experiment.protocol_fingerprint
            (output / "config.json").write_text(
                json.dumps(config), encoding="utf-8"
            )

            marker["protocol_fingerprint"] = "stale"
            (output / "completion.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            self.assertFalse(experiment_is_complete(suite, experiment))

            marker["protocol_fingerprint"] = experiment.protocol_fingerprint
            marker["claim_valid"] = False
            marker["invalid_runs"] = 1
            (output / "completion.json").write_text(
                json.dumps(marker), encoding="utf-8"
            )
            self.assertFalse(experiment_is_complete(suite, experiment))

    def test_full_scope_generalization_excludes_training_adults(self):
        args = self.make_args()
        args.scope = "full"
        experiments = build_experiments(args)
        generalization = [
            item for item in experiments if item.role == "generalization"
        ]
        self.assertEqual(len(generalization), 2)
        self.assertEqual(
            {item.patients.split("#", 1)[0] for item in generalization},
            {"adolescent", "child"},
        )
        self.assertTrue(
            all("adult#001" not in item.patients for item in generalization)
        )


if __name__ == "__main__":
    unittest.main()
