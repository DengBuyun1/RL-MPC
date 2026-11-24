from __future__ import annotations

import math
import unittest

import pandas as pd

from ppo_mpc.compatibility import (
    CausalCompatibilityGate,
    CompatibilityGateConfig,
)
from ppo_mpc.metrics import summarize_closed_loop


def gate_config(**changes) -> CompatibilityGateConfig:
    values = {
        "residual_center_mgdl": 0.0,
        "residual_scale_mgdl": 1.0,
        "ewma_tau_min": 1e-6,
        "enter_threshold_npe": 0.5,
        "exit_threshold_npe": 2.0,
        "min_valid_samples": 2,
        "enter_confirm_steps": 2,
        "exit_confirm_steps": 2,
        "min_off_steps": 1,
        "z_clip": 10.0,
        "calibration_id": "unit-test",
    }
    values.update(changes)
    return CompatibilityGateConfig(**values)


class CompatibilityGateTests(unittest.TestCase):
    def test_config_rejects_nonpositive_scale_and_reversed_hysteresis(self):
        with self.assertRaises(ValueError):
            gate_config(residual_scale_mgdl=0.0)
        with self.assertRaises(ValueError):
            gate_config(enter_threshold_npe=2.0, exit_threshold_npe=1.0)

    def test_exact_warmup_enter_hold_and_exit_boundaries(self):
        gate = CausalCompatibilityGate(gate_config())
        first = gate.update(0.0, sample_time_min=5.0, candidate_valid=True)
        self.assertFalse(first.authorized)
        self.assertEqual(first.reason, "warmup")

        second = gate.update(0.0, sample_time_min=5.0, candidate_valid=True)
        self.assertFalse(second.authorized)
        self.assertEqual(second.good_streak, 1)

        entered = gate.update(0.0, sample_time_min=5.0, candidate_valid=True)
        self.assertTrue(entered.authorized)
        self.assertEqual(entered.transition, "enter_compatible")

        held = gate.update(1.0, sample_time_min=5.0, candidate_valid=True)
        self.assertTrue(held.authorized)
        self.assertEqual(held.transition, "")

        first_bad = gate.update(3.0, sample_time_min=5.0, candidate_valid=True)
        self.assertTrue(first_bad.authorized)
        self.assertEqual(first_bad.bad_streak, 1)
        exited = gate.update(3.0, sample_time_min=5.0, candidate_valid=True)
        self.assertFalse(exited.authorized)
        self.assertEqual(exited.transition, "exit_incompatible")

    def test_invalid_residual_or_candidate_is_immediately_fail_closed(self):
        gate = CausalCompatibilityGate(
            gate_config(min_valid_samples=1, enter_confirm_steps=1)
        )
        self.assertTrue(
            gate.update(0.0, sample_time_min=5.0, candidate_valid=True).authorized
        )
        invalid = gate.update(
            float("nan"), sample_time_min=5.0, candidate_valid=True
        )
        self.assertFalse(invalid.authorized)
        self.assertEqual(invalid.transition, "exit_invalid")
        self.assertEqual(invalid.reason, "residual_unavailable")

        candidate = gate.update(
            0.0, sample_time_min=5.0, candidate_valid=False
        )
        self.assertFalse(candidate.authorized)
        self.assertEqual(candidate.reason, "mpc_candidate_invalid")

    def test_ewma_uses_actual_sample_time(self):
        config = gate_config(
            ewma_tau_min=10.0,
            min_valid_samples=1,
            enter_confirm_steps=99,
            enter_threshold_npe=0.1,
            exit_threshold_npe=9.0,
        )
        gate = CausalCompatibilityGate(config)
        gate.update(1.0, sample_time_min=5.0, candidate_valid=True)
        decision = gate.update(3.0, sample_time_min=5.0, candidate_valid=True)
        alpha = 1.0 - math.exp(-0.5)
        expected = (1.0 - alpha) * 1.0 + alpha * 3.0
        self.assertAlmostEqual(
            decision.normalized_prediction_error_ewma, expected, places=12
        )
        gate.reset()
        self.assertEqual(gate.valid_samples, 0)
        self.assertFalse(gate.compatible)


class CompatibilityMetricTests(unittest.TestCase):
    def test_residual_validity_and_authorization_violation_metrics(self):
        rollout = pd.DataFrame(
            {
                "BG": [100.0, 110.0, 120.0],
                "CGM": [100.0, 110.0, 120.0],
                "insulin": [0.01, 0.01, 0.01],
            }
        )
        history = pd.DataFrame(
            {
                "selected": ["mpc", "ppo"],
                "model_residual_raw_mgdl": [float("nan"), 2.0],
                "model_bg_pred_mgdl": [float("nan"), 100.0],
                "model_bg_meas_mgdl": [float("nan"), 102.0],
                "model_npe": [float("nan"), 1.0],
                "model_npe_ewma": [float("nan"), 1.0],
                "model_compatibility_score": [0.0, math.exp(-1.0)],
                "model_compatible": [False, False],
                "mpc_model_authorized": [False, False],
                "mpc_candidate_valid": [True, True],
                "compat_gate_transition": ["", "exit_incompatible"],
            }
        )
        switch = summarize_closed_loop(
            rollout,
            patient="adult#006",
            mode="switch_shield",
            scenario="A",
            seed=1,
            switch_history=history,
        )
        self.assertAlmostEqual(switch["model_residual_valid_rate"], 0.5)
        self.assertAlmostEqual(
            switch["model_residual_consistency_max_mgdl"], 0.0
        )
        self.assertEqual(switch["mpc_authorization_violations"], 1)
        self.assertEqual(switch["compat_gate_transition_count"], 1)

        pure_mpc = summarize_closed_loop(
            rollout,
            patient="adult#006",
            mode="mpc_shield",
            scenario="A",
            seed=1,
            switch_history=history,
        )
        self.assertEqual(pure_mpc["mpc_authorization_violations"], 0)


if __name__ == "__main__":
    unittest.main()
