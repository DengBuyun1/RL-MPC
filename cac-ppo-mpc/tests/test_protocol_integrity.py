from __future__ import annotations

import unittest
from datetime import datetime, timedelta
import json
import tempfile
from pathlib import Path
import pandas as pd
from types import SimpleNamespace

from ppo_mpc.controllers import (
    PpoMpcSwitchController,
    SharedSafetyGovernorController,
    SwitchConfig,
    require_finite_cgm,
)
from ppo_mpc.envs import GlucoseStateTracker
from ppo_mpc.metrics import summarize_closed_loop
from ppo_mpc.paths import add_asset_paths
from ppo_mpc.provenance import (
    assert_runtime_matches_lock,
    parse_exact_requirements_lock,
    sha256_file,
)
from ppo_mpc.safety import SafetyGovernorConfig, apply_safety_governor
from ppo_mpc.scenarios import ScenarioA, aggregate_same_minute_meals
from scripts.evaluate_ppo_mpc import (
    causal_controller_info,
    load_compatibility_calibration,
    patient_specific_theta_overlap,
)


add_asset_paths()
from simglucose.simulation.env import is_terminal_bg  # noqa: E402
from simglucose.controller.base import Action  # noqa: E402


class ConstantMealScenario:
    def get_action(self, _time):
        return SimpleNamespace(meal=10.0)


class DummyStatefulController:
    def __init__(self):
        self.tracker = SimpleNamespace(
            last_basal_rate=0.01,
            insulin_history=[0.0, 0.07],
        )
        self.uhat_hist = [2.0]
        self.workspace_commits = []
        self.history = []

    def reset(self):
        self.history = []

    def _commit_workspace_zone_control(self, value):
        self.workspace_commits.append(float(value))

    def policy(self, _observation, _reward, _done, **_kwargs):
        self.history.append(
            {"selected": "ppo", "selected_basal": 0.05, "selected_bolus": 0.02}
        )
        return Action(basal=0.05, bolus=0.02)


class CausalProtocolTests(unittest.TestCase):
    def make_env(self):
        return SimpleNamespace(
            sample_time=5.0,
            patient=SimpleNamespace(name="adult#006"),
            time=datetime(2018, 1, 1, 8, 0),
            scenario=ConstantMealScenario(),
        )

    def test_unannounced_strips_oracle_meal_and_private_fields(self):
        raw = {"meal": 12.0, "iob": 3.5, "secret_oracle": 99, "sample_time": 5.0}
        payload = causal_controller_info(raw, self.make_env(), "unannounced")
        self.assertEqual(payload["meal"], 0.0)
        self.assertEqual(payload["announced_meal"], 0.0)
        self.assertFalse(payload["meal_is_announced"])
        self.assertNotIn("iob", payload)
        self.assertNotIn("secret_oracle", payload)

    def test_nonfinite_cgm_does_not_fall_back_to_true_bg(self):
        with self.assertRaisesRegex(ValueError, "true BG fallback is prohibited"):
            require_finite_cgm(SimpleNamespace(CGM=float("nan"), BG=123.0))

    def test_iob_uses_insulin_units_and_the_actual_sample_grid(self):
        tracker = GlucoseStateTracker(body_weight=70.0, basal_rate=0.01, sample_time=5.0)
        tracker.insulin_history.clear()
        for _ in range(tracker.insulin_history.maxlen - 1):
            tracker.insulin_history.append(0.0)
        tracker.record_action(0.2, 0.0)
        self.assertAlmostEqual(tracker.current_iob(), 1.0, places=6)

        tracker.record_action(0.0, 0.0)
        self.assertAlmostEqual(tracker.current_iob(), float(tracker.iob_tail[1]), places=6)

        one_minute = GlucoseStateTracker(70.0, 0.01, sample_time=1.0)
        one_minute.insulin_history.clear()
        for _ in range(one_minute.insulin_history.maxlen - 1):
            one_minute.insulin_history.append(0.0)
        one_minute.record_action(1.0, 0.0)
        self.assertAlmostEqual(one_minute.current_iob(), 1.0, places=6)

    def test_steady_state_iob_has_units_and_matches_seeded_history(self):
        tracker = GlucoseStateTracker(70.0, 0.01, sample_time=5.0)
        tracker.reset(120.0)
        self.assertGreater(tracker.steady_state_iob(), 60.0 * tracker.basal_rate)
        self.assertAlmostEqual(
            tracker.current_iob(), tracker.steady_state_iob(), places=6
        )

    def test_announced_routes_future_grams_as_rate(self):
        payload = causal_controller_info({}, self.make_env(), "announced")
        self.assertEqual(payload["announced_meal"], 50.0)
        self.assertEqual(payload["meal"], 10.0)
        self.assertTrue(payload["meal_is_announced"])

    def test_terminal_glucose_boundaries(self):
        for value in [float("nan"), float("inf"), -float("inf"), 9.9, 600.1]:
            self.assertTrue(is_terminal_bg(value))
        for value in [10.0, 110.0, 600.0]:
            self.assertFalse(is_terminal_bg(value))

    def test_multiday_scenario_resamples_daily_schedule(self):
        start = datetime(2018, 1, 1, 0, 0)
        scenario = ScenarioA(start, seed=123, patient_name="adult#006")
        first_day = list(scenario.meals)
        scenario.get_action(start + timedelta(days=1))
        second_day = scenario._meals_by_day[(start + timedelta(days=1)).date()]
        self.assertNotEqual(first_day, second_day)

    def test_same_minute_meals_preserve_total_carbohydrate(self):
        combined = aggregate_same_minute_meals(
            [(580, 93.001), (580, 12.153), (600, 20.0)]
        )
        self.assertEqual([minute for minute, _amount in combined], [580, 600])
        self.assertAlmostEqual(combined[0][1], 105.154, places=6)

    def test_patient_specific_theta_overlap_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "theta.json"
            path.write_text(
                json.dumps(
                    {
                        "__population__": {"theta": [1.0, 0.0, 0.0, 0.0]},
                        "adult#006": {"theta": [1.0, 0.0, 0.0, 0.0]},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                patient_specific_theta_overlap(path, ["adult#006", "adult#007"]),
                {"adult#006"},
            )

    def test_compatibility_calibration_provenance_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "compatibility.json"
            payload = {
                "schema_version": "causal_arx_compat_v1",
                "source_patients": [
                    f"adult#{index:03d}" for index in range(1, 6)
                ],
                "source_mode": "switch_nogate_shield",
                "split_label": "development",
                "calibration_id": "compat-unit-test",
                "gate": {
                    "schema_version": "causal_arx_compat_v1",
                    "calibration_id": "compat-unit-test",
                    "residual_center_mgdl": 0.0,
                    "residual_scale_mgdl": 1.0,
                    "ewma_tau_min": 30.0,
                    "enter_threshold_npe": 1.0,
                    "exit_threshold_npe": 2.0,
                },
                "provenance": {
                    "source_tree_sha256": "source",
                    "checkpoint_sha256": "checkpoint",
                    "arx_param_sha256": "reference-arx",
                    "requirements_lock_sha256": "lock",
                },
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded, config, _digest, _resolved = load_compatibility_calibration(
                path,
                expected_sha256=sha256_file(path),
                current_source_sha256="source",
                current_checkpoint_sha256="checkpoint",
                patients=["adult#006"],
                reference_arx_sha256="reference-arx",
                current_requirements_lock_sha256="lock",
            )
            self.assertEqual(loaded["calibration_id"], "compat-unit-test")
            self.assertEqual(config.calibration_id, "compat-unit-test")
            with self.assertRaisesRegex(SystemExit, "reference ARX"):
                load_compatibility_calibration(
                    path,
                    expected_sha256=sha256_file(path),
                    current_source_sha256="source",
                    current_checkpoint_sha256="checkpoint",
                    patients=["adult#006"],
                    reference_arx_sha256="wrong",
                    current_requirements_lock_sha256="lock",
                )
            with self.assertRaisesRegex(SystemExit, "overlap"):
                load_compatibility_calibration(
                    path,
                    expected_sha256=sha256_file(path),
                    current_source_sha256="source",
                    current_checkpoint_sha256="checkpoint",
                    patients=["adult#001"],
                    reference_arx_sha256="reference-arx",
                    current_requirements_lock_sha256="lock",
                )

    def test_runtime_lock_is_not_a_self_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            mismatch = Path(directory) / "mismatch.txt"
            mismatch.write_text(
                "# python-version==0.0.0\nnumpy==0.0.0\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "runtime does not match"):
                assert_runtime_matches_lock(mismatch)

            ranged = Path(directory) / "ranged.txt"
            ranged.write_text("numpy>=1\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exact pins"):
                parse_exact_requirements_lock(ranged)

    def test_clinical_metrics_exclude_the_precontrol_initial_sample(self):
        frame = pd.DataFrame(
            {
                "BG": [100.0, 200.0],
                "CGM": [100.0, 200.0],
                "insulin": [0.01, float("nan")],
            }
        )
        summary = summarize_closed_loop(
            frame,
            patient="adult#006",
            mode="mpc",
            scenario="A",
            seed=1,
        )
        self.assertEqual(summary["raw_steps"], 1)
        self.assertEqual(summary["tir"], 0.0)
        self.assertEqual(summary["mean_bg"], 200.0)

    def test_shared_shield_is_controller_independent(self):
        config = SafetyGovernorConfig(enabled=True)
        kwargs = dict(
            glucose=70.0,
            glucose_rate=-0.5,
            basal=0.05,
            bolus=0.02,
            basal_reference=0.01,
            config=config,
        )
        outputs = [apply_safety_governor(**kwargs) for _ in ["ppo", "mpc", "switch"]]
        self.assertTrue(all(output == outputs[0] for output in outputs))
        self.assertTrue(outputs[0].overridden)
        self.assertEqual(outputs[0].basal, 0.0)
        self.assertEqual(outputs[0].bolus, 0.0)

    def test_shared_shield_synchronizes_the_executed_action(self):
        base = DummyStatefulController()
        wrapped = SharedSafetyGovernorController(
            base,
            SafetyGovernorConfig(enabled=True),
            controller_family="ppo",
        )
        action = wrapped.policy(
            SimpleNamespace(CGM=70.0, BG=70.0),
            0.0,
            False,
            patient_name="adult#006",
            sample_time=5.0,
        )

        self.assertEqual(float(action.basal), 0.0)
        self.assertEqual(float(action.bolus), 0.0)
        self.assertEqual(base.tracker.last_basal_rate, 0.0)
        self.assertEqual(base.tracker.insulin_history[-1], 0.0)
        self.assertAlmostEqual(base.uhat_hist[-1], -1.0)
        self.assertLess(base.workspace_commits[-1], 0.0)
        self.assertTrue(wrapped.history[-1]["safety_override"])
        self.assertFalse(wrapped.history[-1]["ppo_effective"])

    def test_risk_selector_never_authorizes_the_higher_insulin_candidate(self):
        controller = object.__new__(PpoMpcSwitchController)
        controller.config = SwitchConfig(
            warmup_steps=0,
            risk_low_glucose=90.0,
            risk_current_glucose=130.0,
            risk_projected_glucose=130.0,
            risk_fast_fall_rate=-2.0,
        )
        controller.basal_rate = 0.01
        controller.step_index = 12
        ppo_low = Action(basal=0.02, bolus=0.0)
        mpc_high = Action(basal=0.20, bolus=0.0)
        selected, reason, risk = controller._decide(270.0, -2.8, ppo_low, mpc_high)
        self.assertEqual(selected, "ppo")
        self.assertTrue(risk)
        self.assertIn("lower_insulin", reason)

        ppo_high = Action(basal=0.08, bolus=0.0)
        mpc_low = Action(basal=0.02, bolus=0.0)
        selected, reason, risk = controller._decide(125.0, -0.1, ppo_high, mpc_low)
        self.assertEqual(selected, "mpc")
        self.assertTrue(risk)
        self.assertIn("lower_insulin", reason)

    def test_incompatible_model_blocks_mpc_only_in_risk_arbitration(self):
        controller = object.__new__(PpoMpcSwitchController)
        controller.config = SwitchConfig(
            warmup_steps=0,
            risk_low_glucose=90.0,
            risk_current_glucose=130.0,
            risk_projected_glucose=130.0,
            risk_fast_fall_rate=-2.0,
        )
        controller.basal_rate = 0.01
        controller.step_index = 12
        blocked = controller._decide(
            125.0,
            -0.1,
            Action(basal=0.08, bolus=0.0),
            Action(basal=0.02, bolus=0.0),
            model_authorized=False,
        )
        self.assertEqual(blocked[0], "ppo")
        self.assertTrue(blocked[2])
        self.assertIn("blocked_incompatible", blocked[1])

        nominal = controller._decide(
            200.0,
            0.0,
            Action(basal=0.08, bolus=0.0),
            Action(basal=0.02, bolus=0.0),
            model_authorized=False,
        )
        self.assertEqual(nominal, ("ppo", "nominal_ppo", False))

    def test_bidirectional_selector_uses_higher_candidate_during_high_rising_risk(self):
        controller = object.__new__(PpoMpcSwitchController)
        controller.config = SwitchConfig(
            warmup_steps=0,
            high_risk_arbitration=True,
            high_risk_glucose=180.0,
            high_risk_projected_glucose=200.0,
            high_risk_rising_rate=0.1,
        )
        controller.basal_rate = 0.01
        controller.step_index = 12

        selected = controller._decide(
            220.0,
            0.4,
            Action(basal=0.02, bolus=0.0),
            Action(basal=0.08, bolus=0.0),
            model_authorized=True,
        )
        self.assertEqual(selected[0], "mpc")
        self.assertTrue(selected[2])
        self.assertIn("higher_insulin", selected[1])

        ppo_higher = controller._decide(
            220.0,
            0.4,
            Action(basal=0.08, bolus=0.0),
            Action(basal=0.02, bolus=0.0),
            model_authorized=True,
        )
        self.assertEqual(ppo_higher[0], "ppo")
        self.assertTrue(ppo_higher[2])
        self.assertIn("higher_insulin", ppo_higher[1])

    def test_bidirectional_selector_fails_closed_and_preserves_low_risk_priority(self):
        controller = object.__new__(PpoMpcSwitchController)
        controller.config = SwitchConfig(
            warmup_steps=0,
            risk_fast_fall_rate=-2.0,
            high_risk_arbitration=True,
            high_risk_glucose=180.0,
            high_risk_projected_glucose=200.0,
            high_risk_rising_rate=0.1,
        )
        controller.basal_rate = 0.01
        controller.step_index = 12

        blocked = controller._decide(
            220.0,
            0.4,
            Action(basal=0.02, bolus=0.0),
            Action(basal=0.08, bolus=0.0),
            model_authorized=False,
        )
        self.assertEqual(blocked[0], "ppo")
        self.assertTrue(blocked[2])
        self.assertIn("blocked_incompatible", blocked[1])

        falling = controller._decide(
            270.0,
            -2.8,
            Action(basal=0.02, bolus=0.0),
            Action(basal=0.08, bolus=0.0),
            model_authorized=True,
        )
        self.assertEqual(falling[0], "ppo")
        self.assertTrue(falling[2])
        self.assertIn("lower_insulin", falling[1])

    def test_high_correction_expert_raises_mpc_candidate_with_iob_credit(self):
        controller = object.__new__(PpoMpcSwitchController)
        controller.config = SwitchConfig(
            high_risk_correction_enabled=True,
            high_risk_correction_target_glucose=110.0,
            high_risk_correction_horizon_min=60.0,
            high_risk_correction_max_basal_multiplier=8.0,
        )
        controller.basal_rate = 0.01
        controller.correction_factor = 50.0
        controller.tracker = SimpleNamespace(
            current_iob=lambda: 1.0,
            steady_state_iob=lambda: 0.5,
        )

        corrected, diagnostics = controller._high_correction_candidate(
            glucose=210.0,
            action=Action(basal=0.02, bolus=0.0),
        )
        # Need=(210-110)/50=2 U, IOB credit=0.5 U, so the 60-minute
        # correction adds 0.025 U/min above the 0.01 U/min basal reference.
        self.assertAlmostEqual(corrected.basal, 0.035, places=12)
        self.assertAlmostEqual(diagnostics["residual_units"], 1.5, places=12)

        controller.tracker = SimpleNamespace(
            current_iob=lambda: 3.0,
            steady_state_iob=lambda: 0.5,
        )
        credited, credited_diagnostics = controller._high_correction_candidate(
            glucose=210.0,
            action=Action(basal=0.02, bolus=0.0),
        )
        self.assertAlmostEqual(credited.basal, 0.02, places=12)
        self.assertEqual(credited_diagnostics["residual_units"], 0.0)

    def test_high_risk_selector_returns_to_ppo_when_iob_is_already_excessive(self):
        controller = object.__new__(PpoMpcSwitchController)
        controller.config = SwitchConfig(
            warmup_steps=0,
            high_risk_arbitration=True,
            high_risk_glucose=160.0,
            high_risk_rising_rate=0.0,
            high_risk_iob_max_ratio=2.0,
        )
        controller.basal_rate = 0.01
        controller.step_index = 12
        controller.tracker = SimpleNamespace(
            current_iob=lambda: 2.1,
            steady_state_iob=lambda: 1.0,
        )
        decision = controller._decide(
            220.0,
            0.4,
            Action(basal=0.02, bolus=0.0),
            Action(basal=0.20, bolus=0.0),
            model_authorized=True,
        )
        self.assertEqual(decision[0], "ppo")
        self.assertIn("iob_limit", decision[1])

    def test_high_risk_iob_taper_linearly_blends_mpc_toward_ppo(self):
        controller = object.__new__(PpoMpcSwitchController)
        controller.config = SwitchConfig(
            high_risk_iob_taper_start_ratio=1.25,
            high_risk_iob_max_ratio=2.0,
        )
        controller.tracker = SimpleNamespace(
            current_iob=lambda: 1.625,
            steady_state_iob=lambda: 1.0,
        )
        blended, alpha, ratio = controller._high_iob_taper_candidate(
            Action(basal=0.02, bolus=0.0),
            Action(basal=0.10, bolus=0.0),
        )
        self.assertAlmostEqual(ratio, 1.625, places=12)
        self.assertAlmostEqual(alpha, 0.5, places=12)
        self.assertAlmostEqual(blended.basal, 0.06, places=12)

    def test_high_risk_iob_projection_accounts_for_pending_insulin(self):
        controller = object.__new__(PpoMpcSwitchController)
        controller.config = SwitchConfig(
            high_risk_iob_prediction_enabled=True,
            high_risk_iob_prediction_horizon_min=90.0,
            high_risk_iob_prediction_effect_fraction=0.5,
        )
        controller.correction_factor = 50.0
        controller.tracker = SimpleNamespace(
            current_iob=lambda: 3.0,
            steady_state_iob=lambda: 1.0,
        )
        projected = controller._high_iob_effect_projected_glucose(200.0, 0.0)
        self.assertAlmostEqual(projected, 150.0, places=12)

    def test_enabled_gate_requires_a_frozen_configuration(self):
        with self.assertRaisesRegex(ValueError, "frozen"):
            PpoMpcSwitchController(
                object(),
                object(),
                SwitchConfig(compatibility_gate_enabled=True),
            )

    def test_action_gap_and_legacy_low_band_are_diagnostic_only(self):
        controller = object.__new__(PpoMpcSwitchController)
        controller.config = SwitchConfig(
            warmup_steps=0,
            max_action_gap_basal=0.01,
            ppo_min_glucose=1000.0,
            ppo_min_projected_glucose=1000.0,
            ppo_min_rate=1000.0,
        )
        controller.basal_rate = 0.01
        controller.step_index = 12
        decision = controller._decide(
            200.0,
            0.0,
            Action(basal=0.20, bolus=0.0),
            Action(basal=0.01, bolus=0.0),
        )
        self.assertEqual(decision, ("ppo", "nominal_ppo", False))

    def test_multihorizon_shield_acts_when_30_minute_screen_is_not_low(self):
        decision = apply_safety_governor(
            glucose=140.0,
            glucose_rate=-0.6,
            basal=0.05,
            bolus=0.0,
            basal_reference=0.01,
            config=SafetyGovernorConfig(enabled=True),
            iob_units=1.0,
            basal_iob_units=1.0,
        )
        self.assertGreater(decision.projected_glucose_30min, 90.0)
        self.assertLessEqual(decision.projected_glucose_90min, 90.0)
        self.assertEqual(decision.basal, 0.0)
        self.assertIn("multihorizon", decision.reason)

    def test_higher_causal_iob_cannot_increase_executed_insulin(self):
        kwargs = dict(
            glucose=150.0,
            glucose_rate=-0.25,
            basal=0.05,
            bolus=0.0,
            basal_reference=0.01,
            config=SafetyGovernorConfig(enabled=True),
            basal_iob_units=1.0,
        )
        baseline = apply_safety_governor(**kwargs, iob_units=1.0)
        excess = apply_safety_governor(**kwargs, iob_units=1.5)
        self.assertLessEqual(excess.basal, baseline.basal)
        self.assertEqual(excess.basal, 0.0)
        self.assertEqual(excess.reason, "safety_cap_iob_excess_falling")
        with self.assertRaisesRegex(ValueError, "non-finite causal IOB"):
            apply_safety_governor(**kwargs, iob_units=float("nan"))

    def test_severe_high_guard_uses_a_bounded_prescription_correction_floor(self):
        decision = apply_safety_governor(
            glucose=400.0,
            glucose_rate=0.0,
            basal=0.01,
            bolus=0.0,
            basal_reference=0.01,
            config=SafetyGovernorConfig(enabled=True),
            iob_units=1.0,
            basal_iob_units=1.0,
            correction_factor_mgdl_per_u=50.0,
        )
        expected = 0.01 + ((400.0 - 180.0) / 50.0) / 180.0
        self.assertAlmostEqual(decision.basal, expected, places=8)
        self.assertEqual(decision.floor_basal, decision.basal)
        self.assertLessEqual(decision.basal, 5.0 * 0.01)
        self.assertEqual(decision.reason, "safety_floor_severe_high")
        stacked = apply_safety_governor(
            glucose=400.0,
            glucose_rate=0.0,
            basal=0.01,
            bolus=0.0,
            basal_reference=0.01,
            config=SafetyGovernorConfig(enabled=True),
            iob_units=4.0,
            basal_iob_units=1.0,
            correction_factor_mgdl_per_u=50.0,
        )
        self.assertLess(stacked.basal, decision.basal)
        self.assertAlmostEqual(stacked.correction_iob_credit_units, 3.0)
        with self.assertRaisesRegex(ValueError, "correction factor"):
            apply_safety_governor(
                glucose=400.0,
                glucose_rate=0.0,
                basal=0.01,
                bolus=0.0,
                basal_reference=0.01,
                config=SafetyGovernorConfig(enabled=True),
                correction_factor_mgdl_per_u=float("nan"),
            )


if __name__ == "__main__":
    unittest.main()
