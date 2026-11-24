from __future__ import annotations

import importlib.util
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .envs import (
    GlucoseStateTracker,
    basal_to_raw_action,
    map_raw_action_to_basal,
    patient_basal_rate,
    patient_correction_factor,
)
from .compatibility import (
    CausalCompatibilityGate,
    CompatibilityDecision,
    CompatibilityGateConfig,
)
from .paths import DEFAULT_ARX_THETA, MPC_ASSETS_ROOT, add_asset_paths
from .safety import SafetyGovernorConfig, apply_safety_governor

if TYPE_CHECKING:
    from .ppo import PPOAgent


add_asset_paths()

from simglucose.controller.base import Action, Controller  # noqa: E402
from simglucose.patient.t1dpatient import T1DPatient  # noqa: E402


def require_finite_cgm(observation) -> float:
    """Return the causal sensor value; never substitute simulator true BG."""
    glucose = float(getattr(observation, "CGM", np.nan))
    if not np.isfinite(glucose):
        raise ValueError("controller received non-finite CGM; true BG fallback is prohibited")
    return glucose


def update_causal_glucose_rate(
    history: deque[float], glucose: float, sample_time_min: float
) -> float:
    """Robust causal CGM slope from the current and three preceding samples."""
    history.append(float(glucose))
    if len(history) < 2:
        return 0.0
    values = np.asarray(history, dtype=float)
    adjacent_rates = np.diff(values) / max(float(sample_time_min), 1e-8)
    return float(np.median(adjacent_rates))


def load_arx_controller_class():
    module_path = MPC_ASSETS_ROOT / "ARX-RLS" / "arx_mpc_ctrller.py"
    spec = importlib.util.spec_from_file_location("ppo_mpc_arx_mpc_ctrller", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load ARX controller module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.ArxRlsVffMpcController


ArxRlsVffMpcController = load_arx_controller_class()

def commit_executed_action_to_mpc(controller, action: Action, basal_rate: float, sample_time: float) -> None:
    # Keep copied ARX-RLS/Zone-MPC observers aligned with the executed action.
    total_rate = float(action.basal) + float(action.bolus)
    denom = max(float(getattr(controller, "_basal_u_per_min", basal_rate)), 1e-8)
    if hasattr(controller, "uhat_hist"):
        try:
            if len(controller.uhat_hist) > 0:
                controller.uhat_hist[-1] = float(total_rate / denom - 1.0)
        except Exception:
            pass

    if hasattr(controller, "_commit_workspace_zone_control"):
        try:
            basal_units = float(basal_rate) * float(sample_time)
            executed_basal_units = float(action.basal) * float(sample_time)
            controller._commit_workspace_zone_control(executed_basal_units - basal_units)
        except Exception:
            pass



@dataclass
class ArxMpcConfig:
    param_file: str | Path = DEFAULT_ARX_THETA
    target_glucose: float = 110.0
    pred_horizon_min: float = 120.0
    control_horizon_min: float = 45.0
    P0: float = 100.0
    lam_base: float = 0.999
    kappa: float = 0.002
    rls_err_clip: float = 20.0
    rls_err_deadzone: float = 20.0
    rls_phi_min: float = 0.001
    min_order: int = 6
    u_delay_steps: int = 3
    m_delay_steps: int = 1
    meal_filter_tau_min: float = 15.0
    meal_release_g_per_min: float = 5.0
    meal_c_max: float = 5.0
    meal_forecast_steps: int = 8
    enable_meal_bolus: bool = False
    meal_bolus_scale: float = 0.75
    workspace_mpc_variant: str = "adaptive"
    workspace_mpc_prediction_model: str = "arx"
    use_iob_constraint: bool = False
    max_insulin_units_per_sample: float = 1.0
    max_insulin_tdi_fraction: float | None = None
    model_gain_factor: float = 1.0
    r_plus_scale: float = 1.0
    enable_low_glucose_suspend: bool = False


def make_arx_rls_zone_mpc(config: ArxMpcConfig | None = None):
    cfg = config or ArxMpcConfig()
    param_file = Path(cfg.param_file)
    if not param_file.is_absolute():
        param_file = MPC_ASSETS_ROOT / param_file
    return ArxRlsVffMpcController(
        target=cfg.target_glucose,
        sample_time_min=5.0,
        pred_horizon_min=cfg.pred_horizon_min,
        control_horizon_min=cfg.control_horizon_min,
        P0=cfg.P0,
        lam_base=cfg.lam_base,
        kappa=cfg.kappa,
        rls_err_clip=cfg.rls_err_clip,
        rls_err_deadzone=cfg.rls_err_deadzone,
        rls_phi_min=cfg.rls_phi_min,
        min_order=cfg.min_order,
        u_delay_steps=cfg.u_delay_steps,
        m_delay_steps=cfg.m_delay_steps,
        use_meal_filter=True,
        meal_filter_tau_min=cfg.meal_filter_tau_min,
        use_meal_buffer=True,
        meal_release_g_per_min=cfg.meal_release_g_per_min,
        meal_c_max=cfg.meal_c_max,
        meal_forecast_steps=cfg.meal_forecast_steps,
        theta_input_mode="uhat",
        enable_meal_bolus=cfg.enable_meal_bolus,
        meal_bolus_scale=cfg.meal_bolus_scale,
        param_file=str(param_file),
        allow_fallback_theta=False,
        use_workspace_mpc=True,
        workspace_mpc_prediction_model=cfg.workspace_mpc_prediction_model,
        workspace_mpc_variant=cfg.workspace_mpc_variant,
        workspace_mpc_use_iob_constraint=cfg.use_iob_constraint,
        workspace_mpc_max_insulin_units_per_sample=cfg.max_insulin_units_per_sample,
        workspace_mpc_max_insulin_tdi_fraction=cfg.max_insulin_tdi_fraction,
        workspace_mpc_model_gain_factor=cfg.model_gain_factor,
        workspace_mpc_r_plus_scale=cfg.r_plus_scale,
        workspace_mpc_enable_low_glucose_suspend=cfg.enable_low_glucose_suspend,
    )


def patient_profile(patient_name: str) -> tuple[float, float, float]:
    patient = T1DPatient.withName(patient_name)
    basal = patient_basal_rate(patient)
    body_weight = float(patient._params["BW"])
    return basal, body_weight, 5.0


class PpoBasalController(Controller):
    def __init__(
        self,
        agent: PPOAgent,
        max_basal_multiplier: float = 5.0,
        deterministic: bool = True,
        bolus_source: Controller | None = None,
    ):
        super().__init__(init_state=None)
        self.agent = agent
        self.max_basal_multiplier = float(max_basal_multiplier)
        self.deterministic = bool(deterministic)
        self.bolus_source = bolus_source
        self.tracker: GlucoseStateTracker | None = None
        self.patient_name: str | None = None
        self.basal_rate: float = 0.01
        self.max_basal: float = 10.0
        self.step_index = 0
        self.history: list[dict] = []

    def reset(self):
        self.tracker = None
        self.patient_name = None
        self.step_index = 0
        self.history = []
        if self.bolus_source is not None:
            self.bolus_source.reset()

    def _ensure_tracker(self, patient_name: str, glucose: float, sample_time: float) -> None:
        if self.tracker is not None and self.patient_name == patient_name:
            return
        basal, body_weight, _sample = patient_profile(patient_name)
        self.patient_name = patient_name
        self.basal_rate = basal
        self.tracker = GlucoseStateTracker(body_weight, basal, sample_time)
        self.tracker.reset(glucose)

    def _state_from_observation(self, observation, kwargs) -> np.ndarray:
        patient_name = kwargs.get("patient_name") or self.patient_name or "adult#001"
        sample_time = float(kwargs.get("sample_time", 5.0))
        glucose = require_finite_cgm(observation)
        self._ensure_tracker(patient_name, glucose, sample_time)
        assert self.tracker is not None
        # PPO receives only an explicit announcement.  The simulator's raw
        # ``meal`` diagnostic contains consumed CHO and is not causal in the
        # unannounced protocol.
        meal_grams = float(kwargs.get("announced_meal", 0.0) or 0.0)
        return self.tracker.make_state(
            glucose,
            meal_grams=meal_grams,
            bolus_rate=float(kwargs.get("bolus_rate", 0.0) or 0.0),
            now=kwargs.get("time"),
        )

    def policy(self, observation, reward, done, **kwargs):
        state = self._state_from_observation(observation, kwargs)
        raw_action, _log_prob, value = self.agent.act(state, deterministic=self.deterministic)
        basal = float(
            map_raw_action_to_basal(
                raw_action,
                self.basal_rate,
                self.max_basal,
                self.max_basal_multiplier,
            )[0]
        )
        bolus = 0.0
        if self.bolus_source is not None:
            bolus_action = self.bolus_source.policy(observation, reward, done, **kwargs)
            bolus = float(getattr(bolus_action, "bolus", 0.0))
            commit_executed_action_to_mpc(
                self.bolus_source,
                Action(basal=basal, bolus=bolus),
                self.basal_rate,
                float(kwargs.get("sample_time", 5.0)),
            )
        if self.tracker is not None:
            self.tracker.record_action(basal, bolus)
        glucose = require_finite_cgm(observation)
        self.history.append(
            {
                "step": self.step_index,
                "selected": "ppo",
                "glucose": glucose,
                "ppo_raw_action": float(raw_action[0]),
                "ppo_value": float(value),
                "ppo_basal": basal,
                "selected_basal": basal,
                "selected_bolus": bolus,
            }
        )
        self.step_index += 1
        return Action(basal=basal, bolus=bolus)


@dataclass
class SwitchConfig:
    warmup_steps: int = 12
    trend_horizon_steps: int = 6  # deprecated compatibility field
    max_action_gap_basal: float = 1.25
    max_basal_multiplier: float = 5.0
    deterministic_ppo: bool = True
    # The legacy ppo_min_* fields are retained for checkpoint-era callers but
    # no longer authorize MPC or influence the causal compatibility protocol.
    ppo_min_glucose: float = 0.0
    ppo_min_projected_glucose: float = 0.0
    ppo_min_rate: float = -999.0
    risk_low_glucose: float = 90.0
    risk_current_glucose: float = 130.0
    risk_projected_glucose: float = 130.0
    risk_fast_fall_rate: float = -2.0
    risk_prediction_horizons_min: tuple[float, ...] = (30.0, 60.0, 90.0)
    risk_strategy: str = "lower_insulin"
    high_risk_arbitration: bool = False
    high_risk_glucose: float = 180.0
    high_risk_projected_glucose: float = 200.0
    high_risk_rising_rate: float = 0.1
    high_risk_strategy: str = "higher_insulin"
    high_risk_iob_taper_start_ratio: float = 1.0e9
    high_risk_iob_max_ratio: float = 1.0e9
    high_risk_iob_prediction_enabled: bool = False
    high_risk_iob_prediction_horizon_min: float = 90.0
    high_risk_iob_prediction_effect_fraction: float = 0.5
    high_risk_iob_prediction_guard_glucose: float = 130.0
    high_risk_correction_enabled: bool = False
    high_risk_correction_target_glucose: float = 110.0
    high_risk_correction_horizon_min: float = 60.0
    high_risk_correction_max_basal_multiplier: float = 8.0
    safety_governor: bool = True
    safety_prediction_horizons_min: tuple[float, ...] = (30.0, 60.0, 90.0)
    safety_suspend_glucose: float = 75.0
    safety_low_glucose: float = 90.0
    safety_projected_low_glucose: float = 90.0
    safety_falling_glucose: float = 130.0
    safety_falling_rate: float = -0.25
    safety_low_basal_fraction: float = 0.0
    safety_falling_basal_fraction: float = 0.5
    safety_iob_excess_ratio: float = 1.25
    safety_iob_guard_projected_glucose: float = 130.0
    safety_iob_guard_basal_fraction: float = 0.0
    compatibility_gate_enabled: bool = False
    compatibility_gate_config: CompatibilityGateConfig | None = None
    compatibility_calibration_id: str = ""


class PpoMpcSwitchController(Controller):
    def __init__(
        self,
        agent: PPOAgent,
        mpc_controller: Controller,
        config: SwitchConfig | None = None,
    ):
        super().__init__(init_state=None)
        self.agent = agent
        self.mpc = mpc_controller
        self.config = config or SwitchConfig()
        self.tracker: GlucoseStateTracker | None = None
        self.patient_name: str | None = None
        self.basal_rate: float = 0.01
        self.correction_factor: float | None = None
        self.max_basal: float = 10.0
        self.last_glucose: float | None = None
        self.glucose_history: deque[float] = deque(maxlen=4)
        self.step_index = 0
        self.history: list[dict] = []
        if (
            self.config.compatibility_gate_enabled
            and self.config.compatibility_gate_config is None
        ):
            raise ValueError(
                "compatibility_gate_enabled requires a frozen "
                "compatibility_gate_config"
            )
        self.compatibility_gate = (
            CausalCompatibilityGate(self.config.compatibility_gate_config)
            if self.config.compatibility_gate_enabled
            else None
        )
        self.last_compatibility: CompatibilityDecision | None = None

    def reset(self):
        self.mpc.reset()
        self.tracker = None
        self.patient_name = None
        self.correction_factor = None
        self.last_glucose = None
        self.glucose_history.clear()
        self.step_index = 0
        self.history = []
        if self.compatibility_gate is not None:
            self.compatibility_gate.reset()
        self.last_compatibility = None

    def _ensure_tracker(self, patient_name: str, glucose: float, sample_time: float) -> None:
        if self.tracker is not None and self.patient_name == patient_name:
            return
        basal, body_weight, _sample = patient_profile(patient_name)
        self.patient_name = patient_name
        self.basal_rate = basal
        self.correction_factor = patient_correction_factor(patient_name)
        self.tracker = GlucoseStateTracker(body_weight, basal, sample_time)
        self.tracker.reset(glucose)

    def _ppo_candidate(self, observation, kwargs) -> tuple[Action, np.ndarray, float, float]:
        patient_name = kwargs.get("patient_name") or self.patient_name or "adult#001"
        sample_time = float(kwargs.get("sample_time", 5.0))
        glucose = require_finite_cgm(observation)
        self._ensure_tracker(patient_name, glucose, sample_time)
        assert self.tracker is not None
        meal_grams = float(kwargs.get("announced_meal", 0.0) or 0.0)
        state = self.tracker.make_state(glucose, meal_grams=meal_grams, now=kwargs.get("time"))
        raw_action, _log_prob, value = self.agent.act(
            state, deterministic=self.config.deterministic_ppo
        )
        basal = float(
            map_raw_action_to_basal(
                raw_action,
                self.basal_rate,
                self.max_basal,
                self.config.max_basal_multiplier,
            )[0]
        )
        return Action(basal=basal, bolus=0.0), raw_action, float(value), state

    @staticmethod
    def _candidate_total_rate(action: Action) -> float:
        return float(action.basal) + float(action.bolus)

    def _risk_projection_values(self, glucose: float, rate: float) -> dict[float, float]:
        horizons = tuple(
            sorted(
                {
                    float(value)
                    for value in self.config.risk_prediction_horizons_min
                    if np.isfinite(value) and float(value) > 0.0
                }
            )
        ) or (30.0,)
        return {horizon: float(glucose + rate * horizon) for horizon in horizons}

    def _high_risk_reason(self, glucose: float, rate: float) -> str:
        cfg = self.config
        if not cfg.high_risk_arbitration:
            return ""
        projected_max = max(self._risk_projection_values(glucose, rate).values())
        if glucose >= cfg.high_risk_glucose and rate >= cfg.high_risk_rising_rate:
            return "current_high_rising"
        if projected_max >= cfg.high_risk_projected_glucose and rate > 0.0:
            return "projected_high_rising"
        return ""

    def _high_correction_candidate(
        self,
        *,
        glucose: float,
        action: Action,
    ) -> tuple[Action, dict[str, float]]:
        cfg = self.config
        correction_factor = self.correction_factor
        tracker = self.tracker
        if (
            not cfg.high_risk_correction_enabled
            or correction_factor is None
            or not math.isfinite(float(correction_factor))
            or float(correction_factor) <= 0.0
            or tracker is None
        ):
            return action, {
                "need_units": 0.0,
                "iob_credit_units": 0.0,
                "residual_units": 0.0,
                "floor_basal": float(action.basal),
            }
        need_units = max(
            0.0,
            (float(glucose) - float(cfg.high_risk_correction_target_glucose))
            / float(correction_factor),
        )
        current_iob = float(tracker.current_iob())
        basal_iob = float(tracker.steady_state_iob())
        iob_credit_units = max(0.0, current_iob - basal_iob)
        residual_units = max(0.0, need_units - iob_credit_units)
        correction_rate = residual_units / max(
            float(cfg.high_risk_correction_horizon_min), 1e-8
        )
        floor_basal = min(
            float(cfg.high_risk_correction_max_basal_multiplier) * self.basal_rate,
            self.basal_rate + correction_rate,
        )
        corrected = Action(
            basal=max(float(action.basal), max(0.0, floor_basal)),
            bolus=float(action.bolus),
        )
        return corrected, {
            "need_units": need_units,
            "iob_credit_units": iob_credit_units,
            "residual_units": residual_units,
            "floor_basal": floor_basal,
        }

    def _high_iob_taper_candidate(
        self, ppo_action: Action, mpc_action: Action
    ) -> tuple[Action, float, float]:
        tracker = getattr(self, "tracker", None)
        if tracker is None:
            return mpc_action, 1.0, float("nan")
        basal_iob = float(tracker.steady_state_iob())
        current_iob = float(tracker.current_iob())
        if basal_iob <= 0.0:
            return mpc_action, 1.0, float("nan")
        ratio = current_iob / basal_iob
        start = float(self.config.high_risk_iob_taper_start_ratio)
        stop = float(self.config.high_risk_iob_max_ratio)
        if stop <= start or ratio <= start:
            alpha = 1.0
        elif ratio >= stop:
            alpha = 0.0
        else:
            alpha = (stop - ratio) / (stop - start)
        blended = Action(
            basal=float(ppo_action.basal)
            + alpha * (float(mpc_action.basal) - float(ppo_action.basal)),
            bolus=float(ppo_action.bolus)
            + alpha * (float(mpc_action.bolus) - float(ppo_action.bolus)),
        )
        return blended, alpha, ratio

    def _high_iob_effect_projected_glucose(
        self, glucose: float, rate: float
    ) -> float:
        tracker = getattr(self, "tracker", None)
        correction_factor = getattr(self, "correction_factor", None)
        if (
            tracker is None
            or correction_factor is None
            or not math.isfinite(float(correction_factor))
            or float(correction_factor) <= 0.0
        ):
            return float("nan")
        excess_iob = max(
            0.0,
            float(tracker.current_iob()) - float(tracker.steady_state_iob()),
        )
        return (
            float(glucose)
            + float(rate) * float(self.config.high_risk_iob_prediction_horizon_min)
            - float(correction_factor)
            * excess_iob
            * float(self.config.high_risk_iob_prediction_effect_fraction)
        )

    def _latest_mpc_diagnostics(self) -> dict:
        getter = getattr(self.mpc, "get_latest_model_diagnostics", None)
        if getter is None:
            return {
                "available": False,
                "err_raw": float("nan"),
                "bg_pred": float("nan"),
                "bg_meas": float("nan"),
                "mpc_status": "",
                "problem_status": "",
                "solver_error": "",
                "sample_count": 0,
            }
        try:
            return dict(getter())
        except Exception:
            return {
                "available": False,
                "err_raw": float("nan"),
                "bg_pred": float("nan"),
                "bg_meas": float("nan"),
                "mpc_status": "",
                "problem_status": "",
                "solver_error": "",
                "sample_count": 0,
            }

    @staticmethod
    def _mpc_candidate_valid(action: Action, diagnostics: dict) -> bool:
        """Validate the action/solver without making no-gate depend on residual warmup."""
        if not np.isfinite(float(action.basal)) or not np.isfinite(float(action.bolus)):
            return False
        problem_status = str(diagnostics.get("problem_status", "")).strip().lower()
        if problem_status and "optimal" not in problem_status:
            return False
        mpc_status = str(diagnostics.get("mpc_status", "")).strip().lower()
        if any(token in mpc_status for token in ("error", "failed", "infeasible")):
            return False
        return True

    def _update_compatibility(
        self, diagnostics: dict, *, sample_time: float, candidate_valid: bool
    ) -> CompatibilityDecision | None:
        if self.compatibility_gate is None:
            self.last_compatibility = None
            return None
        decision = self.compatibility_gate.update(
            float(diagnostics.get("err_raw", float("nan"))),
            sample_time_min=sample_time,
            candidate_valid=candidate_valid,
        )
        self.last_compatibility = decision
        return decision

    def _decide(
        self,
        glucose: float,
        rate: float,
        ppo_action: Action,
        mpc_action: Action,
        model_authorized: bool = True,
    ) -> tuple[str, str, bool]:
        """Risk means conservative arbitration, never automatic MPC authority."""
        cfg = self.config
        projections = self._risk_projection_values(glucose, rate)
        projected_min = min(projections.values())
        risk_reason = ""
        if self.step_index < cfg.warmup_steps:
            risk_reason = "warmup_uncertainty"
        elif glucose <= cfg.risk_low_glucose:
            risk_reason = "current_low"
        elif rate <= cfg.risk_fast_fall_rate:
            risk_reason = "fast_fall"
        elif glucose <= cfg.risk_current_glucose and rate < 0.0:
            risk_reason = "current_falling"
        elif projected_min <= cfg.risk_projected_glucose and rate < 0.0:
            risk_reason = "multihorizon_falling"

        if not risk_reason:
            high_reason = self._high_risk_reason(glucose, rate)
            if not high_reason:
                return "ppo", "nominal_ppo", False
            if cfg.high_risk_strategy != "higher_insulin":
                raise ValueError(
                    f"unsupported switch high-risk strategy: {cfg.high_risk_strategy}"
                )
            if cfg.high_risk_iob_prediction_enabled:
                projected = self._high_iob_effect_projected_glucose(glucose, rate)
                if (
                    math.isfinite(projected)
                    and projected <= cfg.high_risk_iob_prediction_guard_glucose
                ):
                    ppo_total = self._candidate_total_rate(ppo_action)
                    mpc_total = self._candidate_total_rate(mpc_action)
                    selected = "ppo" if ppo_total <= mpc_total else "mpc"
                    return (
                        selected,
                        f"high_risk_{high_reason}_{selected}_iob_projection_guard",
                        True,
                    )
            tracker = getattr(self, "tracker", None)
            if tracker is not None:
                basal_iob = float(tracker.steady_state_iob())
                current_iob = float(tracker.current_iob())
                if (
                    basal_iob > 0.0
                    and current_iob / basal_iob
                    >= float(cfg.high_risk_iob_max_ratio)
                ):
                    return (
                        "ppo",
                        f"high_risk_{high_reason}_ppo_iob_limit",
                        True,
                    )
            if not bool(model_authorized):
                return (
                    "ppo",
                    f"high_risk_{high_reason}_mpc_blocked_incompatible",
                    True,
                )
            ppo_total = self._candidate_total_rate(ppo_action)
            mpc_total = self._candidate_total_rate(mpc_action)
            selected = "mpc" if mpc_total > ppo_total else "ppo"
            return (
                selected,
                f"high_risk_{high_reason}_{selected}_higher_insulin",
                True,
            )
        if cfg.risk_strategy == "ppo":
            return "ppo", f"risk_{risk_reason}_ppo_only", True
        if cfg.risk_strategy != "lower_insulin":
            raise ValueError(f"unsupported switch risk strategy: {cfg.risk_strategy}")
        if not bool(model_authorized):
            return "ppo", f"risk_{risk_reason}_mpc_blocked_incompatible", True
        ppo_total = self._candidate_total_rate(ppo_action)
        mpc_total = self._candidate_total_rate(mpc_action)
        selected = "ppo" if ppo_total <= mpc_total else "mpc"
        return selected, f"risk_{risk_reason}_{selected}_lower_insulin", True

    def _apply_safety_governor(
        self, glucose: float, rate: float, action: Action, sample_time: float
    ) -> tuple[Action, bool, str, float]:
        cfg = self.config
        current_iob = self.tracker.current_iob() if self.tracker is not None else None
        basal_iob = self.tracker.steady_state_iob() if self.tracker is not None else None
        decision = apply_safety_governor(
            glucose=glucose,
            glucose_rate=rate,
            basal=float(action.basal),
            bolus=float(action.bolus),
            basal_reference=self.basal_rate,
            config=SafetyGovernorConfig(
                enabled=cfg.safety_governor,
                prediction_horizons_min=cfg.safety_prediction_horizons_min,
                suspend_glucose=cfg.safety_suspend_glucose,
                low_glucose=cfg.safety_low_glucose,
                projected_low_glucose=cfg.safety_projected_low_glucose,
                falling_glucose=cfg.safety_falling_glucose,
                falling_rate=cfg.safety_falling_rate,
                low_basal_fraction=cfg.safety_low_basal_fraction,
                falling_basal_fraction=cfg.safety_falling_basal_fraction,
                iob_excess_ratio=cfg.safety_iob_excess_ratio,
                iob_guard_projected_glucose=cfg.safety_iob_guard_projected_glucose,
                iob_guard_basal_fraction=cfg.safety_iob_guard_basal_fraction,
            ),
            sample_time_min=sample_time,
            iob_units=current_iob,
            basal_iob_units=basal_iob,
        )
        cap = float("nan") if decision.cap_basal is None else decision.cap_basal
        return (
            Action(basal=decision.basal, bolus=decision.bolus),
            decision.overridden,
            decision.reason,
            cap,
        )

    def policy(self, observation, reward, done, **kwargs):
        glucose = require_finite_cgm(observation)
        sample_time = float(kwargs.get("sample_time", 5.0))
        rate = update_causal_glucose_rate(self.glucose_history, glucose, sample_time)

        ppo_action, raw_action, ppo_value, _state = self._ppo_candidate(observation, kwargs)
        causal_iob_pre = self.tracker.current_iob() if self.tracker is not None else np.nan
        basal_iob = self.tracker.steady_state_iob() if self.tracker is not None else np.nan
        mpc_raw_action = self.mpc.policy(observation, reward, done, **kwargs)
        mpc_action = mpc_raw_action
        high_correction = {
            "need_units": 0.0,
            "iob_credit_units": 0.0,
            "residual_units": 0.0,
            "floor_basal": float(mpc_raw_action.basal),
        }
        high_reason = self._high_risk_reason(glucose, rate)
        high_iob_effect_projection = self._high_iob_effect_projected_glucose(
            glucose, rate
        )
        iob_taper_alpha = 1.0
        iob_taper_ratio = float("nan")
        if high_reason:
            mpc_action, high_correction = self._high_correction_candidate(
                glucose=glucose,
                action=mpc_raw_action,
            )
            mpc_action, iob_taper_alpha, iob_taper_ratio = (
                self._high_iob_taper_candidate(ppo_action, mpc_action)
            )
        mpc_diagnostics = self._latest_mpc_diagnostics()
        mpc_candidate_valid = self._mpc_candidate_valid(mpc_action, mpc_diagnostics)
        compatibility = self._update_compatibility(
            mpc_diagnostics,
            sample_time=sample_time,
            candidate_valid=mpc_candidate_valid,
        )
        model_authorized = (
            mpc_candidate_valid
            if not self.config.compatibility_gate_enabled
            else bool(compatibility is not None and compatibility.authorized)
        )
        selected, reason, risk_active = self._decide(
            glucose, rate, ppo_action, mpc_action, model_authorized=model_authorized
        )
        if risk_active and selected == "ppo":
            action = Action(basal=float(ppo_action.basal), bolus=float(ppo_action.bolus))
        elif risk_active and selected == "mpc":
            action = Action(basal=float(mpc_action.basal), bolus=float(mpc_action.bolus))
        elif selected == "ppo":
            action = Action(basal=float(ppo_action.basal), bolus=float(getattr(mpc_action, "bolus", 0.0)))
        else:
            action = Action(basal=float(mpc_action.basal), bolus=float(mpc_action.bolus))
        pre_safety_basal = float(action.basal)
        pre_safety_bolus = float(action.bolus)
        action, safety_override, safety_reason, safety_cap_basal = self._apply_safety_governor(
            glucose, rate, action, sample_time
        )
        commit_executed_action_to_mpc(self.mpc, action, self.basal_rate, sample_time)

        if self.tracker is not None:
            self.tracker.record_action(action.basal, action.bolus)
        projections = self._risk_projection_values(glucose, rate)
        projection_at = lambda horizon: projections[
            min(projections, key=lambda value: abs(value - horizon))
        ]
        self.history.append(
            {
                "step": self.step_index,
                "selected": selected,
                "reason": reason,
                "risk_active": bool(risk_active),
                "risk_reason": reason if risk_active else "",
                "glucose": glucose,
                "glucose_rate": rate,
                "ppo_raw_action": float(raw_action[0]),
                "ppo_value": ppo_value,
                "ppo_basal": float(ppo_action.basal),
                "mpc_basal": float(mpc_action.basal),
                "mpc_bolus": float(mpc_action.bolus),
                "mpc_raw_basal": float(mpc_raw_action.basal),
                "mpc_raw_bolus": float(mpc_raw_action.bolus),
                "high_correction_need_units": high_correction["need_units"],
                "high_correction_iob_credit_units": high_correction[
                    "iob_credit_units"
                ],
                "high_correction_residual_units": high_correction[
                    "residual_units"
                ],
                "high_correction_floor_basal": high_correction["floor_basal"],
                "high_iob_taper_alpha": iob_taper_alpha,
                "high_iob_taper_ratio": iob_taper_ratio,
                "high_iob_effect_projected_glucose": high_iob_effect_projection,
                "selector_total_ppo": self._candidate_total_rate(ppo_action),
                "selector_total_mpc": self._candidate_total_rate(mpc_action),
                "risk_projected_30min": projection_at(30.0),
                "risk_projected_60min": projection_at(60.0),
                "risk_projected_90min": projection_at(90.0),
                "risk_projected_min": min(projections.values()),
                "causal_iob_units": causal_iob_pre,
                "basal_iob_units": basal_iob,
                "iob_ratio": causal_iob_pre / max(basal_iob, 1e-8),
                "cob_available": False,
                "announced_cob_grams": 0.0,
                "pre_safety_basal": pre_safety_basal,
                "pre_safety_bolus": pre_safety_bolus,
                "safety_override": bool(safety_override),
                "safety_reason": safety_reason,
                "safety_cap_basal": safety_cap_basal,
                "selected_basal": float(action.basal),
                "selected_bolus": float(action.bolus),
                "action_gap_basal": abs(float(ppo_action.basal) - float(mpc_action.basal))
                / max(self.basal_rate, 1e-8),
                "ppo_raw_equiv_mpc": float(
                    basal_to_raw_action(
                        float(mpc_action.basal),
                        self.basal_rate,
                        self.config.max_basal_multiplier,
                    )[0]
                ),
                "model_residual_raw_mgdl": float(
                    mpc_diagnostics.get("err_raw", float("nan"))
                ),
                "model_bg_pred_mgdl": float(
                    mpc_diagnostics.get("bg_pred", float("nan"))
                ),
                "model_bg_meas_mgdl": float(
                    mpc_diagnostics.get("bg_meas", float("nan"))
                ),
                "model_diagnostic_sample_count": int(
                    mpc_diagnostics.get("sample_count", 0)
                ),
                "model_residual_abs_mgdl": float(
                    compatibility.residual_abs_mgdl
                    if compatibility is not None
                    else abs(float(mpc_diagnostics.get("err_raw", float("nan"))))
                ),
                "model_residual_scale_mgdl": float(
                    compatibility.residual_scale_mgdl
                    if compatibility is not None
                    else float("nan")
                ),
                "model_npe": float(
                    compatibility.normalized_prediction_error
                    if compatibility is not None
                    else float("nan")
                ),
                "model_npe_ewma": float(
                    compatibility.normalized_prediction_error_ewma
                    if compatibility is not None
                    else float("nan")
                ),
                "model_compatibility_score": float(
                    compatibility.compatibility_score
                    if compatibility is not None
                    else float("nan")
                ),
                "model_compatible": bool(
                    compatibility.compatible if compatibility is not None else False
                ),
                "compat_gate_transition": str(
                    compatibility.transition if compatibility is not None else ""
                ),
                "compat_reason": str(
                    compatibility.reason
                    if compatibility is not None
                    else "gate_disabled"
                ),
                "compat_valid_samples": int(
                    compatibility.valid_samples if compatibility is not None else 0
                ),
                "compat_good_streak": int(
                    compatibility.good_streak if compatibility is not None else 0
                ),
                "compat_bad_streak": int(
                    compatibility.bad_streak if compatibility is not None else 0
                ),
                "compat_off_steps": int(
                    compatibility.off_steps if compatibility is not None else 0
                ),
                "mpc_candidate_valid": bool(mpc_candidate_valid),
                "mpc_model_authorized": bool(model_authorized),
                "mpc_status": str(mpc_diagnostics.get("mpc_status", "")),
                "mpc_problem_status": str(
                    mpc_diagnostics.get("problem_status", "")
                ),
                "compatibility_gate_enabled": bool(
                    self.config.compatibility_gate_enabled
                ),
                "compatibility_calibration_id": str(
                    self.config.compatibility_calibration_id
                ),
            }
        )
        self.last_glucose = glucose
        self.step_index += 1
        return action

    def switch_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)


def synchronize_executed_action(
    controller,
    action: Action,
    sample_time: float,
    basal_reference: float | None = None,
) -> None:
    """Align controller-side histories with the action sent to the simulator."""
    basal_rate = float(
        basal_reference
        if basal_reference is not None
        else getattr(controller, "basal_rate", 0.01)
    )
    tracker = getattr(controller, "tracker", None)
    if tracker is not None:
        tracker.last_basal_rate = float(action.basal)
        if len(tracker.insulin_history) > 0:
            tracker.insulin_history[-1] = float(action.basal) + float(action.bolus)

    if isinstance(controller, PpoMpcSwitchController):
        commit_executed_action_to_mpc(controller.mpc, action, basal_rate, sample_time)
    elif isinstance(controller, PpoBasalController) and controller.bolus_source is not None:
        commit_executed_action_to_mpc(
            controller.bolus_source, action, basal_rate, sample_time
        )
    else:
        commit_executed_action_to_mpc(controller, action, basal_rate, sample_time)

    history = getattr(controller, "history", None)
    if history:
        history[-1]["selected_basal"] = float(action.basal)
        history[-1]["selected_bolus"] = float(action.bolus)


class SharedSafetyGovernorController(Controller):
    """Wrap PPO, MPC, or Switch with the exact same post-policy shield."""

    def __init__(
        self,
        controller: Controller,
        config: SafetyGovernorConfig,
        controller_family: str,
    ):
        super().__init__(init_state=None)
        self.controller = controller
        self.config = config
        self.controller_family = str(controller_family)
        self.last_glucose: float | None = None
        self.basal_rate = 0.01
        self.patient_name: str | None = None
        self.correction_factor: float | None = None
        self.tracker: GlucoseStateTracker | None = None
        self.glucose_history: deque[float] = deque(maxlen=4)
        self.step_index = 0
        self.history: list[dict] = []

    def reset(self):
        self.controller.reset()
        self.last_glucose = None
        self.patient_name = None
        self.correction_factor = None
        self.tracker = None
        self.glucose_history.clear()
        self.step_index = 0
        self.history = []

    def _ensure_profile(
        self, patient_name: str, glucose: float, sample_time: float
    ) -> None:
        if (
            self.patient_name == patient_name
            and self.tracker is not None
            and abs(self.tracker.sample_time - sample_time) <= 1e-9
        ):
            return
        basal, body_weight, _sample_time = patient_profile(patient_name)
        self.patient_name = patient_name
        self.basal_rate = basal
        self.correction_factor = patient_correction_factor(patient_name)
        self.tracker = GlucoseStateTracker(body_weight, basal, sample_time)
        self.tracker.reset(glucose)
        self.glucose_history.clear()

    def policy(self, observation, reward, done, **kwargs):
        glucose = require_finite_cgm(observation)
        sample_time = float(kwargs.get("sample_time", 5.0))
        patient_name = kwargs.get("patient_name") or self.patient_name or "adult#001"
        self._ensure_profile(patient_name, glucose, sample_time)
        rate = update_causal_glucose_rate(self.glucose_history, glucose, sample_time)
        assert self.tracker is not None
        causal_iob = self.tracker.current_iob()
        basal_iob = self.tracker.steady_state_iob()

        proposed = self.controller.policy(observation, reward, done, **kwargs)
        decision = apply_safety_governor(
            glucose=glucose,
            glucose_rate=rate,
            basal=float(proposed.basal),
            bolus=float(proposed.bolus),
            basal_reference=self.basal_rate,
            config=self.config,
            sample_time_min=sample_time,
            iob_units=causal_iob,
            basal_iob_units=basal_iob,
            correction_factor_mgdl_per_u=self.correction_factor,
        )
        executed = Action(basal=decision.basal, bolus=decision.bolus)
        synchronize_executed_action(
            self.controller,
            executed,
            sample_time,
            basal_reference=self.basal_rate,
        )
        self.tracker.record_action(executed.basal, executed.bolus)

        base_history = getattr(self.controller, "history", None)
        row = dict(base_history[-1]) if base_history else {}
        selected = str(row.get("selected", self.controller_family))
        row.update(
            {
                "step": self.step_index,
                "controller_family": self.controller_family,
                "selected": selected,
                "glucose": glucose,
                "glucose_rate": rate,
                "pre_safety_basal": float(proposed.basal),
                "pre_safety_bolus": float(proposed.bolus),
                "safety_triggered": (
                    decision.cap_basal is not None or decision.floor_basal is not None
                ),
                "safety_override": decision.overridden,
                "safety_reason": decision.reason,
                "safety_cap_basal": decision.cap_basal,
                "safety_floor_basal": decision.floor_basal,
                "safety_projected_glucose": decision.projected_glucose,
                "safety_projected_30min": decision.projected_glucose_30min,
                "safety_projected_60min": decision.projected_glucose_60min,
                "safety_projected_90min": decision.projected_glucose_90min,
                "safety_projected_min": decision.projected_glucose_min,
                "causal_iob_units": causal_iob,
                "basal_iob_units": basal_iob,
                "iob_ratio": decision.iob_ratio,
                "correction_factor_mgdl_per_u": self.correction_factor,
                "correction_need_units": decision.correction_need_units,
                "correction_iob_credit_units": decision.correction_iob_credit_units,
                "correction_residual_units": decision.correction_residual_units,
                "cob_available": False,
                "announced_cob_grams": 0.0,
                "selected_basal": float(executed.basal),
                "selected_bolus": float(executed.bolus),
                "ppo_authorized": selected == "ppo",
                "ppo_effective": selected == "ppo" and not decision.overridden,
                "model_residual_raw_mgdl": float(
                    row.get("model_residual_raw_mgdl", float("nan"))
                ),
                "model_bg_pred_mgdl": float(
                    row.get("model_bg_pred_mgdl", float("nan"))
                ),
                "model_bg_meas_mgdl": float(
                    row.get("model_bg_meas_mgdl", float("nan"))
                ),
                "model_diagnostic_sample_count": int(
                    row.get("model_diagnostic_sample_count", 0)
                ),
                "model_residual_abs_mgdl": float(
                    row.get("model_residual_abs_mgdl", float("nan"))
                ),
                "model_residual_scale_mgdl": float(
                    row.get("model_residual_scale_mgdl", float("nan"))
                ),
                "model_npe": float(row.get("model_npe", float("nan"))),
                "model_npe_ewma": float(row.get("model_npe_ewma", float("nan"))),
                "model_compatibility_score": float(
                    row.get("model_compatibility_score", float("nan"))
                ),
                "model_compatible": bool(row.get("model_compatible", False)),
                "compat_gate_transition": str(row.get("compat_gate_transition", "")),
                "compat_reason": str(row.get("compat_reason", "")),
                "compat_valid_samples": int(row.get("compat_valid_samples", 0)),
                "compat_good_streak": int(row.get("compat_good_streak", 0)),
                "compat_bad_streak": int(row.get("compat_bad_streak", 0)),
                "compat_off_steps": int(row.get("compat_off_steps", 0)),
                "mpc_candidate_valid": bool(row.get("mpc_candidate_valid", False)),
                "mpc_model_authorized": bool(row.get("mpc_model_authorized", False)),
                "mpc_status": str(row.get("mpc_status", "")),
                "mpc_problem_status": str(row.get("mpc_problem_status", "")),
                "compatibility_gate_enabled": bool(
                    row.get("compatibility_gate_enabled", False)
                ),
                "compatibility_calibration_id": str(
                    row.get("compatibility_calibration_id", "")
                ),
            }
        )
        self.history.append(row)
        self.last_glucose = glucose
        self.step_index += 1
        return executed

    def switch_history(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)
