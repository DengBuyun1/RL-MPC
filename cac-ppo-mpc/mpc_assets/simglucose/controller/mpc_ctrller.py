from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import pkg_resources
from scipy.linalg import solve_discrete_are
from scipy.optimize import minimize

from .base import Action, Controller

logger = logging.getLogger(__name__)

CONTROL_QUEST = pkg_resources.resource_filename("simglucose", "params/Quest.csv")
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    "simglucose", "params/vpatient_params.csv"
)


@dataclass(frozen=True)
class PatientControlProfile:
    basal_rate: float  # U/min
    tdi: float  # U/day
    cf: float  # mg/dL/U
    cr: float  # g/U


class ZoneMPCController(Controller):
    """Paper-informed zone MPC controller for the local simglucose env.

    This implementation follows the controller structure in:
    - van Heusden et al., TBME 2012: control-relevant model personalized by TDI.
    - Gondhalekar et al., Automatica 2018: zone MPC with velocity weighting
      and velocity penalty.
    - Shi et al., TBME 2019: adaptive insulin penalty based on predicted
      glucose and velocity.

    It is intentionally self-contained and uses scipy's SLSQP solver because
    this workspace does not include a dedicated QP solver. The exact clinical
    controller also uses details from prior papers/supplements. The observer is
    implemented from Gondhalekar 2016; pump carry-over discretization and the
    full IOB-history constraint are still approximated here so the controller
    can run in the provided simglucose environment.
    """

    VALID_VARIANTS = {"previous", "velocity", "adaptive"}

    def __init__(
        self,
        variant: str = "adaptive",
        prediction_horizon: int = 9,
        control_horizon: int = 5,
        target_glucose: float = 110.0,
        model_sample_time: float = 5.0,
        announce_meals: bool = False,
        use_iob_constraint: bool = False,
        max_insulin_units_per_sample: float = 1.0,
        max_insulin_tdi_fraction: Optional[float] = None,
        model_gain_factor: float = 1.0,
        r_plus_scale: float = 1.0,
        meal_bolus_scale: float = 1.0,
        enable_low_glucose_suspend: bool = False,
        suspend_glucose: float = 80.0,
        predictive_suspend_glucose: float = 100.0,
        predictive_suspend_velocity: float = 0.0,
        observer_gain: Optional[Tuple[float, float, float]] = None,
        max_slsqp_iter: int = 80,
    ):
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"variant must be one of {sorted(self.VALID_VARIANTS)}")
        if control_horizon > prediction_horizon:
            raise ValueError("control_horizon cannot exceed prediction_horizon")

        self.variant = variant
        self.Ny = int(prediction_horizon)
        self.Nu = int(control_horizon)
        self.ys = float(target_glucose)
        self.model_sample_time = float(model_sample_time)
        self.announce_meals = bool(announce_meals)
        self.use_iob_constraint = bool(use_iob_constraint)
        self.max_insulin_units_per_sample = float(max_insulin_units_per_sample)
        self.max_insulin_tdi_fraction = max_insulin_tdi_fraction
        self.model_gain_factor = float(model_gain_factor)
        self.r_plus_scale = float(r_plus_scale)
        self.meal_bolus_scale = float(meal_bolus_scale)
        self.enable_low_glucose_suspend = bool(enable_low_glucose_suspend)
        self.suspend_glucose = float(suspend_glucose)
        self.predictive_suspend_glucose = float(predictive_suspend_glucose)
        self.predictive_suspend_velocity = float(predictive_suspend_velocity)
        self._observer_gain_arg = observer_gain
        self.max_slsqp_iter = int(max_slsqp_iter)

        self.quest = pd.read_csv(CONTROL_QUEST)
        self.patient_params = pd.read_csv(PATIENT_PARA_FILE)

        self.profile: Optional[PatientControlProfile] = None
        self.patient_name: Optional[str] = None
        self.xhat: Optional[np.ndarray] = None
        self.last_u_dev: float = 0.0
        self.last_solution: Optional[np.ndarray] = None
        self.last_meal_correction_time: Optional[datetime] = None
        self.solve_times = []
        self.solve_successes = []

        self.p1 = 0.98
        self.p2 = 0.965
        self.K = 90.0 * (self.p1 - 1.0) * (self.p2 - 1.0) ** 2
        self.A = np.array(
            [
                [
                    self.p1 + 2.0 * self.p2,
                    -(2.0 * self.p1 * self.p2 + self.p2**2),
                    self.p1 * self.p2**2,
                ],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=float,
        )
        self.Cy = np.array([0.0, 0.0, 1.0], dtype=float)
        self.Cv = np.array([0.1, 0.0, -0.1], dtype=float)
        self.observer_gain = self._make_observer_gain(observer_gain)

        # 2018 velocity-weighting/penalty parameters.
        self.r_plus_previous = 7000.0
        self.r_plus_velocity = 6500.0
        self.r_minus_const = 100.0
        self.velocity_penalty_D = 1000.0
        self.velocity_penalty_interval = (140.0, 180.0)
        self.q_epsilon = 1e-6

        # 2019 adaptive R+ surface parameters, represented as
        # theta=(delta, a1, a2, b1, alpha, eta, ell), tau.
        # The table in the paper is partially ambiguous in text extraction;
        # eta=130/180 follows the visible table and surface minima.
        self.theta_plus_H = (16500.0, 0.14, 0.32, 5500.0, 0.75, 130.0, 0.20)
        self.theta_plus_L = (15500.0, 0.11, 0.20, 2000.0, 0.75, 130.0, 0.20)
        self.tau_plus = 0.20
        self.theta_minus_H = (1_000_000.0, 0.03, 0.02, 5000.0, 1.0, 180.0, 0.20)
        self.theta_minus_L = (1_000_000.0, 0.03, 0.02, 4910.0, 1.0, 180.0, 0.20)
        self.tau_minus = 0.20

    def policy(self, observation, reward, done, **kwargs):
        sample_time = float(kwargs.get("sample_time", self.model_sample_time))
        patient_name = kwargs.get("patient_name")
        now = kwargs.get("time")
        meal_rate = float(kwargs.get("meal", 0.0) or 0.0)
        iob = kwargs.get("iob")

        if patient_name is None:
            patient_name = "Average"
        self._ensure_profile(patient_name)

        glucose = float(getattr(observation, "CGM", np.nan))
        if not np.isfinite(glucose):
            glucose = float(getattr(observation, "BG"))

        self._update_observer(glucose)
        u_dev = self._solve_mpc(glucose, sample_time, now=now, iob=iob)
        u_dev = self._apply_low_glucose_suspend(u_dev, glucose, sample_time)
        bolus_rate = self._meal_bolus_rate(
            meal_rate=meal_rate, glucose=glucose, sample_time=sample_time, now=now
        )

        basal_units = self.profile.basal_rate * sample_time
        absolute_units = max(basal_units + u_dev, 0.0)
        basal_rate = absolute_units / sample_time

        self.last_u_dev = float(u_dev)
        return Action(basal=basal_rate, bolus=bolus_rate)

    def reset(self):
        self.xhat = None
        self.last_u_dev = 0.0
        self.last_solution = None
        self.last_meal_correction_time = None
        self.solve_times = []
        self.solve_successes = []

    # ------------------------------------------------------------------
    # Patient profile and model state
    # ------------------------------------------------------------------
    def _ensure_profile(self, patient_name: str) -> None:
        if self.profile is not None and self.patient_name == patient_name:
            return

        params = self.patient_params.loc[self.patient_params.Name == patient_name]
        quest = self.quest.loc[self.quest.Name == patient_name]

        if params.empty:
            logger.warning("Unknown patient %s; using average control profile.", patient_name)
            basal_rate = 1.43 * 57.0 / 6000.0
            tdi = 30.0
            cf = 50.0
            cr = 15.0
        else:
            p = params.iloc[0]
            basal_rate = float(p.u2ss) * float(p.BW) / 6000.0
            if quest.empty:
                tdi = max(float(basal_rate) * 60.0 * 24.0, 1e-6)
                cf = 1800.0 / tdi
                cr = 15.0
            else:
                q = quest.iloc[0]
                tdi = float(q.TDI)
                cf = float(q.CF)
                cr = float(q.CR)

        self.profile = PatientControlProfile(
            basal_rate=basal_rate,
            tdi=max(tdi, 1e-6),
            cf=max(cf, 1e-6),
            cr=max(cr, 1e-6),
        )
        self.patient_name = patient_name
        self.B = (self.model_gain_factor * 1800.0 * self.K / self.profile.tdi) * np.array(
            [1.0, 0.0, 0.0], dtype=float
        )
        self.reset()

    def _update_observer(self, glucose: float) -> None:
        y = glucose - self.ys
        if self.xhat is None:
            # x[2] is current y; x[1]/x[0] are short-ahead outputs.
            self.xhat = np.array([y, y, y], dtype=float)
            return

        x_pred = self.A @ self.xhat + self.B * self.last_u_dev
        innovation = y - float(self.Cy @ x_pred)
        self.xhat = x_pred + self.observer_gain * innovation

    def _apply_low_glucose_suspend(
        self, u_dev: float, glucose: float, sample_time: float
    ) -> float:
        if not self.enable_low_glucose_suspend or self.profile is None:
            return float(u_dev)

        basal_units = self.profile.basal_rate * sample_time
        velocity = float(self.Cv @ self.xhat) if self.xhat is not None else 0.0
        should_suspend = glucose <= self.suspend_glucose or (
            glucose <= self.predictive_suspend_glucose
            and velocity <= self.predictive_suspend_velocity
        )
        if not should_suspend:
            return float(u_dev)
        return float(min(u_dev, -basal_units))

    # ------------------------------------------------------------------
    # MPC optimization
    # ------------------------------------------------------------------
    def _solve_mpc(
        self,
        glucose: float,
        sample_time: float,
        now: Optional[datetime] = None,
        iob: Optional[float] = None,
    ) -> float:
        if self.xhat is None:
            return 0.0

        bounds = self._input_bounds(sample_time=sample_time, now=now, glucose=glucose, iob=iob)
        x0 = self._initial_guess(bounds)

        def objective(u_seq: np.ndarray) -> float:
            return self._objective(u_seq, glucose=glucose, sample_time=sample_time, now=now)

        tic = time.perf_counter()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step.*",
                category=RuntimeWarning,
            )
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                bounds=bounds,
                options={"maxiter": self.max_slsqp_iter, "ftol": 1e-5, "disp": False},
            )
        self.solve_times.append(time.perf_counter() - tic)
        self.solve_successes.append(bool(result.success))

        if result.success and np.all(np.isfinite(result.x)):
            sol = np.asarray(result.x, dtype=float)
            self.last_solution = sol
            return float(sol[0])

        logger.debug("MPC solve failed: %s", getattr(result, "message", "unknown"))
        self.last_solution = x0
        return float(x0[0])

    def _objective(
        self,
        u_seq: np.ndarray,
        glucose: float,
        sample_time: float,
        now: Optional[datetime],
    ) -> float:
        x = np.array(self.xhat, dtype=float)
        cost = 0.0
        current_abs_bg = glucose
        d_hat = self.velocity_penalty_D if self._velocity_penalty_active(current_abs_bg) else 0.0

        for k in range(self.Ny):
            u = float(u_seq[k]) if k < self.Nu else 0.0
            x = self.A @ x + self.B * u
            y_abs = float(self.Cy @ x + self.ys)
            v = float(self.Cv @ x)
            lower, upper = self._target_zone(now, k, sample_time)
            z_low = max(lower - y_abs, 0.0)
            z_high = max(y_abs - upper, 0.0)

            if self.variant in {"velocity", "adaptive"}:
                q = self._velocity_weight(v)
            else:
                q = 1.0

            cost += z_low**2 + q * z_high**2 + d_hat * max(v, 0.0) ** 2

            if k < self.Nu:
                r_plus, r_minus = self._input_penalties(y_abs=y_abs, velocity=v)
                cost += r_plus * max(u, 0.0) ** 2 + r_minus * min(u, 0.0) ** 2

        return float(cost)

    def _input_bounds(
        self,
        sample_time: float,
        now: Optional[datetime],
        glucose: float,
        iob: Optional[float],
    ):
        basal_units = self.profile.basal_rate * sample_time
        bounds = []
        for k in range(self.Nu):
            upper_abs = self._absolute_insulin_upper(now, k, sample_time, basal_units)
            lower_dev = -basal_units
            upper_dev = upper_abs - basal_units
            if k == 0 and self.use_iob_constraint and iob is not None:
                try:
                    required_iob = (glucose - self.ys) / self.profile.cf
                    iob_limit = max(required_iob - float(iob), 0.0)
                    upper_dev = min(upper_dev, iob_limit)
                except (TypeError, ValueError):
                    pass
            if upper_dev < lower_dev:
                upper_dev = lower_dev
            bounds.append((float(lower_dev), float(upper_dev)))
        return bounds

    def _initial_guess(self, bounds):
        if self.last_solution is None:
            guess = np.zeros(self.Nu, dtype=float)
        else:
            guess = np.r_[self.last_solution[1:], 0.0]
        for i, (lo, hi) in enumerate(bounds):
            guess[i] = min(max(guess[i], lo), hi)
        return guess

    # ------------------------------------------------------------------
    # Cost components
    # ------------------------------------------------------------------
    def _input_penalties(self, y_abs: float, velocity: float) -> Tuple[float, float]:
        if self.variant == "previous":
            return self.r_plus_scale * self.r_plus_previous, self.r_minus_const
        if self.variant == "velocity":
            return self.r_plus_scale * self.r_plus_velocity, self.r_minus_const
        return (
            self.r_plus_scale * self._adaptive_r_plus(y_abs, velocity),
            self._adaptive_r_minus(y_abs),
        )

    def _velocity_weight(self, velocity: float) -> float:
        if velocity >= 0.0:
            return 1.0
        if velocity <= -1.0:
            return self.q_epsilon
        return 0.5 * (np.cos(np.pi * velocity) * (1.0 - self.q_epsilon) + (1.0 + self.q_epsilon))

    def _velocity_penalty_active(self, glucose: float) -> bool:
        if self.variant == "previous":
            return False
        lo, hi = self.velocity_penalty_interval
        return lo <= glucose <= hi

    def _adaptive_r_plus(self, y_abs: float, velocity: float) -> float:
        if velocity >= 0.0:
            upper = self._bowl(y_abs, self.theta_plus_H)
            lower = self._bowl(y_abs, self.theta_plus_L)
            value = lower + np.exp(-self.tau_plus * velocity) * (upper - lower)
        else:
            upper = self._bowl(y_abs, self.theta_minus_H)
            lower = self._bowl(y_abs, self.theta_minus_L)
            value = upper - np.exp(self.tau_minus * velocity) * (upper - lower)
        return float(np.clip(value, 1.0, 1_000_000.0))

    def _adaptive_r_minus(self, y_abs: float) -> float:
        if y_abs > 140.0:
            return 100.0
        if y_abs >= 120.0:
            return 10.0
        return 1.0

    @staticmethod
    def _bowl(y_abs: float, theta: Tuple[float, float, float, float, float, float, float]) -> float:
        delta, a1, a2, b1, alpha, eta, ell = theta
        if y_abs <= eta:
            exponent = np.clip(a2 * (eta - y_abs + ell), -50.0, 50.0)
            offset = np.exp(np.clip(a2 * ell, -50.0, 50.0))
            value = np.exp(exponent) + b1 - offset
        else:
            exponent = np.clip(a1 * (y_abs - eta) ** alpha, -50.0, 50.0)
            value = np.exp(exponent) + b1
        return float(min(delta, value))

    def _make_observer_gain(
        self, observer_gain: Optional[Tuple[float, float, float]]
    ) -> np.ndarray:
        if observer_gain is not None:
            return np.asarray(observer_gain, dtype=float)

        q = np.eye(3)
        r = np.array([[1000.0]])
        c = self.Cy.reshape(1, 3)
        p = solve_discrete_are(self.A.T, c.T, q, r)
        gain = (self.A @ p @ c.T) @ np.linalg.inv(r + c @ p @ c.T)
        return gain.ravel()

    # ------------------------------------------------------------------
    # Zones, constraints, and optional meal bolus
    # ------------------------------------------------------------------
    def _target_zone(
        self, now: Optional[datetime], step: int, sample_time: float
    ) -> Tuple[float, float]:
        t = self._future_time(now, step, sample_time)
        hour = t.hour + t.minute / 60.0 if t is not None else 12.0

        if 6.0 <= hour < 22.0:
            lower = 80.0
        elif hour >= 22.0:
            lower = 80.0 + (hour - 22.0) / 2.0 * 10.0
        elif hour < 4.0:
            lower = 90.0
        else:
            lower = 90.0 - (hour - 4.0) / 2.0 * 10.0
        return float(lower), 140.0

    def _absolute_insulin_upper(
        self,
        now: Optional[datetime],
        step: int,
        sample_time: float,
        basal_units: float,
    ) -> float:
        t = self._future_time(now, step, sample_time)
        hour = t.hour + t.minute / 60.0 if t is not None else 12.0
        scale = sample_time / self.model_sample_time
        if 4.0 <= hour < 22.0:
            upper = self.max_insulin_units_per_sample * scale
            if self.max_insulin_tdi_fraction is not None:
                upper = min(upper, self.max_insulin_tdi_fraction * self.profile.tdi)
            return upper
        return 1.8 * basal_units

    @staticmethod
    def _future_time(
        now: Optional[datetime], step: int, sample_time: float
    ) -> Optional[datetime]:
        if now is None:
            return None
        return now + timedelta(minutes=step * sample_time)

    def _meal_bolus_rate(
        self,
        meal_rate: float,
        glucose: float,
        sample_time: float,
        now: Optional[datetime],
    ) -> float:
        if not self.announce_meals or meal_rate <= 0.0:
            return 0.0

        meal_grams = meal_rate * sample_time
        basic_units = self.meal_bolus_scale * meal_grams / self.profile.cr
        if glucose >= 140.0:
            correction_allowed = (
                self.last_meal_correction_time is None
                or now is None
                or now - self.last_meal_correction_time >= timedelta(hours=2)
            )
            correction_units = 0.0
            if correction_allowed:
                correction_units = min((glucose - 140.0) / self.profile.cf, 2.0)
                if now is not None and correction_units > 0.0:
                    self.last_meal_correction_time = now
            total_units = basic_units + max(correction_units, 0.0)
        else:
            total_units = 0.8 * basic_units
        return total_units / sample_time


class MPCController(ZoneMPCController):
    """Backward-compatible name used by simglucose.controller.__init__."""

    pass
