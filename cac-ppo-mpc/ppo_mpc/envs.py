from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import gamma

from .metrics import clinical_reward
from .paths import MPC_ASSETS_ROOT, add_asset_paths
from .scenarios import make_paper_scenario


add_asset_paths()

from simglucose.actuator.pump import InsulinPump  # noqa: E402
from simglucose.controller.base import Action as ControllerAction  # noqa: E402
from simglucose.patient.t1dpatient import T1DPatient  # noqa: E402
from simglucose.sensor.cgm import CGMSensor  # noqa: E402
from simglucose.simulation.env import T1DSimEnv  # noqa: E402


CONTROL_QUEST = MPC_ASSETS_ROOT / 'simglucose' / 'params' / 'Quest.csv'
STATE_DIM = 7


def patient_basal_rate(patient) -> float:
    params = patient._params
    return float(params["u2ss"]) * float(params["BW"]) / 6000.0


@lru_cache(maxsize=64)
def patient_correction_factor(patient_name: str) -> float:
    """Return the prescribed correction factor (mg/dL/U), not simulator state."""
    table = pd.read_csv(CONTROL_QUEST).set_index("Name")
    if patient_name not in table.index:
        raise KeyError(f"missing prescribed correction factor for {patient_name}")
    value = float(table.loc[patient_name, "CF"])
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"invalid prescribed correction factor for {patient_name}: {value}")
    return value


def insulin_tail_factors(
    t_peak: float = 55.0,
    n_steps: int = 160,
    sample_time: float = 5.0,
):
    shape_k = 2.0
    scale_theta = t_peak / (shape_k - 1.0)
    ages_min = np.arange(n_steps, dtype=float) * float(sample_time)
    return gamma.sf(ages_min, a=shape_k, scale=scale_theta)


class GlucoseStateTracker:
    def __init__(self, body_weight: float, basal_rate: float, sample_time: float = 5.0):
        self.body_weight = float(body_weight)
        self.basal_rate = float(max(basal_rate, 1e-8))
        self.sample_time = float(sample_time)
        self.glucose_history = deque(maxlen=2)
        self.insulin_history = deque(maxlen=160)
        self.iob_tail = insulin_tail_factors(
            n_steps=self.insulin_history.maxlen,
            sample_time=self.sample_time,
        )
        self.last_basal_rate = self.basal_rate

    def reset(self, glucose: float) -> None:
        self.glucose_history.clear()
        self.glucose_history.append(float(glucose))
        self.glucose_history.append(float(glucose))
        self.insulin_history.clear()
        # The prescribed basal before t=0 is known causal information.  Seed a
        # steady-state basal history instead of introducing an artificial
        # zero-IOB transient at every episode reset.
        for _ in range(self.insulin_history.maxlen):
            self.insulin_history.append(self.basal_rate)
        self.last_basal_rate = self.basal_rate

    def record_action(self, basal_rate: float, bolus_rate: float = 0.0) -> None:
        self.last_basal_rate = float(basal_rate)
        self.insulin_history.append(float(basal_rate) + float(bolus_rate))

    def current_iob(self) -> float:
        history = np.asarray(list(self.insulin_history)[::-1], dtype=np.float32)
        # History stores U/min. Multiplication by the sample duration produces
        # insulin units before applying the fraction still active at each age.
        return float(np.sum(history * self.sample_time * self.iob_tail))

    def steady_state_iob(self) -> float:
        """IOB (U) implied by the known prescribed basal before episode start."""
        return float(self.basal_rate * self.sample_time * np.sum(self.iob_tail))

    def make_state(
        self,
        glucose: float,
        *,
        meal_grams: float = 0.0,
        bolus_rate: float = 0.0,
        now: datetime | None = None,
    ) -> np.ndarray:
        glucose = float(glucose)
        previous = float(self.glucose_history[-1])
        self.glucose_history.append(glucose)
        rate = (glucose - previous) / self.sample_time
        hour = 0.0
        if now is not None:
            hour = float(now.hour) + float(now.minute) / 60.0
        angle = 2.0 * np.pi * hour / 24.0
        return np.asarray(
            [
                glucose / 400.0,
                rate / 5.0,
                self.current_iob() / max(60.0 * self.basal_rate, 1e-8),
                float(meal_grams) / 100.0,
                float(bolus_rate) / max(10.0 * self.basal_rate, 1e-8),
                np.sin(angle),
                np.cos(angle),
            ],
            dtype=np.float32,
        )


def map_raw_action_to_basal(
    raw_action,
    basal_rate: float,
    max_basal: float,
    max_basal_multiplier: float = 5.0,
) -> np.ndarray:
    raw = float(np.asarray(raw_action).reshape(-1)[0])
    normalized = (np.clip(raw, -1.0, 1.0) + 1.0) / 2.0
    basal = normalized * float(max_basal_multiplier) * float(basal_rate)
    return np.asarray([np.clip(basal, 0.0, max_basal)], dtype=np.float32)


def basal_to_raw_action(
    basal_rate_command: float,
    basal_rate: float,
    max_basal_multiplier: float = 5.0,
) -> np.ndarray:
    normalized = float(basal_rate_command) / max(float(max_basal_multiplier) * float(basal_rate), 1e-8)
    raw = 2.0 * np.clip(normalized, 0.0, 1.0) - 1.0
    return np.asarray([raw], dtype=np.float32)


class PPOGlucoseEnv:
    """Small Gym-like wrapper used by PPO training."""

    def __init__(
        self,
        patient_name: str = "adult#001",
        scenario: str = "train",
        seed: int = 1,
        max_steps: int = 288,
        meal_announcement: str = "announced",
        sensor_name: str = "GuardianRT",
        pump_name: str = "Insulet",
        start_time: datetime | None = None,
        target_glucose: float = 140.0,
        max_basal_multiplier: float = 5.0,
    ):
        self.patient_name = patient_name
        self.scenario_name = scenario
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.meal_announcement = meal_announcement
        self.sensor_name = sensor_name
        self.pump_name = pump_name
        self.start_time = start_time or datetime(2018, 1, 1, 0, 0, 0)
        self.target_glucose = float(target_glucose)
        self.max_basal_multiplier = float(max_basal_multiplier)
        self.quest = pd.read_csv(CONTROL_QUEST).set_index("Name")
        self.elapsed_steps = 0
        self.env = self._create_env(self.seed)
        self.patient = self.env.patient
        self.basal_rate = patient_basal_rate(self.patient)
        self.max_basal = float(self.env.pump._params["max_basal"])
        body_weight = float(self.patient._params["BW"])
        self.tracker = GlucoseStateTracker(body_weight, self.basal_rate, self.env.sample_time)
        self.previous_basal = self.basal_rate

    @property
    def sample_time(self) -> float:
        return float(self.env.sample_time)

    def _create_env(self, seed: int) -> T1DSimEnv:
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed + 101)
        sensor = CGMSensor.withName(self.sensor_name, seed=seed + 202)
        pump = InsulinPump.withName(self.pump_name)
        scenario = make_paper_scenario(
            self.scenario_name,
            self.start_time,
            seed=seed + 303,
            patient_name=self.patient_name,
        )
        return T1DSimEnv(patient, sensor, pump, scenario)

    def _announced_meal_in_next_sample(self) -> float:
        if self.meal_announcement != "announced":
            return 0.0
        total = 0.0
        for minute_offset in range(int(self.sample_time)):
            action = self.env.scenario.get_action(self.env.time + timedelta(minutes=minute_offset))
            total += float(getattr(action, "meal", 0.0))
        return float(total)

    def meal_bolus_rate(self, meal_grams: float, glucose: float) -> float:
        if self.meal_announcement != "announced" or meal_grams <= 0.0:
            return 0.0
        if self.patient_name in self.quest.index:
            cr = float(self.quest.loc[self.patient_name, "CR"])
            cf = float(self.quest.loc[self.patient_name, "CF"])
        else:
            cr = 15.0
            cf = 50.0
        meal_bolus = float(meal_grams) / max(cr, 1e-8)
        if glucose < 120.0:
            meal_bolus *= 0.8
        correction = 0.0
        if glucose > 150.0:
            correction = min((glucose - self.target_glucose) / max(cf, 1e-8), 2.0)
        return max(0.0, meal_bolus + correction) / max(self.sample_time, 1e-8)

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.seed = int(seed)
        self.env = self._create_env(self.seed)
        self.patient = self.env.patient
        self.basal_rate = patient_basal_rate(self.patient)
        self.max_basal = float(self.env.pump._params["max_basal"])
        body_weight = float(self.patient._params["BW"])
        self.tracker = GlucoseStateTracker(body_weight, self.basal_rate, self.env.sample_time)
        self.elapsed_steps = 0
        obs, _reward, _done, _info = self.env.reset()
        glucose = float(obs.CGM)
        self.tracker.reset(glucose)
        self.previous_basal = self.basal_rate
        meal_grams = self._announced_meal_in_next_sample()
        state = self.tracker.make_state(glucose, meal_grams=meal_grams, now=self.env.time)
        return state

    def step(self, raw_action):
        basal_rate = float(
            map_raw_action_to_basal(
                raw_action,
                self.basal_rate,
                self.max_basal,
                self.max_basal_multiplier,
            )[0]
        )
        current_cgm = float(self.env.CGM_hist[-1])
        meal_grams = self._announced_meal_in_next_sample()
        bolus_rate = self.meal_bolus_rate(meal_grams, current_cgm)
        obs, _sim_reward, done, info = self.env.step(
            ControllerAction(basal=basal_rate, bolus=bolus_rate)
        )
        self.elapsed_steps += 1
        truncated = self.elapsed_steps >= self.max_steps
        next_cgm = float(obs.CGM)
        reward = clinical_reward(
            next_cgm,
            basal_rate=basal_rate,
            basal_reference=self.basal_rate,
            previous_basal=self.previous_basal,
        )
        self.tracker.record_action(basal_rate, bolus_rate)
        next_meal = self._announced_meal_in_next_sample()
        next_state = self.tracker.make_state(
            next_cgm,
            meal_grams=next_meal,
            bolus_rate=bolus_rate,
            now=self.env.time,
        )
        self.previous_basal = basal_rate
        merged_info = dict(info)
        merged_info.update(
            {
                "patient_name": self.patient_name,
                "basal_rate": basal_rate,
                "bolus_rate": bolus_rate,
                "meal_grams": meal_grams if self.meal_announcement == "announced" else 0.0,
                "raw_action": float(np.asarray(raw_action).reshape(-1)[0]),
                "sample_time": self.sample_time,
            }
        )
        return next_state, float(reward), bool(done), bool(truncated), merged_info

    def close(self) -> None:
        if hasattr(self.env, "_close_viewer"):
            self.env._close_viewer()
