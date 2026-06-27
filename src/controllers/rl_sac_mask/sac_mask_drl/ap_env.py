from __future__ import annotations

import sys
import uuid
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import gymnasium
import numpy as np
import pandas as pd
import pkg_resources
from gymnasium.envs.registration import register
from gymnasium import spaces
from scipy.stats import gamma

from .ap_scenarios import add_local_simglucose_path, make_paper_scenario


add_local_simglucose_path()

from simglucose.actuator.pump import InsulinPump  # noqa: E402
from simglucose.controller.base import Action as ControllerAction  # noqa: E402
from simglucose.patient.t1dpatient import T1DPatient  # noqa: E402
from simglucose.sensor.cgm import CGMSensor  # noqa: E402
from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv  # noqa: E402


CONTROL_QUEST = pkg_resources.resource_filename("simglucose", "params/Quest.csv")


def adult_patients():
    return [f"adult#{idx:03d}" for idx in range(1, 11)]


def patient_basal_rate(patient) -> float:
    params = patient._params
    return float(params["u2ss"]) * float(params["BW"]) / 6000.0


def zone_reward(glucose: float) -> float:
    g = float(glucose)
    if g < 39.0:
        return -20.0
    if g > 400.0:
        return -10.0
    if g < 54.0:
        return 0.0
    if g < 100.0:
        return 1.0 - 0.00201 * ((100.0 - g) ** 1.622)
    if g <= 140.0:
        return 1.0
    if g <= 300.0:
        return 1.0 - 0.00473 * ((g - 140.0) ** 0.918)
    return 0.5


def insulin_tail_factors(t_peak=55.0, t_end=480.0, n_steps=160):
    shape_k = 2.0
    scale_theta = t_peak / (shape_k - 1.0)
    time_points = np.linspace(0.0, t_end, n_steps)
    cdf_values = gamma.cdf(time_points, a=shape_k, scale=scale_theta)
    return 1.0 - cdf_values


class APStateTracker:
    def __init__(
        self,
        body_weight: float,
        basal_rate: float,
        sample_time: float = 5.0,
        state_mode: str = "paper",
    ):
        self.body_weight = float(body_weight)
        self.basal_rate = float(max(basal_rate, 1e-8))
        self.sample_time = float(sample_time)
        self.state_mode = state_mode
        self.glucose_history = deque(maxlen=2)
        self.insulin_history = deque(maxlen=160)
        self.iob_tail = insulin_tail_factors()
        self.last_bolus_rate = 0.0
        self.last_meal_grams = 0.0

    def reset(self, glucose: float, bolus_rate: float = 0.0, meal_grams: float = 0.0) -> np.ndarray:
        self.glucose_history.clear()
        self.glucose_history.append(float(glucose))
        self.glucose_history.append(float(glucose))
        self.insulin_history.clear()
        for _ in range(160):
            self.insulin_history.append(0.0)
        self.last_bolus_rate = float(bolus_rate)
        self.last_meal_grams = float(meal_grams)
        return self.make_state(glucose, bolus_rate=bolus_rate, meal_grams=meal_grams)

    def record_action(self, basal_rate: float, bolus_rate: float = 0.0) -> None:
        self.insulin_history.append(float(basal_rate) + float(bolus_rate))

    def current_iob(self) -> float:
        history = np.asarray(list(self.insulin_history)[::-1], dtype=np.float32)
        return float(np.sum(history * self.iob_tail))

    def make_state(self, glucose: float, bolus_rate: float | None = None, meal_grams: float | None = None) -> np.ndarray:
        glucose = float(glucose)
        previous = self.glucose_history[-1]
        self.glucose_history.append(glucose)
        rate = (glucose - previous) / self.sample_time
        iob = self.current_iob()
        if bolus_rate is None:
            bolus_rate = self.last_bolus_rate
        if meal_grams is None:
            meal_grams = self.last_meal_grams
        self.last_bolus_rate = float(bolus_rate)
        self.last_meal_grams = float(meal_grams)
        if self.state_mode == "paper":
            return np.asarray(
                [
                    glucose / 400.0,
                    float(bolus_rate) / max(10.0 * self.basal_rate, 1e-8),
                    float(meal_grams) / 100.0,
                    iob / max(10.0 * self.basal_rate, 1e-8),
                ],
                dtype=np.float32,
            )
        if self.state_mode != "legacy":
            raise ValueError(f"Unsupported state_mode: {self.state_mode}")
        return np.asarray(
            [
                glucose / 400.0,
                rate / 5.0,
                iob / (10.0 * self.basal_rate),
                self.body_weight / 100.0,
            ],
            dtype=np.float32,
        )


class PaperSimglucoseEnv(gymnasium.Env):
    """Paper-oriented AP wrapper.

    The stock simglucose Gym wrapper hard-codes bolus=0 and uses Dexcom
    sample_time=3. This wrapper exposes the same one-dimensional basal action
    while internally adding paper-style meal bolus and using 5-minute CGM.
    """

    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(
        self,
        patient_name: str,
        scenario_name: str,
        seed: int,
        max_steps: int,
        meal_announcement: str = "announced",
        sensor_name: str = "GuardianRT",
        pump_name: str = "Insulet",
        start_time: datetime | None = None,
        target_glucose: float = 140.0,
    ):
        super().__init__()
        self.patient_name = patient_name
        self.scenario_name = scenario_name
        self.seed = int(seed)
        self.max_steps = int(max_steps)
        self.meal_announcement = meal_announcement
        self.sensor_name = sensor_name
        self.pump_name = pump_name
        self.start_time = start_time or datetime(2018, 1, 1, 0, 0, 0)
        self.target_glucose = float(target_glucose)
        self.quest = pd.read_csv(CONTROL_QUEST).set_index("Name")
        self._rng = np.random.RandomState(self.seed)
        self.elapsed_steps = 0
        self.last_meal_grams = 0.0
        self.last_bolus_rate = 0.0
        self.env = self._create_env(self.seed)
        self.action_space = spaces.Box(low=0.0, high=float(self.env.pump._params["max_basal"]), shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1000.0, shape=(1,), dtype=np.float32)

    @property
    def sample_time(self) -> float:
        return float(self.env.sample_time)

    def _create_env(self, seed: int):
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed + 101)
        sensor = CGMSensor.withName(self.sensor_name, seed=seed + 202)
        pump = InsulinPump.withName(self.pump_name)
        scenario = make_paper_scenario(self.scenario_name, self.start_time, seed=seed + 303, patient_name=self.patient_name)
        return _T1DSimEnv(patient, sensor, pump, scenario)

    def _announced_meal_in_next_sample(self) -> float:
        if self.meal_announcement != "announced":
            return 0.0
        total = 0.0
        for minute_offset in range(int(self.sample_time)):
            action = self.env.scenario.get_action(self.env.time + timedelta(minutes=minute_offset))
            total += float(getattr(action, "meal", 0.0))
        return total

    def _meal_bolus_rate(self, meal_grams: float, glucose: float) -> float:
        if self.meal_announcement != "announced" or meal_grams <= 0:
            return 0.0
        if self.patient_name in self.quest.index:
            cr = float(self.quest.loc[self.patient_name, "CR"])
            cf = float(self.quest.loc[self.patient_name, "CF"])
        else:
            cr = 15.0
            cf = 50.0
        meal_bolus = float(meal_grams) / cr
        if glucose < 120.0:
            meal_bolus *= 0.8
        correction = 0.0
        if glucose > 150.0:
            correction = min((glucose - self.target_glucose) / cf, 2.0)
        total_units = max(0.0, meal_bolus + correction)
        return total_units / max(self.sample_time, 1e-8)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed = int(seed)
        self.env = self._create_env(self.seed)
        self.elapsed_steps = 0
        self.last_meal_grams = 0.0
        self.last_bolus_rate = 0.0
        obs, _, _, info = self.env.reset()
        info = dict(info)
        info["meal_grams"] = 0.0
        info["bolus_rate"] = 0.0
        return np.array([obs.CGM], dtype=np.float32), info

    def step(self, action):
        if hasattr(action, "basal"):
            basal_rate = float(action.basal)
            bolus_override = float(getattr(action, "bolus", 0.0))
        else:
            action_arr = np.asarray(action).reshape(-1)
            basal_rate = float(action_arr[0])
            bolus_override = float(action_arr[1]) if action_arr.size > 1 else None
        current_cgm = float(self.env.CGM_hist[-1])
        meal_grams = self._announced_meal_in_next_sample()
        if bolus_override is None:
            bolus_rate = self._meal_bolus_rate(meal_grams, current_cgm)
        else:
            bolus_rate = bolus_override
        obs, reward, done, info = self.env.step(ControllerAction(basal=basal_rate, bolus=bolus_rate))
        self.elapsed_steps += 1
        truncated = self.elapsed_steps >= self.max_steps
        info = dict(info)
        info["meal_grams"] = meal_grams if self.meal_announcement == "announced" else 0.0
        info["actual_meal_rate"] = float(info.get("meal", 0.0))
        info["bolus_rate"] = bolus_rate
        info["sample_time"] = self.sample_time
        self.last_meal_grams = info["meal_grams"]
        self.last_bolus_rate = bolus_rate
        return np.array([obs.CGM], dtype=np.float32), reward, bool(done), bool(truncated), info

    def render(self):
        self.env.render()

    def close(self):
        self.env._close_viewer()


def make_simglucose_env(patient_name: str, scenario_name: str, seed: int, max_steps: int):
    patient = T1DPatient.withName(patient_name)
    start_time = datetime(2018, 1, 1, 0, 0, 0)
    scenario = make_paper_scenario(scenario_name, start_time, seed=seed, patient_name=patient_name)
    env_id = f"simglucose/hycpap-mask-{scenario_name}-{patient_name.replace('#', '-')}-{uuid.uuid4().hex[:8]}-v0"
    register(
        id=env_id,
        entry_point="simglucose.envs.simglucose_gym_env:T1DSimGymnaisumEnv",
        max_episode_steps=max_steps,
        kwargs={"patient_name": patient_name, "custom_scenario": scenario, "seed": seed},
    )
    return gymnasium.make(env_id), patient


def make_paper_simglucose_env(
    patient_name: str,
    scenario_name: str,
    seed: int,
    max_steps: int,
    meal_announcement: str = "announced",
    sensor_name: str = "GuardianRT",
):
    env = PaperSimglucoseEnv(
        patient_name=patient_name,
        scenario_name=scenario_name,
        seed=seed,
        max_steps=max_steps,
        meal_announcement=meal_announcement,
        sensor_name=sensor_name,
    )
    return env, env.env.patient


def map_action_to_basal(
    raw_action,
    basal_rate: float,
    max_basal: float,
    max_basal_multiplier: float,
    action_mapping: str = "zero_to_max",
    basal_delta_multiplier: float = 1.0,
):
    raw = float(np.asarray(raw_action).reshape(-1)[0])
    if action_mapping == "zero_to_max":
        normalized = (raw + 1.0) / 2.0
        basal = normalized * max_basal_multiplier * basal_rate
    elif action_mapping == "paper_basal_centered":
        if raw >= 0.0:
            multiplier = 1.0 + raw * (max_basal_multiplier - 1.0)
        else:
            multiplier = 1.0 + raw
        basal = multiplier * basal_rate
    elif action_mapping == "basal_delta":
        basal = (1.0 + raw * basal_delta_multiplier) * basal_rate
        basal = np.clip(basal, 0.0, max_basal_multiplier * basal_rate)
    else:
        raise ValueError(f"Unsupported action_mapping: {action_mapping}")
    return np.asarray([np.clip(basal, 0.0, max_basal)], dtype=np.float32)


def basal_multiplier_to_raw_action(
    basal_multiplier: float,
    max_basal_multiplier: float,
    action_mapping: str = "paper_basal_centered",
    basal_delta_multiplier: float = 1.0,
):
    multiplier = float(np.clip(basal_multiplier, 0.0, max_basal_multiplier))
    if action_mapping == "zero_to_max":
        raw = 2.0 * multiplier / max_basal_multiplier - 1.0
    elif action_mapping == "paper_basal_centered":
        if multiplier >= 1.0:
            raw = (multiplier - 1.0) / max(max_basal_multiplier - 1.0, 1e-8)
        else:
            raw = multiplier - 1.0
    elif action_mapping == "basal_delta":
        raw = (multiplier - 1.0) / max(basal_delta_multiplier, 1e-8)
    else:
        raise ValueError(f"Unsupported action_mapping: {action_mapping}")
    return np.asarray([np.clip(raw, -1.0, 1.0)], dtype=np.float32)


def apply_safety_guard(
    basal_action,
    state,
    hypo_threshold: float = 80.0,
    predictive_threshold: float = 75.0,
    prediction_minutes: float = 20.0,
    max_iob: float = 4.0,
    falling_rate_threshold: float = -1.5,
):
    glucose = float(state[0]) * 400.0
    rate = float(state[1]) * 5.0
    iob_norm = float(state[2])
    if glucose < hypo_threshold:
        return np.asarray([0.0], dtype=np.float32)
    if glucose + rate * prediction_minutes < predictive_threshold:
        return np.asarray([0.0], dtype=np.float32)
    if iob_norm >= max_iob:
        return np.asarray([0.0], dtype=np.float32)
    if rate < falling_rate_threshold:
        return np.asarray(0.5 * basal_action, dtype=np.float32)
    return basal_action


def clinical_metrics(glucose_values):
    arr = np.asarray(glucose_values, dtype=np.float32)
    return {
        "titr": 100.0 * np.mean((arr >= 70.0) & (arr <= 140.0)),
        "tir": 100.0 * np.mean((arr >= 70.0) & (arr <= 180.0)),
        "tbr70": 100.0 * np.mean(arr < 70.0),
        "tbr54": 100.0 * np.mean(arr < 54.0),
        "tar180": 100.0 * np.mean(arr > 180.0),
        "tar250": 100.0 * np.mean(arr > 250.0),
        "mean_bg": float(np.mean(arr)),
        "sd_bg": float(np.std(arr)),
    }
