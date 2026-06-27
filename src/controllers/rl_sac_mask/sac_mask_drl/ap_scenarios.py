from __future__ import annotations

import sys
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import truncnorm


def add_local_simglucose_path() -> None:
    project_root = Path(__file__).resolve().parents[5]
    sim_root = project_root / "src" / "envs" / "simglucose_mpc"
    if sim_root.exists() and str(sim_root) not in sys.path:
        sys.path.insert(0, str(sim_root))
    dateutil_site = Path.home() / "AppData/Local/Programs/Python/Python39/lib/site-packages"
    if (dateutil_site / "dateutil").exists() and str(dateutil_site) not in sys.path:
        sys.path.append(str(dateutil_site))


add_local_simglucose_path()

from simglucose.simulation.scenario import Scenario  # noqa: E402


PatientAction = namedtuple("patient_action", ["meal", "insulin"])


class PaperTrainingScenario(Scenario):
    """Supplement setting: 3 randomized meals and 3 randomized snacks per day."""

    def __init__(self, start_time: datetime, seed: int | None = None, patient_name: str | None = None):
        super().__init__(start_time)
        self.random_gen = np.random.RandomState(seed)
        self.patient_name = patient_name or ""
        self.meals = self._generate_daily_meals()

    def _generate_daily_meals(self):
        meal_times = np.array([7.0, 9.5, 12.0, 15.0, 18.0, 21.5]) * 60.0
        time_std = np.array([60.0, 30.0, 60.0, 30.0, 60.0, 30.0])
        if self.patient_name.startswith("child#"):
            amounts = np.array([40.0, 10.0, 55.0, 10.0, 55.0, 10.0])
        else:
            amounts = np.array([50.0, 10.0, 75.0, 10.0, 75.0, 10.0])
        amount_std = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        probabilities = np.ones(6)
        return self._sample_meals(meal_times, time_std, amounts, amount_std, probabilities)

    def _sample_meals(self, meal_times, time_std, amounts, amount_std, probabilities):
        meals = []
        for idx in range(len(meal_times)):
            if self.random_gen.rand() > probabilities[idx]:
                continue
            lower = max(0.0, meal_times[idx] - 3.0 * time_std[idx])
            upper = min(24.0 * 60.0 - 1.0, meal_times[idx] + 3.0 * time_std[idx])
            a = (lower - meal_times[idx]) / time_std[idx]
            b = (upper - meal_times[idx]) / time_std[idx]
            time_min = truncnorm.rvs(a, b, loc=meal_times[idx], scale=time_std[idx], random_state=self.random_gen)
            amount = self.random_gen.normal(amounts[idx], amount_std[idx])
            amount = float(max(0.0, amount))
            meals.append((int(round(time_min)), amount))
        meals.sort(key=lambda item: item[0])
        return meals

    def get_action(self, t):
        minute = t.hour * 60 + t.minute
        for meal_time, amount in self.meals:
            if minute == meal_time:
                return PatientAction(meal=amount, insulin=0)
        return PatientAction(meal=0, insulin=0)

    def reset(self):
        self.meals = self._generate_daily_meals()


class ScenarioA(PaperTrainingScenario):
    """Paper Scenario A for adult evaluation."""

    def _generate_daily_meals(self):
        meal_times = np.array([8.0, 9.5, 13.0, 15.0, 19.0, 21.5]) * 60.0
        time_std = np.array([60.0, 60.0, 60.0, 60.0, 60.0, 60.0])
        amounts = np.array([50.0, 10.0, 75.0, 30.0, 75.0, 20.0])
        amount_std = amounts * 0.40
        probabilities = np.array([0.75, 0.30, 0.75, 0.30, 0.75, 0.30])
        return self._sample_meals(meal_times, time_std, amounts, amount_std, probabilities)


class ScenarioATransformed(PaperTrainingScenario):
    """Scenario A with simple time and carbohydrate perturbations."""

    def __init__(
        self,
        start_time: datetime,
        seed: int | None = None,
        patient_name: str | None = None,
        time_shift_min: float = 0.0,
        carb_scale: float = 1.0,
    ):
        self.time_shift_min = float(time_shift_min)
        self.carb_scale = float(carb_scale)
        super().__init__(start_time, seed=seed, patient_name=patient_name)

    def _generate_daily_meals(self):
        meal_times = np.array([8.0, 9.5, 13.0, 15.0, 19.0, 21.5]) * 60.0
        meal_times = np.clip(meal_times + self.time_shift_min, 0.0, 24.0 * 60.0 - 1.0)
        time_std = np.array([60.0, 60.0, 60.0, 60.0, 60.0, 60.0])
        amounts = np.array([50.0, 10.0, 75.0, 30.0, 75.0, 20.0]) * self.carb_scale
        amount_std = amounts * 0.40
        probabilities = np.array([0.75, 0.30, 0.75, 0.30, 0.75, 0.30])
        return self._sample_meals(meal_times, time_std, amounts, amount_std, probabilities)


class NoMealScenario(Scenario):
    """Paper Scenario E base: no meals. Extra insulin shock is handled by the runner."""

    def get_action(self, t):
        return PatientAction(meal=0, insulin=0)

    def reset(self):
        pass


def make_paper_scenario(name: str, start_time: datetime, seed: int | None = None, patient_name: str | None = None):
    if name == "train":
        return PaperTrainingScenario(start_time, seed=seed, patient_name=patient_name)
    if name == "A":
        return ScenarioA(start_time, seed=seed)
    if name == "A_shift_early":
        return ScenarioATransformed(start_time, seed=seed, patient_name=patient_name, time_shift_min=-120.0)
    if name == "A_shift_late":
        return ScenarioATransformed(start_time, seed=seed, patient_name=patient_name, time_shift_min=120.0)
    if name == "A_carb_low":
        return ScenarioATransformed(start_time, seed=seed, patient_name=patient_name, carb_scale=0.70)
    if name == "A_carb_high":
        return ScenarioATransformed(start_time, seed=seed, patient_name=patient_name, carb_scale=1.30)
    if name == "E":
        return NoMealScenario(start_time)
    raise ValueError(f"Unsupported scenario: {name}")
