from __future__ import annotations

from collections import namedtuple
from datetime import datetime

import numpy as np

from .paths import add_asset_paths


add_asset_paths()

from simglucose.simulation.scenario import Scenario  # noqa: E402


PatientAction = namedtuple("patient_action", ["meal", "insulin"])


def aggregate_same_minute_meals(
    meals: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Preserve total CHO when rounded stochastic meal times collide."""
    combined: dict[int, float] = {}
    for minute, amount in meals:
        minute_key = int(minute)
        combined[minute_key] = combined.get(minute_key, 0.0) + float(amount)
    return sorted(combined.items())


def _sample_clipped_normal(
    rng: np.random.RandomState,
    mean: float,
    std: float,
    low: float,
    high: float,
) -> float:
    if std <= 0:
        return float(np.clip(mean, low, high))
    value = float(rng.normal(mean, std))
    return float(np.clip(value, low, high))


class PaperTrainingScenario(Scenario):
    """Three randomized meals and three randomized snacks per day."""

    def __init__(
        self,
        start_time: datetime,
        seed: int | None = None,
        patient_name: str | None = None,
    ):
        super().__init__(start_time)
        self.random_gen = np.random.RandomState(seed)
        self.patient_name = patient_name or ""
        self.meals = self._generate_daily_meals()
        self._meals_by_day = {self.start_time.date(): self.meals}

    def _generate_daily_meals(self) -> list[tuple[int, float]]:
        meal_times = np.array([7.0, 9.5, 12.0, 15.0, 18.0, 21.5]) * 60.0
        time_std = np.array([60.0, 30.0, 60.0, 30.0, 60.0, 30.0])
        if self.patient_name.startswith("child#"):
            amounts = np.array([40.0, 10.0, 55.0, 10.0, 55.0, 10.0])
        else:
            amounts = np.array([50.0, 10.0, 75.0, 10.0, 75.0, 10.0])
        amount_std = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
        probabilities = np.ones(6)
        return self._sample_meals(meal_times, time_std, amounts, amount_std, probabilities)

    def _sample_meals(
        self,
        meal_times: np.ndarray,
        time_std: np.ndarray,
        amounts: np.ndarray,
        amount_std: np.ndarray,
        probabilities: np.ndarray,
    ) -> list[tuple[int, float]]:
        meals: list[tuple[int, float]] = []
        for idx in range(len(meal_times)):
            if self.random_gen.rand() > probabilities[idx]:
                continue
            lower = max(0.0, float(meal_times[idx] - 3.0 * time_std[idx]))
            upper = min(24.0 * 60.0 - 1.0, float(meal_times[idx] + 3.0 * time_std[idx]))
            time_min = _sample_clipped_normal(
                self.random_gen, meal_times[idx], time_std[idx], lower, upper
            )
            amount = float(max(0.0, self.random_gen.normal(amounts[idx], amount_std[idx])))
            meals.append((int(round(time_min)), amount))
        return aggregate_same_minute_meals(meals)

    def get_action(self, t):
        minute = t.hour * 60 + t.minute
        day = t.date()
        if day not in self._meals_by_day:
            self._meals_by_day[day] = self._generate_daily_meals()
        for meal_time, amount in self._meals_by_day[day]:
            if minute == meal_time:
                return PatientAction(meal=amount, insulin=0.0)
        return PatientAction(meal=0.0, insulin=0.0)

    def reset(self) -> None:
        self.meals = self._generate_daily_meals()
        self._meals_by_day = {self.start_time.date(): self.meals}


class ScenarioA(PaperTrainingScenario):
    """Evaluation scenario with stochastic regular meals and snacks."""

    def _generate_daily_meals(self) -> list[tuple[int, float]]:
        meal_times = np.array([8.0, 9.5, 13.0, 15.0, 19.0, 21.5]) * 60.0
        time_std = np.array([60.0, 60.0, 60.0, 60.0, 60.0, 60.0])
        amounts = np.array([50.0, 10.0, 75.0, 30.0, 75.0, 20.0])
        amount_std = amounts * 0.40
        probabilities = np.array([0.75, 0.30, 0.75, 0.30, 0.75, 0.30])
        return self._sample_meals(meal_times, time_std, amounts, amount_std, probabilities)


class ScenarioATransformed(PaperTrainingScenario):
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

    def _generate_daily_meals(self) -> list[tuple[int, float]]:
        meal_times = np.array([8.0, 9.5, 13.0, 15.0, 19.0, 21.5]) * 60.0
        meal_times = np.clip(meal_times + self.time_shift_min, 0.0, 24.0 * 60.0 - 1.0)
        time_std = np.array([60.0, 60.0, 60.0, 60.0, 60.0, 60.0])
        amounts = np.array([50.0, 10.0, 75.0, 30.0, 75.0, 20.0]) * self.carb_scale
        amount_std = amounts * 0.40
        probabilities = np.array([0.75, 0.30, 0.75, 0.30, 0.75, 0.30])
        return self._sample_meals(meal_times, time_std, amounts, amount_std, probabilities)


class NoMealScenario(Scenario):
    def get_action(self, t):
        return PatientAction(meal=0.0, insulin=0.0)

    def reset(self) -> None:
        pass


def make_paper_scenario(
    name: str,
    start_time: datetime,
    seed: int | None = None,
    patient_name: str | None = None,
):
    if name == "train":
        return PaperTrainingScenario(start_time, seed=seed, patient_name=patient_name)
    if name == "A":
        return ScenarioA(start_time, seed=seed, patient_name=patient_name)
    if name == "A_shift_early":
        return ScenarioATransformed(
            start_time, seed=seed, patient_name=patient_name, time_shift_min=-120.0
        )
    if name == "A_shift_late":
        return ScenarioATransformed(
            start_time, seed=seed, patient_name=patient_name, time_shift_min=120.0
        )
    if name == "A_carb_low":
        return ScenarioATransformed(
            start_time, seed=seed, patient_name=patient_name, carb_scale=0.70
        )
    if name == "A_carb_high":
        return ScenarioATransformed(
            start_time, seed=seed, patient_name=patient_name, carb_scale=1.30
        )
    if name == "E":
        return NoMealScenario(start_time)
    raise ValueError(f"unsupported scenario: {name}")
