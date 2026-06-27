from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

for candidate in [
    Path.home() / "AppData/Local/Programs/Python/Python39/lib/site-packages",
]:
    if (candidate / "dateutil").exists() and str(candidate) not in sys.path:
        sys.path.append(str(candidate))

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
MPC_ROOT = ROOT / "src" / "envs" / "simglucose_mpc"
RL_ROOT = ROOT / "src" / "controllers" / "rl_sac_mask"
DEFAULT_MODEL = (
    ROOT
    / "outputs"
    / "runs"
    / "hycpap_demo_adult-002_3meal_20260513_064057"
    / "best_model.pth"
)
DEFAULT_DRL_MODEL = ROOT / "outputs" / "models" / "trainable_ensemble_adult" / "best_model.pth"

# Keep MPC's local simglucose first so every controller uses the same simulator.
for path in [str(MPC_ROOT), str(RL_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0 if path == str(MPC_ROOT) else 1, path)

from agents.sac_baseline import SACBaselineAgent  # noqa: E402
from agents.ensemble_agent import EnsembleAgent  # noqa: E402
from scipy.stats import gamma  # noqa: E402
from simglucose.actuator.pump import InsulinPump  # noqa: E402
from simglucose.controller.base import Action  # noqa: E402
from simglucose.controller.mpc_ctrller import MPCController  # noqa: E402
from simglucose.patient.t1dpatient import T1DPatient  # noqa: E402
from simglucose.sensor.cgm import CGMSensor  # noqa: E402
from simglucose.simulation.env import T1DSimEnv  # noqa: E402
from simglucose.simulation.scenario import CustomScenario  # noqa: E402
from utils.safety2_closed_loop import SafetyLayer  # noqa: E402
from utils.state_management_closed_loop_ensemble import StateRewardManager  # noqa: E402


def adult_patients() -> list[str]:
    return [f"adult#{idx:03d}" for idx in range(1, 11)]


def parse_patients(value: str, cohort: str | None) -> list[str]:
    if value:
        if value.lower() == "adult":
            return adult_patients()
        return [token.strip() for token in value.split(",") if token.strip()]
    if cohort == "adult":
        return adult_patients()
    return ["adult#002"]


def parse_clock_list(value: str) -> list[float]:
    parsed = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            hour, minute = token.split(":", 1)
            parsed.append(float(hour) + float(minute) / 60.0)
        else:
            parsed.append(float(token))
    if not parsed:
        raise ValueError("meal times cannot be empty")
    return parsed


def parse_float_list(value: str) -> list[float]:
    parsed = [float(token.strip()) for token in str(value).split(",") if token.strip()]
    if not parsed:
        raise ValueError("meal amounts cannot be empty")
    return parsed


def clock_hour_to_datetime(start_time: datetime, hour: float, day: int = 0) -> datetime:
    hh = int(hour)
    mm = int(round((hour - hh) * 60.0))
    date = start_time.date() + timedelta(days=day)
    candidate = datetime.combine(date, datetime.min.time()) + timedelta(hours=hh, minutes=mm)
    if candidate < start_time:
        candidate += timedelta(days=1)
    return candidate


def fixed_meal_scenario(
    start_time: datetime, meal_times: str, meal_amounts: str, days: int
) -> CustomScenario:
    times = parse_clock_list(meal_times)
    amounts = parse_float_list(meal_amounts)
    if len(times) != len(amounts):
        raise ValueError("meal times and amounts must have the same length")

    events = []
    for day in range(days):
        for hour, amount in zip(times, amounts):
            events.append((clock_hour_to_datetime(start_time, hour, day=day), amount))
    return CustomScenario(start_time, events)


def scenario_a(start_time: datetime, days: int, seed: int, child: bool = False) -> CustomScenario:
    rng = np.random.default_rng(seed)
    regular_times = [8.0, 13.0, 19.0]
    regular_amounts = [40.0, 55.0, 55.0] if child else [50.0, 75.0, 75.0]
    snack_times = [9.5, 15.0, 21.5]
    snack_amounts = [10.0, 30.0, 20.0]

    events = []
    for day in range(days):
        for hour, amount in zip(regular_times, regular_amounts):
            if rng.random() <= 0.75:
                events.append(random_meal_event(start_time, day, hour, amount, rng, time_std=60.0))
        for hour, amount in zip(snack_times, snack_amounts):
            if rng.random() <= 0.30:
                events.append(random_meal_event(start_time, day, hour, amount, rng, time_std=60.0))
    events.sort(key=lambda item: item[0])
    return CustomScenario(start_time, events)


def random_meal_event(
    start_time: datetime,
    day: int,
    hour: float,
    grams: float,
    rng: np.random.Generator,
    time_std: float,
) -> tuple[datetime, float]:
    minute = int(round(rng.normal(hour * 60.0, time_std)))
    minute = int(np.clip(minute, 0, 24 * 60 - 1))
    amount = max(float(rng.normal(grams, 0.40 * grams)), 0.0)
    time = datetime.combine(start_time.date() + timedelta(days=day), datetime.min.time())
    return time + timedelta(minutes=minute), amount


def make_scenario(args, patient_name: str, run_seed: int) -> tuple[CustomScenario, timedelta]:
    start_time = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M")
    if args.scenario == "scenario-a":
        days = args.days if args.days is not None else 2
        return scenario_a(start_time, days, run_seed, child=patient_name.startswith("child#")), timedelta(days=days)
    days = args.days if args.days is not None else max(1, int(np.ceil(args.hours / 24.0)))
    return (
        fixed_meal_scenario(start_time, args.meal_times, args.meal_amounts, days),
        timedelta(hours=args.hours),
    )


def patient_basal_rate(patient: T1DPatient) -> float:
    params = patient._params
    return float(params["u2ss"]) * float(params["BW"]) / 6000.0


def pkpd_tail_factors(t_peak: float = 55, t_end: float = 480, n_steps: int = 160) -> np.ndarray:
    shape_k = 2
    scale_theta = t_peak / (shape_k - 1)
    time_points = np.linspace(0, t_end, n_steps)
    return 1.0 - gamma.cdf(time_points, a=shape_k, scale=scale_theta)


class BasicSACState:
    def __init__(self, body_weight: float, basal_rate: float, sample_time: float = 5.0):
        self.body_weight = float(body_weight)
        self.basal_rate = float(max(basal_rate, 1e-8))
        self.sample_time = float(sample_time)
        self.glucose_history = deque(maxlen=2)
        self.insulin_history = deque(maxlen=160)
        self.iob_tail = pkpd_tail_factors()

    def reset(self, glucose: float) -> None:
        self.glucose_history.clear()
        self.glucose_history.append(float(glucose))
        self.glucose_history.append(float(glucose))
        self.insulin_history.clear()
        for _ in range(160):
            self.insulin_history.append(0.0)

    def record_action(self, basal_rate: float) -> None:
        self.insulin_history.append(float(basal_rate))

    def current_iob(self) -> float:
        history = np.array(list(self.insulin_history)[::-1], dtype=np.float32)
        return float(np.sum(history * self.iob_tail))

    def make_state(self, glucose: float) -> np.ndarray:
        glucose = float(glucose)
        previous = self.glucose_history[-1]
        self.glucose_history.append(glucose)
        rate = (glucose - previous) / self.sample_time
        iob = self.current_iob()
        return np.array(
            [
                glucose / 400.0,
                rate / 5.0,
                iob / (10.0 * self.basal_rate),
                self.body_weight / 100.0,
            ],
            dtype=np.float32,
        )


def scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def load_sac_agents(model_paths: list[Path], device: torch.device) -> list[SACBaselineAgent]:
    agents = []
    for model_path in model_paths:
        if not model_path.exists():
            raise FileNotFoundError(f"missing SAC checkpoint: {model_path}")
        agent = SACBaselineAgent(state_dim=4, action_dim=1, max_action=1.0, device=device)
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)
        agent.actor.load_state_dict(checkpoint["actor"])
        if "critic" in checkpoint:
            agent.critic.load_state_dict(checkpoint["critic"])
        agent.actor.eval()
        agent.critic.eval()
        agents.append(agent)
    return agents


def load_ensemble_agent(model_path: Path, device: torch.device) -> EnsembleAgent:
    if not model_path.exists():
        raise FileNotFoundError(f"missing DRL ensemble checkpoint: {model_path}")
    agent = EnsembleAgent(state_dim=4, action_dim=1, max_action=1.0, device=device)
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    agent.sac_agent.actor.load_state_dict(checkpoint["sac_actor"])
    agent.sac_agent.critic.load_state_dict(checkpoint["sac_critic"])
    agent.td3_agent.actor.load_state_dict(checkpoint["td3_actor"])
    agent.td3_agent.critic.load_state_dict(checkpoint["td3_critic"])
    agent.meta_controller.load_state_dict(checkpoint["meta_controller"])
    agent.sac_agent.actor.eval()
    agent.td3_agent.actor.eval()
    agent.meta_controller.eval()
    return agent


def map_raw_distribution_to_basal(
    action_mean, action_var, basal_rate: float, max_basal: float, multiplier: float
) -> tuple[np.ndarray, np.ndarray]:
    scale = 0.5 * multiplier * basal_rate
    basal_mean = ((np.asarray(action_mean) + 1.0) * scale).clip(0.0, max_basal)
    basal_var = np.asarray(action_var) * (scale**2)
    return basal_mean.astype(np.float32), basal_var.astype(np.float32)


def sac_ensemble_distribution(
    agents: list[SACBaselineAgent],
    state: np.ndarray,
    basal_rate: float,
    max_basal: float,
    max_basal_multiplier: float,
    uncertainty_scale: float,
) -> dict[str, float]:
    means = []
    variances = []
    raw_means = []
    raw_variances = []
    for agent in agents:
        dist = agent.action_distribution(state)
        raw_mean = scalar(dist["action_mean"])
        raw_var = max(scalar(dist["action_var"]), 1e-10)
        basal_mean, basal_var = map_raw_distribution_to_basal(
            raw_mean, raw_var, basal_rate, max_basal, max_basal_multiplier
        )
        means.append(scalar(basal_mean))
        variances.append(max(scalar(basal_var), 1e-10))
        raw_means.append(raw_mean)
        raw_variances.append(raw_var)

    means = np.asarray(means, dtype=np.float64)
    variances = np.asarray(variances, dtype=np.float64)
    raw_means = np.asarray(raw_means, dtype=np.float64)
    raw_variances = np.asarray(raw_variances, dtype=np.float64)

    mean = float(np.mean(means))
    variance = float(np.mean(variances + means**2) - mean**2)
    raw_mean = float(np.mean(raw_means))
    raw_variance = float(np.mean(raw_variances + raw_means**2) - raw_mean**2)

    variance = max(variance, 1e-10) * uncertainty_scale
    raw_variance = max(raw_variance, 1e-10) * uncertainty_scale
    return {
        "mean": float(np.clip(mean, 0.0, max_basal)),
        "var": variance,
        "std": float(np.sqrt(variance)),
        "raw_mean": raw_mean,
        "raw_var": raw_variance,
        "raw_std": float(np.sqrt(raw_variance)),
    }


def gaussian_product(policy_a: dict[str, float], policy_b: dict[str, float]) -> dict[str, float]:
    var_a = max(policy_a["var"], 1e-10)
    var_b = max(policy_b["var"], 1e-10)
    mean = (policy_a["mean"] * var_b + policy_b["mean"] * var_a) / (var_a + var_b)
    var = (var_a * var_b) / (var_a + var_b)
    return {"mean": float(mean), "var": float(var), "std": float(np.sqrt(var))}


class DrlController:
    def __init__(
        self,
        agents: list[SACBaselineAgent],
        patient_name: str,
        max_basal: float,
        max_basal_multiplier: float,
        uncertainty_scale: float,
    ):
        self.agents = agents
        self.patient = T1DPatient.withName(patient_name)
        self.basal_rate = patient_basal_rate(self.patient)
        self.body_weight = float(self.patient._params["BW"])
        self.max_basal = float(max_basal)
        self.max_basal_multiplier = float(max_basal_multiplier)
        self.uncertainty_scale = float(uncertainty_scale)
        self.tracker = BasicSACState(self.body_weight, self.basal_rate)
        self.initialized = False
        self.policy_rows = []

    def reset(self) -> None:
        self.initialized = False
        self.policy_rows = []

    def policy(self, observation, reward, done, **info):
        glucose = float(getattr(observation, "CGM", getattr(observation, "BG", np.nan)))
        sample_time = float(info.get("sample_time", 5.0))
        if not self.initialized:
            self.tracker = BasicSACState(self.body_weight, self.basal_rate, sample_time=sample_time)
            self.tracker.reset(glucose)
            self.initialized = True
        state = self.tracker.make_state(glucose)
        drl_policy = sac_ensemble_distribution(
            self.agents,
            state,
            self.basal_rate,
            self.max_basal,
            self.max_basal_multiplier,
            self.uncertainty_scale,
        )
        basal = float(np.clip(drl_policy["mean"], 0.0, self.max_basal))
        self.tracker.record_action(basal)
        self.policy_rows.append(
            {
                "time": info.get("time"),
                "drl_mean": drl_policy["mean"],
                "drl_std": drl_policy["std"],
                "selected_basal": basal,
            }
        )
        return Action(basal=basal, bolus=0.0)


class EnsembleDrlController:
    """Aggressive existing DRL baseline from the local trainable SAC-TD3 ensemble."""

    def __init__(self, agent: EnsembleAgent, patient_name: str, cohort: str = "adult"):
        self.agent = agent
        self.patient = T1DPatient.withName(patient_name)
        self.body_weight = float(self.patient._params["BW"])
        self.cohort = cohort
        self.manager = StateRewardManager(state_dim=4)
        self.safety_layer = SafetyLayer(cohort=cohort)
        self.policy_rows = []

    def reset(self) -> None:
        self.manager = StateRewardManager(state_dim=4)
        self.policy_rows = []

    def clinical_max(self) -> float:
        if self.cohort == "child":
            return 0.25
        if self.cohort == "adolescent":
            return 0.75
        return 1.5

    def policy(self, observation, reward, done, **info):
        glucose = float(getattr(observation, "CGM", getattr(observation, "BG", np.nan)))
        unnormalized_state = self.manager.get_full_state(glucose, self.body_weight)
        normalized_state = self.manager.get_normalized_state(unnormalized_state)
        action, w_sac, w_td3 = self.agent.select_action(normalized_state, evaluate=True)
        normalized_action = (float(action[0]) + 1.0) / 2.0
        if self.cohort == "adult":
            insulin = np.array([normalized_action * self.clinical_max()])
        else:
            insulin = np.array([(normalized_action**2) * self.clinical_max()])
        safe_action = self.safety_layer.apply(insulin, unnormalized_state)
        basal = float(np.asarray(safe_action).reshape(-1)[0])
        self.manager.insulin_history.append(basal)
        self.policy_rows.append(
            {
                "time": info.get("time"),
                "raw_action": float(action[0]),
                "normalized_action": normalized_action,
                "w_sac": float(w_sac),
                "w_td3": float(w_td3),
                "selected_basal": basal,
            }
        )
        return Action(basal=basal, bolus=0.0)


class HybridController:
    def __init__(
        self,
        agents: list[SACBaselineAgent],
        patient_name: str,
        max_basal: float,
        max_basal_multiplier: float,
        drl_uncertainty_scale: float,
        mpc_sigma_factor: float,
        mpc_sigma_floor: float,
        mpc_kwargs: dict,
    ):
        self.drl = DrlController(
            agents,
            patient_name,
            max_basal=max_basal,
            max_basal_multiplier=max_basal_multiplier,
            uncertainty_scale=drl_uncertainty_scale,
        )
        self.mpc = MPCController(**mpc_kwargs)
        self.max_basal = float(max_basal)
        self.mpc_sigma_factor = float(mpc_sigma_factor)
        self.mpc_sigma_floor = float(mpc_sigma_floor)
        self.policy_rows = []

    def reset(self) -> None:
        self.drl.reset()
        self.mpc.reset()
        self.policy_rows = []

    def policy(self, observation, reward, done, **info):
        glucose = float(getattr(observation, "CGM", getattr(observation, "BG", np.nan)))
        sample_time = float(info.get("sample_time", 5.0))

        if not self.drl.initialized:
            self.drl.tracker = BasicSACState(
                self.drl.body_weight, self.drl.basal_rate, sample_time=sample_time
            )
            self.drl.tracker.reset(glucose)
            self.drl.initialized = True
        state = self.drl.tracker.make_state(glucose)
        drl_policy = sac_ensemble_distribution(
            self.drl.agents,
            state,
            self.drl.basal_rate,
            self.drl.max_basal,
            self.drl.max_basal_multiplier,
            self.drl.uncertainty_scale,
        )

        mpc_action = self.mpc.policy(observation, reward, done, **info)
        mpc_mean = float(np.clip(mpc_action.basal + mpc_action.bolus, 0.0, self.max_basal))
        mpc_std = max(self.mpc_sigma_floor, self.mpc_sigma_factor * self.drl.basal_rate)
        hybrid_policy = gaussian_product(drl_policy, {"mean": mpc_mean, "var": mpc_std**2})
        basal = float(np.clip(hybrid_policy["mean"], 0.0, self.max_basal))

        # The MPC observer should see the insulin actually delivered by HyCPAP.
        basal_units = self.mpc.profile.basal_rate * sample_time
        self.mpc.last_u_dev = basal * sample_time - basal_units
        self.drl.tracker.record_action(basal)
        self.policy_rows.append(
            {
                "time": info.get("time"),
                "drl_mean": drl_policy["mean"],
                "drl_std": drl_policy["std"],
                "mpc_mean": mpc_mean,
                "mpc_std": mpc_std,
                "hybrid_mean": hybrid_policy["mean"],
                "hybrid_std": hybrid_policy["std"],
                "selected_basal": basal,
            }
        )
        return Action(basal=basal, bolus=0.0)


def mpc_kwargs_from_args(args) -> dict:
    return {
        "variant": "adaptive",
        "announce_meals": args.announce_meals,
        "use_iob_constraint": args.use_iob_constraint,
        "max_insulin_units_per_sample": args.max_insulin_units_per_sample,
        "max_insulin_tdi_fraction": args.max_insulin_tdi_fraction,
        "model_gain_factor": args.model_gain_factor,
        "r_plus_scale": args.r_plus_scale,
        "meal_bolus_scale": args.meal_bolus_scale,
        "enable_low_glucose_suspend": args.enable_low_glucose_suspend,
        "suspend_glucose": args.suspend_glucose,
        "predictive_suspend_glucose": args.predictive_suspend_glucose,
        "predictive_suspend_velocity": args.predictive_suspend_velocity,
    }


def make_env(patient_name: str, args, run_seed: int) -> tuple[T1DSimEnv, timedelta]:
    patient = T1DPatient.withName(
        patient_name, random_init_bg=args.random_init_bg, seed=run_seed + 100_000
    )
    sensor = CGMSensor.withName(args.sensor, seed=run_seed)
    pump = InsulinPump.withName(args.pump)
    scenario, sim_time = make_scenario(args, patient_name, run_seed)
    return T1DSimEnv(patient, sensor, pump, scenario), sim_time


def run_controller(controller, patient_name: str, controller_name: str, repeat: int, args, run_seed: int):
    env, sim_time = make_env(patient_name, args, run_seed)
    controller.reset()
    step = env.reset()
    observation, reward, done, info = step.observation, step.reward, step.done, step.info
    while env.time < env.scenario.start_time + sim_time:
        action = controller.policy(observation, reward, done, **info)
        step = env.step(action)
        observation, reward, done, info = step.observation, step.reward, step.done, step.info
        if done:
            break

    df = env.show_history().reset_index()
    df.insert(0, "controller", controller_name)
    df.insert(1, "patient", patient_name)
    df.insert(2, "repeat", repeat)
    return df, summarize(df, controller_name, patient_name, repeat, getattr(controller, "mpc", controller))


def summarize(df: pd.DataFrame, controller_name: str, patient_name: str, repeat: int, controller) -> dict:
    bg = pd.to_numeric(df["BG"], errors="coerce").dropna()
    cgm = pd.to_numeric(df["CGM"], errors="coerce").dropna()
    insulin = pd.to_numeric(df["insulin"], errors="coerce").dropna()
    cho = pd.to_numeric(df["CHO"], errors="coerce").fillna(0.0)
    solve_times = np.asarray(getattr(controller, "solve_times", []), dtype=float)
    solve_successes = np.asarray(getattr(controller, "solve_successes", []), dtype=bool)

    return {
        "controller": controller_name,
        "patient": patient_name,
        "repeat": repeat,
        "steps": int(len(bg)),
        "TBR2_lt54": float((bg < 54).mean() * 100.0),
        "TBR1_lt70": float((bg < 70).mean() * 100.0),
        "TITR_70_140": float(((bg >= 70) & (bg <= 140)).mean() * 100.0),
        "TIR_70_180": float(((bg >= 70) & (bg <= 180)).mean() * 100.0),
        "TAR1_gt180": float((bg > 180).mean() * 100.0),
        "TAR2_gt250": float((bg > 250).mean() * 100.0),
        "mean_bg": float(bg.mean()),
        "sd_bg": float(bg.std(ddof=0)),
        "mean_cgm": float(cgm.mean()),
        "total_insulin_u": float(insulin.sum() * 5.0),
        "total_carbs_g": float(cho.sum() * 5.0),
        "mean_solve_ms": float(solve_times.mean() * 1000.0) if len(solve_times) else np.nan,
        "p95_solve_ms": float(np.percentile(solve_times, 95) * 1000.0)
        if len(solve_times)
        else np.nan,
        "solve_success_rate": float(solve_successes.mean() * 100.0)
        if len(solve_successes)
        else np.nan,
    }


def make_controller(
    controller_name: str,
    patient_name: str,
    args,
    sac_agents: list[SACBaselineAgent],
    ensemble_agent: EnsembleAgent | None,
):
    pump = InsulinPump.withName(args.pump)
    max_basal = float(pump._params["max_basal"])
    if controller_name == "drl":
        if ensemble_agent is None:
            raise ValueError("drl controller requires --drl-model-path")
        cohort = "adult" if patient_name.startswith("adult#") else args.cohort or "adult"
        return EnsembleDrlController(ensemble_agent, patient_name, cohort=cohort)
    if controller_name == "basic_sac":
        return DrlController(
            sac_agents,
            patient_name,
            max_basal=max_basal,
            max_basal_multiplier=args.max_basal_multiplier,
            uncertainty_scale=args.drl_uncertainty_scale,
        )
    if controller_name == "zone_mpc":
        return MPCController(**mpc_kwargs_from_args(args))
    if controller_name == "hybrid":
        return HybridController(
            sac_agents,
            patient_name,
            max_basal=max_basal,
            max_basal_multiplier=args.max_basal_multiplier,
            drl_uncertainty_scale=args.drl_uncertainty_scale,
            mpc_sigma_factor=args.mpc_prior_sigma_factor,
            mpc_sigma_floor=args.mpc_prior_sigma_floor,
            mpc_kwargs=mpc_kwargs_from_args(args),
        )
    raise ValueError(f"unknown controller: {controller_name}")


def save_policy_rows(controller, output_dir: Path, controller_name: str, patient: str, repeat: int) -> None:
    rows = getattr(controller, "policy_rows", [])
    if not rows:
        return
    path = output_dir / controller_name / f"repeat{repeat:02d}_{patient}_policy.csv"
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_comparison_plot(all_dfs: list[pd.DataFrame], output_dir: Path, patient: str, repeat: int) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
    colors = {"drl": "#2f5597", "zone_mpc": "#6f9e99", "hybrid": "#c9b64b"}
    labels = {"drl": "DRL", "zone_mpc": "Zone-MPC", "hybrid": "Hybrid"}

    start_time = pd.to_datetime(all_dfs[0]["Time"]).min()
    for df in all_dfs:
        controller = df["controller"].iloc[0]
        time_h = (pd.to_datetime(df["Time"]) - start_time).dt.total_seconds() / 3600.0
        axes[0].plot(time_h, df["CGM"], color=colors[controller], linewidth=1.8, label=labels[controller])
        axes[1].plot(time_h, df["BG"], color=colors[controller], linewidth=1.8, label=labels[controller])
        axes[2].plot(time_h, df["insulin"], color=colors[controller], linewidth=1.5, label=labels[controller])

    for ax in axes[:2]:
        ax.axhline(70, color="#3c9d51", linestyle="--", linewidth=1.0, label="Hypoglycemia")
        ax.axhline(180, color="#d95f5f", linestyle="--", linewidth=1.0, label="Hyperglycemia")
        ax.set_ylim(40, 400)
        ax.grid(alpha=0.22)

    first = all_dfs[0]
    time_h = (pd.to_datetime(first["Time"]) - start_time).dt.total_seconds() / 3600.0
    meals = pd.to_numeric(first["CHO"], errors="coerce").fillna(0.0)
    meal_mask = meals > 0
    axes[2].vlines(time_h[meal_mask], ymin=0, ymax=max(0.25, first["insulin"].max()), color="#7a4aa0", alpha=0.45)
    axes[2].grid(alpha=0.22)
    axes[0].set_ylabel("CGM (mg/dL)")
    axes[1].set_ylabel("BG (mg/dL)")
    axes[2].set_ylabel("Insulin (U/min)")
    axes[2].set_xlabel("Time since start (h)")
    axes[0].legend(loc="upper left", ncol=5, fontsize=9)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    path = plot_dir / f"repeat{repeat:02d}_{patient}_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_outputs(output_dir: Path, summaries: list[dict], all_rows: list[pd.DataFrame], config: dict) -> None:
    summary = pd.DataFrame(summaries)
    summary.to_csv(output_dir / "summary.csv", index=False)
    aggregate = (
        summary.groupby("controller")
        .agg(
            runs=("patient", "count"),
            TBR2_lt54=("TBR2_lt54", "mean"),
            TBR1_lt70=("TBR1_lt70", "mean"),
            TITR_70_140=("TITR_70_140", "mean"),
            TIR_70_180=("TIR_70_180", "mean"),
            TAR1_gt180=("TAR1_gt180", "mean"),
            TAR2_gt250=("TAR2_gt250", "mean"),
            mean_bg=("mean_bg", "mean"),
            sd_bg=("sd_bg", "mean"),
            total_insulin_u=("total_insulin_u", "mean"),
            total_carbs_g=("total_carbs_g", "mean"),
            p95_solve_ms=("p95_solve_ms", "mean"),
            solve_success_rate=("solve_success_rate", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)
    pd.concat(all_rows, ignore_index=True).to_csv(output_dir / "all_rollouts.csv", index=False)
    with (output_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run DRL, adaptive Zone-MPC, and HyCPAP-style hybrid simulations."
    )
    parser.add_argument("--patients", default="", help="Comma-separated patients, or 'adult'.")
    parser.add_argument("--cohort", choices=["adult"], default=None)
    parser.add_argument(
        "--controllers",
        nargs="+",
        choices=["drl", "basic_sac", "zone_mpc", "hybrid"],
        default=["drl", "zone_mpc", "hybrid"],
    )
    parser.add_argument("--drl-model-path", default=str(DEFAULT_DRL_MODEL))
    parser.add_argument("--sac-model-paths", nargs="+", default=[str(DEFAULT_MODEL)])
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "three_controller_adult"))
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default=None)

    parser.add_argument("--scenario", choices=["fixed-three-meal", "scenario-a"], default="fixed-three-meal")
    parser.add_argument("--start-time", default="2026-01-01 07:00")
    parser.add_argument("--hours", type=float, default=24.0)
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--meal-times", default="08:00,12:00,19:00")
    parser.add_argument("--meal-amounts", default="50,75,75")
    parser.add_argument("--sensor", default="GuardianRT")
    parser.add_argument("--pump", default="Insulet")
    parser.add_argument("--random-init-bg", action="store_true")

    parser.add_argument("--max-basal-multiplier", type=float, default=10.0)
    parser.add_argument("--drl-uncertainty-scale", type=float, default=1.5)
    parser.add_argument("--mpc-prior-sigma-factor", type=float, default=1.0)
    parser.add_argument("--mpc-prior-sigma-floor", type=float, default=0.005)

    parser.add_argument("--announce-meals", action="store_true")
    parser.add_argument("--use-iob-constraint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-insulin-units-per-sample", type=float, default=1.0)
    parser.add_argument("--max-insulin-tdi-fraction", type=float, default=0.0075)
    parser.add_argument("--model-gain-factor", type=float, default=1.0)
    parser.add_argument("--r-plus-scale", type=float, default=0.5)
    parser.add_argument("--meal-bolus-scale", type=float, default=1.0)
    parser.add_argument("--enable-low-glucose-suspend", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--suspend-glucose", type=float, default=80.0)
    parser.add_argument("--predictive-suspend-glucose", type=float, default=100.0)
    parser.add_argument("--predictive-suspend-velocity", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for controller in args.controllers:
        (output_dir / controller).mkdir(parents=True, exist_ok=True)

    patients = parse_patients(args.patients, args.cohort)
    sac_model_paths = [
        Path(path) if Path(path).is_absolute() else (ROOT / path) for path in args.sac_model_paths
    ]
    drl_model_path = (
        Path(args.drl_model_path)
        if Path(args.drl_model_path).is_absolute()
        else (ROOT / args.drl_model_path)
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    needs_sac = any(name in {"basic_sac", "hybrid"} for name in args.controllers)
    sac_agents = load_sac_agents(sac_model_paths, device) if needs_sac else []
    ensemble_agent = load_ensemble_agent(drl_model_path, device) if "drl" in args.controllers else None

    summaries = []
    all_rows = []
    plot_groups: dict[tuple[str, int], list[pd.DataFrame]] = {}

    for repeat in range(args.repeats):
        for patient_idx, patient_name in enumerate(patients):
            run_seed = args.seed + repeat * 1000 + patient_idx
            for controller_name in args.controllers:
                controller = make_controller(
                    controller_name, patient_name, args, sac_agents, ensemble_agent
                )
                df, summary = run_controller(
                    controller, patient_name, controller_name, repeat, args, run_seed
                )
                out_path = output_dir / controller_name / f"repeat{repeat:02d}_{patient_name}.csv"
                df.to_csv(out_path, index=False)
                save_policy_rows(controller, output_dir, controller_name, patient_name, repeat)
                summaries.append(summary)
                all_rows.append(df)
                plot_groups.setdefault((patient_name, repeat), []).append(df)
                print(
                    f"{controller_name:8s} r={repeat:02d} {patient_name:10s} "
                    f"TIR={summary['TIR_70_180']:.1f}% <70={summary['TBR1_lt70']:.1f}% "
                    f"mean={summary['mean_bg']:.1f}"
                )

    for (patient_name, repeat), dfs in plot_groups.items():
        if len(dfs) >= 2:
            save_comparison_plot(dfs, output_dir, patient_name, repeat)

    config = vars(args).copy()
    config["patients_resolved"] = patients
    config["sac_model_paths_resolved"] = [str(path) for path in sac_model_paths]
    config["drl_model_path_resolved"] = str(drl_model_path)
    config["simulator"] = str(MPC_ROOT / "simglucose")
    write_outputs(output_dir, summaries, all_rows, config)
    print(f"summary: {output_dir / 'summary.csv'}")
    print(f"aggregate: {output_dir / 'aggregate.csv'}")
    print(f"rollouts: {output_dir / 'all_rollouts.csv'}")


if __name__ == "__main__":
    main()
