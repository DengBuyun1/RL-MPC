import argparse
import json
import sys
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIM_ROOT = PROJECT_ROOT / "src" / "envs" / "simglucose_mpc"
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj


def adult_patients():
    return [f"adult#{i:03d}" for i in range(1, 11)]


def build_scenario(start_time):
    return CustomScenario(
        start_time,
        [
            (timedelta(hours=1), 50),
            (timedelta(hours=5), 75),
            (timedelta(hours=12), 75),
        ],
    )


def basal_u_per_min(patient_name: str) -> float:
    patient = T1DPatient.withName(patient_name)
    params = patient._params
    return float(params.u2ss) * float(params.BW) / 6000.0


def meal_filter_alpha(sample_time_min: float, tau_min: float) -> float:
    dt = float(sample_time_min)
    tau = max(float(tau_min), dt)
    return float(np.exp(-dt / tau))


def simulate_patient_trace(patient_name: str, seed: int, hours: float, bb_target: float):
    start_time = datetime(2026, 1, 1, 7, 0)
    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName("GuardianRT", seed=seed)
    pump = InsulinPump.withName("Insulet")
    scenario = build_scenario(start_time)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    controller = BBController(target=bb_target)
    sim = SimObj(env, controller, timedelta(hours=hours), animate=False)
    sim.simulate()
    return sim.results().reset_index(drop=True)


def build_training_frame(
    df: pd.DataFrame,
    patient_name: str,
    sample_time_min: float,
    meal_filter_tau_min: float,
    meal_release_g_per_min: float,
):
    basal = basal_u_per_min(patient_name)
    basal = max(basal, 1e-8)
    alpha = meal_filter_alpha(sample_time_min, meal_filter_tau_min)

    pending_g = 0.0
    meal_effect = 0.0
    uhat = []
    meal_input = []
    meal_model = []

    for _, row in df.iterrows():
        total_u_rate = float(row["insulin"])
        uhat.append(float(total_u_rate / basal - 1.0))

        meal_raw = float(row["CHO"])
        meal_g = meal_raw * float(sample_time_min)
        pending_g += meal_g
        release_per_step = float(meal_release_g_per_min) * float(sample_time_min)
        release = min(pending_g, release_per_step)
        pending_g = max(0.0, pending_g - release)
        meal_input.append(float(release))

        meal_effect = alpha * meal_effect + (1.0 - alpha) * float(release)
        meal_model.append(float(meal_effect))

    return pd.DataFrame(
        {
            "cgm": df["CGM"].astype(float).to_numpy(),
            "uhat": np.asarray(uhat, dtype=float),
            "meal_raw": df["CHO"].astype(float).to_numpy(),
            "meal_input": np.asarray(meal_input, dtype=float),
            "meal_model": np.asarray(meal_model, dtype=float),
        }
    )


def fit_ridge_theta(
    traces,
    order: int,
    u_delay_steps: int,
    m_delay_steps: int,
    ridge_alpha: float,
):
    X_rows = []
    y_rows = []
    hist_len = order + max(u_delay_steps, m_delay_steps) + 1

    for trace in traces:
        g_hist = deque(maxlen=hist_len)
        u_hist = deque(maxlen=hist_len)
        m_hist = deque(maxlen=hist_len)

        g = trace["cgm"].to_numpy(dtype=float)
        u = trace["uhat"].to_numpy(dtype=float)
        m = trace["meal_model"].to_numpy(dtype=float)

        for i in range(len(trace)):
            if len(g_hist) < hist_len:
                g_hist.append(float(g[i]))
                u_hist.append(float(u[i]))
                m_hist.append(float(m[i]))
                continue

            phi = []
            for j in range(order):
                phi.append(float(g_hist[-(j + 1)]))
            for j in range(order):
                phi.append(float(u_hist[-(u_delay_steps + j + 1)]))
            for j in range(order):
                phi.append(float(m_hist[-(m_delay_steps + j + 1)]))
            phi.append(1.0)

            X_rows.append(phi)
            y_rows.append(float(g[i]))

            g_hist.append(float(g[i]))
            u_hist.append(float(u[i]))
            m_hist.append(float(m[i]))

    X = np.asarray(X_rows, dtype=float)
    y = np.asarray(y_rows, dtype=float)
    reg = float(ridge_alpha) * np.eye(X.shape[1], dtype=float)
    theta = np.linalg.solve(X.T @ X + reg, X.T @ y)
    return theta


def main():
    parser = argparse.ArgumentParser(description="Train adult-only ARX theta initializations.")
    parser.add_argument("--orders", nargs="+", type=int, required=True)
    parser.add_argument("--u-delay-steps", type=int, default=3)
    parser.add_argument("--m-delay-steps", type=int, default=1)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--hours", type=float, default=18.0)
    parser.add_argument("--bb-target", type=float, default=140.0)
    parser.add_argument("--sample-time-min", type=float, default=5.0)
    parser.add_argument("--meal-filter-tau-min", type=float, default=30.0)
    parser.add_argument("--meal-release-g-per-min", type=float, default=3.0)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[101, 211])
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "arx_rls_training_adult_custom"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_traces = {patient: [] for patient in adult_patients()}
    for patient_name in adult_patients():
        for seed in args.train_seeds:
            df = simulate_patient_trace(
                patient_name=patient_name,
                seed=seed,
                hours=args.hours,
                bb_target=args.bb_target,
            )
            trace = build_training_frame(
                df=df,
                patient_name=patient_name,
                sample_time_min=args.sample_time_min,
                meal_filter_tau_min=args.meal_filter_tau_min,
                meal_release_g_per_min=args.meal_release_g_per_min,
            )
            all_traces[patient_name].append(trace)

    for order in args.orders:
        result = {
            "__meta__": {
                "training_controller": "BBController",
                "group": "adult_only",
                "glucose_column": "cgm",
                "hours_per_trace": float(args.hours),
                "train_seeds": list(args.train_seeds),
                "bb_target_glucose": float(args.bb_target),
                "sample_time_min": float(args.sample_time_min),
                "theta_input_mode": "uhat",
                "use_meal_filter": True,
                "use_meal_buffer": True,
                "meal_filter_tau_min": float(args.meal_filter_tau_min),
                "meal_release_g_per_min": float(args.meal_release_g_per_min),
                "order": int(order),
                "u_delay_steps": int(args.u_delay_steps),
                "m_delay_steps": int(args.m_delay_steps),
                "ridge_alpha": float(args.ridge_alpha),
            }
        }
        for patient_name in adult_patients():
            theta = fit_ridge_theta(
                traces=all_traces[patient_name],
                order=int(order),
                u_delay_steps=int(args.u_delay_steps),
                m_delay_steps=int(args.m_delay_steps),
                ridge_alpha=float(args.ridge_alpha),
            )
            result[patient_name] = {
                "group": "adult",
                "order": int(order),
                "theta": [float(x) for x in theta.tolist()],
            }

        out_path = output_dir / f"adult_best_theta_order{int(order)}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved: {out_path}")


if __name__ == "__main__":
    main()
