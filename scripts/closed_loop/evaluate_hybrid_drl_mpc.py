from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAC_ROOT = PROJECT_ROOT / "src" / "controllers" / "rl_sac_mask"
SIM_ROOT = PROJECT_ROOT / "src" / "envs" / "simglucose_mpc"
for path in (SAC_ROOT, SIM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sac_mask_drl import AgentConfig, MaskRecurrentSACAgent, MasksemblesConfig, SACConfig
from sac_mask_drl.ap_env import (
    APStateTracker,
    basal_multiplier_to_raw_action,
    clinical_metrics,
    make_paper_simglucose_env,
    map_action_to_basal,
    patient_basal_rate,
    zone_reward,
)
from sac_mask_drl.history import HistoryBuffer

from simglucose.controller.mpc_ctrller import ZoneMPCController  # noqa: E402


Observation = namedtuple("Observation", ["CGM"])


def make_agent(args, device):
    config = AgentConfig(
        observation_dim=4,
        action_dim=1,
        action_limit=1.0,
        gru_hidden_dim=args.gru_hidden_dim,
        sequence_length=args.sequence_length,
        masksembles=MasksemblesConfig(
            num_masks=args.num_masks,
            hidden_dim=args.hidden_dim,
            keep_prob=args.keep_prob,
            seed=args.seed,
        ),
        sac=SACConfig(
            gamma=args.gamma,
            alpha=args.alpha,
            log_std_min=args.log_std_min,
            log_std_max=args.log_std_max,
        ),
    )
    agent = MaskRecurrentSACAgent(config, device=device)
    agent.load(args.model_path, load_optimizers=False)
    agent.actor.eval()
    agent.critic.eval()
    return agent, config


def make_mpc(args):
    return ZoneMPCController(
        variant=args.mpc_variant,
        prediction_horizon=args.mpc_prediction_horizon,
        control_horizon=args.mpc_control_horizon,
        target_glucose=args.mpc_target_glucose,
        model_sample_time=5.0,
        announce_meals=False,
        use_iob_constraint=args.mpc_use_iob_constraint,
        max_insulin_units_per_sample=args.mpc_max_insulin_units_per_sample,
        max_insulin_tdi_fraction=args.mpc_max_insulin_tdi_fraction,
        model_gain_factor=args.mpc_model_gain_factor,
        r_plus_scale=args.mpc_r_plus_scale,
        enable_low_glucose_suspend=args.mpc_low_glucose_suspend,
        suspend_glucose=args.mpc_suspend_glucose,
        predictive_suspend_glucose=args.mpc_predictive_suspend_glucose,
        max_slsqp_iter=args.mpc_max_slsqp_iter,
    )


def gaussian_product_mean(drl_mean, drl_var, mpc_mean, mpc_var):
    drl_var = np.maximum(drl_var, 1e-8)
    mpc_var = np.maximum(mpc_var, 1e-8)
    mean = (drl_mean * mpc_var + mpc_mean * drl_var) / (drl_var + mpc_var)
    var = (drl_var * mpc_var) / (drl_var + mpc_var)
    return mean, np.maximum(var, 1e-8)


def select_hybrid_action(agent, history, mpc_basal, basal_rate, max_basal, args, device, mpc_std_raw):
    with torch.no_grad():
        dist = agent.policy_distribution(history.tensor(device))
    drl_mean = dist.mean.detach().cpu().numpy().reshape(-1)
    drl_var = dist.var.detach().cpu().numpy().reshape(-1) * args.drl_var_scale + args.drl_var_floor
    mpc_multiplier = float(np.clip(mpc_basal / max(basal_rate, 1e-8), 0.0, args.max_basal_multiplier))
    mpc_raw = basal_multiplier_to_raw_action(
        mpc_multiplier,
        args.max_basal_multiplier,
        action_mapping=args.action_mapping,
        basal_delta_multiplier=args.basal_delta_multiplier,
    )
    mpc_var = np.full_like(drl_var, float(mpc_std_raw) ** 2)

    if args.policy_mode == "drl":
        raw_action = drl_mean
        fused_var = drl_var
    elif args.policy_mode == "mpc":
        raw_action = mpc_raw
        fused_var = mpc_var
    else:
        raw_action, fused_var = gaussian_product_mean(drl_mean, drl_var, mpc_raw, mpc_var)

    raw_action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
    basal_action = map_action_to_basal(
        raw_action,
        basal_rate,
        max_basal,
        args.max_basal_multiplier,
        action_mapping=args.action_mapping,
        basal_delta_multiplier=args.basal_delta_multiplier,
    )
    debug = {
        "drl_raw_mean": float(drl_mean[0]),
        "drl_raw_var": float(drl_var[0]),
        "mpc_raw_mean": float(mpc_raw[0]),
        "mpc_raw_var": float(mpc_var[0]),
        "hybrid_raw": float(raw_action[0]),
        "hybrid_raw_var": float(fused_var[0]),
        "mpc_basal": float(mpc_basal),
    }
    return raw_action, basal_action, debug


def evaluate_one(agent, config, patient_name, scenario, seed, args, device, mpc_std_raw):
    env, patient = make_paper_simglucose_env(
        patient_name,
        scenario,
        seed,
        args.steps,
        meal_announcement=args.meal_announcement,
        sensor_name=args.sensor_name,
    )
    mpc = make_mpc(args)
    basal_rate = patient_basal_rate(patient)
    body_weight = float(patient._params["BW"])
    max_basal = float(env.action_space.high[0])
    sample_time = float(getattr(env, "sample_time", 5.0))
    tracker = APStateTracker(body_weight, basal_rate, sample_time=sample_time, state_mode=args.state_mode)
    history = HistoryBuffer(config)

    obs, info = env.reset(seed=seed)
    glucose = float(obs[0])
    state = tracker.reset(glucose)
    history.reset(state)
    glucose_trace = [glucose]
    rows = []
    total_reward = 0.0
    last_reward = 0.0
    done = False

    for step in range(args.steps):
        mpc_action = mpc.policy(
            Observation(CGM=glucose),
            last_reward,
            done,
            sample_time=sample_time,
            patient_name=patient_name,
            time=env.env.time,
            meal=0.0,
            iob=tracker.current_iob(),
        )
        raw_action, basal_action, debug = select_hybrid_action(
            agent,
            history,
            float(mpc_action.basal),
            basal_rate,
            max_basal,
            args,
            device,
            mpc_std_raw=mpc_std_raw,
        )
        next_obs, _, terminated, truncated, info = env.step(basal_action)
        bolus_rate = float(info.get("bolus_rate", 0.0)) if isinstance(info, dict) else 0.0
        meal_grams = float(info.get("meal_grams", 0.0)) if isinstance(info, dict) else 0.0
        tracker.record_action(basal_action[0], bolus_rate=bolus_rate)
        next_glucose = float(next_obs[0])
        reward = zone_reward(next_glucose)
        done = bool(terminated or truncated or next_glucose < 39.0 or next_glucose > 400.0)
        next_state = tracker.make_state(next_glucose, bolus_rate=bolus_rate, meal_grams=meal_grams)
        history.append(next_state, raw_action, reward)

        total_reward += reward
        glucose_trace.append(next_glucose)
        rows.append(
            {
                "patient": patient_name,
                "scenario": scenario,
                "seed": seed,
                "policy_mode": args.policy_mode,
                "mpc_std_raw": mpc_std_raw,
                "step": step,
                "glucose": glucose,
                "next_glucose": next_glucose,
                "reward": reward,
                "basal": float(basal_action[0]),
                "bolus": bolus_rate,
                "meal": meal_grams,
                "actual_meal_rate": float(info.get("actual_meal_rate", 0.0)) if isinstance(info, dict) else 0.0,
                "done": done,
                **debug,
            }
        )
        glucose = next_glucose
        last_reward = reward
        if done:
            break

    metrics = clinical_metrics(glucose_trace)
    metrics.update(
        {
            "patient": patient_name,
            "scenario": scenario,
            "seed": seed,
            "policy_mode": args.policy_mode,
            "mpc_std_raw": mpc_std_raw,
            "steps": len(rows),
            "total_reward": total_reward,
            "mpc_success_rate": float(np.mean(mpc.solve_successes)) if mpc.solve_successes else np.nan,
            "mpc_solve_time_mean": float(np.mean(mpc.solve_times)) if mpc.solve_times else np.nan,
        }
    )
    env.close()
    return rows, metrics


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def evaluate(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    agent, config = make_agent(args, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    std_values = parse_float_list(args.mpc_std_raw_values)
    all_rows = []
    summary = []
    for mpc_std_raw in std_values:
        for repeat in range(args.repeats):
            seed = args.seed + repeat + 1000
            rows, metrics = evaluate_one(agent, config, args.patient, args.scenario, seed, args, device, mpc_std_raw)
            all_rows.extend(rows)
            summary.append(metrics)
            print(
                f"{args.policy_mode} std={mpc_std_raw:.4f} repeat={repeat + 1}/{args.repeats} "
                f"TIR={metrics['tir']:.1f}% TITR={metrics['titr']:.1f}% TBR70={metrics['tbr70']:.1f}% "
                f"reward={metrics['total_reward']:.1f}"
            )

    if all_rows:
        with (output_dir / "rollouts.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
    if summary:
        fields = list(summary[0].keys())
        with (output_dir / "summary.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(summary)
        aggregate = []
        for mpc_std_raw, group in __import__("pandas").DataFrame(summary).groupby("mpc_std_raw"):
            row = {"mpc_std_raw": mpc_std_raw, "n": len(group)}
            for col in ["titr", "tir", "tbr70", "tbr54", "tar180", "mean_bg", "sd_bg", "total_reward", "mpc_success_rate", "mpc_solve_time_mean"]:
                row[col + "_mean"] = float(group[col].mean())
                row[col + "_std"] = float(group[col].std())
            row["score"] = row["tir_mean"] - 2.0 * row["tbr70_mean"] - 5.0 * row["tbr54_mean"]
            aggregate.append(row)
        aggregate = sorted(aggregate, key=lambda row: row["score"], reverse=True)
        with (output_dir / "aggregate_by_mpc_std.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregate[0].keys()))
            writer.writeheader()
            writer.writerows(aggregate)
        print(f"Saved summary: {output_dir / 'summary.csv'}")
        print(f"Saved aggregate: {output_dir / 'aggregate_by_mpc_std.csv'}")
        print(f"Saved rollouts: {output_dir / 'rollouts.csv'}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HyCPAP-style Gaussian product fusion of Mask-SAC and Zone-MPC.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--patient", default="adult#001")
    parser.add_argument("--scenario", default="A")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--steps", type=int, default=576)
    parser.add_argument("--policy-mode", choices=["hybrid", "drl", "mpc"], default="hybrid")
    parser.add_argument("--mpc-std-raw-values", default="0.03,0.05,0.08,0.12,0.20")
    parser.add_argument("--drl-var-scale", type=float, default=1.0)
    parser.add_argument("--drl-var-floor", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--gru-hidden-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-masks", type=int, default=5)
    parser.add_argument("--keep-prob", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.992)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--log-std-min", type=float, default=-5.0)
    parser.add_argument("--log-std-max", type=float, default=-1.5)
    parser.add_argument("--state-mode", choices=["paper", "legacy"], default="paper")
    parser.add_argument("--meal-announcement", choices=["announced", "unannounced"], default="announced")
    parser.add_argument("--sensor-name", choices=["GuardianRT", "Dexcom", "Navigator"], default="GuardianRT")
    parser.add_argument("--action-mapping", choices=["zero_to_max", "paper_basal_centered", "basal_delta"], default="paper_basal_centered")
    parser.add_argument("--basal-delta-multiplier", type=float, default=1.0)
    parser.add_argument("--max-basal-multiplier", type=float, default=10.0)
    parser.add_argument("--mpc-variant", choices=["previous", "velocity", "adaptive"], default="adaptive")
    parser.add_argument("--mpc-prediction-horizon", type=int, default=9)
    parser.add_argument("--mpc-control-horizon", type=int, default=5)
    parser.add_argument("--mpc-target-glucose", type=float, default=110.0)
    parser.add_argument("--mpc-use-iob-constraint", action="store_true")
    parser.add_argument("--mpc-max-insulin-units-per-sample", type=float, default=1.0)
    parser.add_argument("--mpc-max-insulin-tdi-fraction", type=float, default=None)
    parser.add_argument("--mpc-model-gain-factor", type=float, default=1.0)
    parser.add_argument("--mpc-r-plus-scale", type=float, default=1.0)
    parser.add_argument("--mpc-low-glucose-suspend", action="store_true")
    parser.add_argument("--mpc-suspend-glucose", type=float, default=80.0)
    parser.add_argument("--mpc-predictive-suspend-glucose", type=float, default=100.0)
    parser.add_argument("--mpc-max-slsqp-iter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "runs" / "hybrid_drl_mpc_eval"))
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
