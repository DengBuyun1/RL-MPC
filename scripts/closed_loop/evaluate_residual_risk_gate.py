from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAC_ROOT = PROJECT_ROOT / "src" / "controllers" / "rl_sac_mask"
SIM_ROOT = PROJECT_ROOT / "src" / "envs" / "simglucose_mpc"
for path in (SAC_ROOT, SIM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sac_mask_drl import MaskRecurrentSACAgent
from sac_mask_drl.ap_env import (
    APStateTracker,
    make_paper_simglucose_env,
    map_action_to_basal,
    patient_basal_rate,
    zone_reward,
)
from sac_mask_drl.history import HistoryBuffer
from sac_mask_drl.knn_ood_gate import ConditionalActionKnnGate, extract_actor_history_embedding
from sac_mask_drl.residual_gate import (
    OnlineRlsOneStepPredictor,
    residual_feature_vector,
    residual_score,
)


def parse_patients(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_conditions(value: str) -> list[tuple[str, str, str]]:
    out = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            scenario, meal_announcement = item.split(":", 1)
        else:
            scenario, meal_announcement = item, "announced"
        out.append((f"{scenario}_{meal_announcement}", scenario.strip(), meal_announcement.strip()))
    return out


def patient_run_name(patient_name: str, run_template: str) -> str:
    idx3 = patient_name.split("#")[-1]
    return run_template.format(
        patient=patient_name,
        patient_dash=patient_name.replace("#", "-"),
        idx3=idx3,
        idx=int(idx3),
    )


def patient_file_stem(patient_name: str) -> str:
    return patient_name.replace("#", "-")


def load_run_options(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open() as f:
        return json.load(f)


def load_agent(model_path: Path, device: torch.device) -> tuple[MaskRecurrentSACAgent, object]:
    try:
        payload = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(model_path, map_location="cpu")
    if "config" not in payload:
        raise KeyError(f"checkpoint has no AgentConfig: {model_path}")
    config = payload["config"]
    agent = MaskRecurrentSACAgent(config, device=device)
    agent.load(model_path, load_optimizers=False)
    agent.actor.eval()
    agent.critic.eval()
    return agent, config


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def future_stats(glucose: list[float], step: int, samples: int) -> tuple[float, float, float]:
    start = min(step + 1, len(glucose) - 1)
    stop = min(step + 1 + samples, len(glucose))
    window = np.asarray(glucose[start:stop], dtype=np.float32)
    if len(window) == 0:
        return np.nan, np.nan, np.nan
    return float(np.min(window)), float(np.max(window)), float(np.mean(window))


def available_announced_meal(env) -> float:
    if hasattr(env, "_announced_meal_in_next_sample"):
        return float(env._announced_meal_in_next_sample())
    return 0.0


def make_predictor(args: argparse.Namespace) -> OnlineRlsOneStepPredictor:
    return OnlineRlsOneStepPredictor(
        forgetting=args.rls_forgetting,
        p0=args.rls_p0,
        err_clip=args.rls_err_clip,
    )


def rollout_with_residual_gate(
    agent: MaskRecurrentSACAgent,
    config,
    gate: ConditionalActionKnnGate,
    patient_name: str,
    condition_label: str,
    scenario: str,
    meal_announcement: str,
    seed: int,
    episodes: int,
    steps_per_episode: int,
    device: torch.device,
    run_options: dict,
    deterministic_policy: bool,
    residual_threshold: float | None,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    action_mapping = str(run_options.get("action_mapping", "paper_basal_centered"))
    max_basal_multiplier = float(run_options.get("max_basal_multiplier", 10.0))
    basal_delta_multiplier = float(run_options.get("basal_delta_multiplier", 1.0))
    state_mode = str(run_options.get("state_mode", "paper"))
    sensor_name = str(run_options.get("sensor_name", "GuardianRT"))

    all_rows: list[dict] = []
    episode_rows: list[dict] = []

    for episode in range(episodes):
        episode_seed = seed + episode
        env, patient = make_paper_simglucose_env(
            patient_name,
            scenario,
            episode_seed,
            steps_per_episode,
            meal_announcement=meal_announcement,
            sensor_name=sensor_name,
        )
        basal_rate = patient_basal_rate(patient)
        body_weight = float(patient._params["BW"])
        max_basal = float(env.action_space.high[0])
        sample_time = float(getattr(env, "sample_time", 5.0))
        tracker = APStateTracker(body_weight, basal_rate, sample_time=sample_time, state_mode=state_mode)
        history = HistoryBuffer(config)
        predictor = make_predictor(args)

        obs, _ = env.reset(seed=episode_seed)
        glucose = float(obs[0])
        prev_glucose = glucose
        state = tracker.reset(glucose)
        history.reset(state)
        glucose_trace = [glucose]
        rows: list[dict] = []
        total_reward = 0.0
        previous_residual_score = 0.0
        previous_residual_ewma = 0.0
        previous_residual_raw = 0.0
        previous_predicted_next = np.nan

        for step in range(steps_per_episode):
            history_tensor = history.tensor(device)
            embedding = extract_actor_history_embedding(agent, history_tensor)
            raw_action = agent.select_action(history_tensor, deterministic=deterministic_policy).astype(np.float32)
            familiar_arr, state_distance_arr, action_distance_arr = gate.is_familiar(
                embedding,
                raw_action.reshape(1, -1),
            )
            knn_familiar = bool(familiar_arr[0])
            state_distance = float(state_distance_arr[0])
            action_distance = float(action_distance_arr[0])
            residual_ready = step > args.residual_warmup_steps
            residual_gate_score = (
                previous_residual_ewma if args.residual_mode == "ewma" else previous_residual_score
            )
            residual_pass = True
            if args.enable_residual_gate and residual_threshold is not None and residual_ready:
                residual_pass = residual_gate_score <= float(residual_threshold)

            basal_action = map_action_to_basal(
                raw_action,
                basal_rate,
                max_basal,
                max_basal_multiplier,
                action_mapping=action_mapping,
                basal_delta_multiplier=basal_delta_multiplier,
            )
            known_meal = available_announced_meal(env)
            current_iob = tracker.current_iob()
            glucose_rate = (glucose - prev_glucose) / max(sample_time, 1e-8)
            trend_projected_max = glucose + max(0.0, glucose_rate) * sample_time * args.trend_risk_horizon_steps
            risk_guard_pass = True
            if args.enable_risk_guard:
                risk_guard_pass = (
                    glucose <= args.max_current_glucose
                    and trend_projected_max <= args.max_projected_glucose
                )
            phi = residual_feature_vector(
                glucose=glucose,
                prev_glucose=prev_glucose,
                raw_action=float(raw_action.reshape(-1)[0]),
                basal=float(basal_action[0]),
                basal_rate=basal_rate,
                announced_meal=known_meal,
                iob=current_iob,
                sample_time=sample_time,
            )
            predicted_next = predictor.predict(phi)

            combined_familiar = knn_familiar and residual_pass and risk_guard_pass
            next_obs, _, terminated, truncated, info = env.step(basal_action)
            bolus_rate = float(info.get("bolus_rate", 0.0)) if isinstance(info, dict) else 0.0
            meal_grams = float(info.get("meal_grams", 0.0)) if isinstance(info, dict) else 0.0
            actual_meal_rate = float(info.get("actual_meal_rate", 0.0)) if isinstance(info, dict) else 0.0
            tracker.record_action(float(basal_action[0]), bolus_rate=bolus_rate)

            next_glucose = float(next_obs[0])
            reward = zone_reward(next_glucose)
            done = bool(terminated or truncated or next_glucose < 39.0 or next_glucose > 400.0)
            next_state = tracker.make_state(next_glucose, bolus_rate=bolus_rate, meal_grams=meal_grams)
            history.append(next_state, raw_action, reward)

            residual_raw = predictor.update(phi, next_glucose)
            score = residual_score(
                glucose=glucose,
                next_glucose=next_glucose,
                predicted_next_glucose=predicted_next,
                sample_time=sample_time,
                positive_weight=args.residual_positive_weight,
                slope_weight=args.residual_slope_weight,
                absolute_weight=args.residual_absolute_weight,
            )
            residual_ewma = args.residual_ewma_alpha * previous_residual_ewma + (
                1.0 - args.residual_ewma_alpha
            ) * score

            rows.append(
                {
                    "patient": patient_name,
                    "condition": condition_label,
                    "scenario": scenario,
                    "meal_announcement": meal_announcement,
                    "episode": episode,
                    "seed": episode_seed,
                    "step": step,
                    "glucose": glucose,
                    "next_glucose": next_glucose,
                    "glucose_rate": glucose_rate,
                    "raw_action": float(raw_action.reshape(-1)[0]),
                    "basal": float(basal_action[0]),
                    "bolus": bolus_rate,
                    "known_meal": known_meal,
                    "meal": meal_grams,
                    "actual_meal_rate": actual_meal_rate,
                    "iob": current_iob,
                    "reward": reward,
                    "knn_familiar": knn_familiar,
                    "familiar": bool(combined_familiar),
                    "state_reject": bool(state_distance > gate.state_threshold),
                    "action_reject": bool(action_distance > gate.action_threshold),
                    "residual_reject": bool(not residual_pass),
                    "risk_guard_reject": bool(not risk_guard_pass),
                    "enable_residual_gate": bool(args.enable_residual_gate),
                    "enable_risk_guard": bool(args.enable_risk_guard),
                    "residual_ready": bool(residual_ready),
                    "d_state": state_distance,
                    "tau_state": float(gate.state_threshold),
                    "d_action": action_distance,
                    "tau_action": float(gate.action_threshold),
                    "previous_predicted_next_glucose": previous_predicted_next,
                    "previous_residual_raw": previous_residual_raw,
                    "previous_residual_score": previous_residual_score,
                    "previous_residual_ewma": previous_residual_ewma,
                    "residual_gate_score": residual_gate_score,
                    "residual_threshold": np.nan if residual_threshold is None else float(residual_threshold),
                    "trend_projected_max": trend_projected_max,
                    "max_current_glucose": args.max_current_glucose,
                    "max_projected_glucose": args.max_projected_glucose,
                    "predicted_next_glucose": predicted_next,
                    "residual_raw": residual_raw,
                    "residual_score": score,
                    "residual_ewma": residual_ewma,
                    "done": done,
                }
            )

            previous_predicted_next = predicted_next
            previous_residual_raw = residual_raw
            previous_residual_score = score
            previous_residual_ewma = residual_ewma
            prev_glucose = glucose
            glucose = next_glucose
            glucose_trace.append(next_glucose)
            total_reward += reward
            if done:
                break

        for row in rows:
            fmin30, fmax30, fmean30 = future_stats(glucose_trace, int(row["step"]), samples=6)
            fmin60, fmax60, fmean60 = future_stats(glucose_trace, int(row["step"]), samples=12)
            row.update(
                {
                    "future_min_30": fmin30,
                    "future_max_30": fmax30,
                    "future_mean_30": fmean30,
                    "future_min_60": fmin60,
                    "future_max_60": fmax60,
                    "future_mean_60": fmean60,
                    "risk_hypo_60": bool(fmin60 < 70.0) if not np.isnan(fmin60) else False,
                    "risk_hyper_60": bool(fmax60 > 250.0) if not np.isnan(fmax60) else False,
                }
            )
        all_rows.extend(rows)
        episode_rows.append(
            {
                "patient": patient_name,
                "condition": condition_label,
                "scenario": scenario,
                "meal_announcement": meal_announcement,
                "episode": episode,
                "seed": episode_seed,
                "steps": len(glucose_trace) - 1,
                "total_reward": total_reward,
            }
        )
        env.close()

    return all_rows, episode_rows


def summarize_rows(rows: list[dict]) -> dict:
    familiar = np.asarray([bool(r["familiar"]) for r in rows], dtype=bool)
    knn_familiar = np.asarray([bool(r["knn_familiar"]) for r in rows], dtype=bool)
    residual_reject = np.asarray([bool(r["residual_reject"]) for r in rows], dtype=bool)
    risk_guard_reject = np.asarray([bool(r["risk_guard_reject"]) for r in rows], dtype=bool)
    risk = np.asarray([bool(r["risk_hypo_60"]) or bool(r["risk_hyper_60"]) for r in rows], dtype=bool)
    safe = ~risk
    return {
        "n_steps": int(len(rows)),
        "familiar_rate": float(np.mean(familiar)) if len(rows) else np.nan,
        "knn_familiar_rate": float(np.mean(knn_familiar)) if len(rows) else np.nan,
        "residual_reject_rate": float(np.mean(residual_reject)) if len(rows) else np.nan,
        "risk_guard_reject_rate": float(np.mean(risk_guard_reject)) if len(rows) else np.nan,
        "state_reject_rate": float(np.mean([bool(r["state_reject"]) for r in rows])) if rows else np.nan,
        "action_reject_rate": float(np.mean([bool(r["action_reject"]) for r in rows])) if rows else np.nan,
        "risk_any_60_rate": float(np.mean(risk)) if len(rows) else np.nan,
        "unsafe_familiar_rate": float(np.mean(risk[familiar])) if np.any(familiar) else np.nan,
        "risk_capture_rate": float(np.mean(~familiar[risk])) if np.any(risk) else np.nan,
        "over_rejection_rate": float(np.mean(~familiar[safe])) if np.any(safe) else np.nan,
        "mean_residual_gate_score": float(np.mean([float(r["residual_gate_score"]) for r in rows])) if rows else np.nan,
        "p95_residual_gate_score": float(np.quantile([float(r["residual_gate_score"]) for r in rows], 0.95)) if rows else np.nan,
    }


def calibrate_residual_threshold(
    agent: MaskRecurrentSACAgent,
    config,
    gate: ConditionalActionKnnGate,
    patient_name: str,
    device: torch.device,
    run_options: dict,
    args: argparse.Namespace,
) -> tuple[float, list[dict]]:
    label, scenario, meal_announcement = parse_conditions(args.calibration_condition)[0]
    rows, _episodes = rollout_with_residual_gate(
        agent=agent,
        config=config,
        gate=gate,
        patient_name=patient_name,
        condition_label=label,
        scenario=scenario,
        meal_announcement=meal_announcement,
        seed=args.calibration_seed,
        episodes=args.calibration_episodes,
        steps_per_episode=args.steps_per_episode,
        device=device,
        run_options=run_options,
        deterministic_policy=args.deterministic_policy,
        residual_threshold=None,
        args=args,
    )
    score_key = "residual_ewma" if args.residual_mode == "ewma" else "residual_score"
    scores = np.asarray(
        [float(row[score_key]) for row in rows if int(row["step"]) >= args.residual_warmup_steps],
        dtype=np.float32,
    )
    threshold = float(np.quantile(scores, args.residual_threshold_quantile))
    return threshold, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate KNN + online RLS residual gate.")
    parser.add_argument("--patients", default="adult#001,adult#002,adult#003,adult#004,adult#005")
    parser.add_argument("--memory-dir", required=True)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--run-template", default="sac_baseline_fixed_{patient_dash}")
    parser.add_argument("--model-name", default="model_final.pth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--conditions", default="A:announced,A:unannounced")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps-per-episode", type=int, default=288)
    parser.add_argument("--seed", type=int, default=18000)
    parser.add_argument("--calibration-condition", default="train:announced")
    parser.add_argument("--calibration-episodes", type=int, default=20)
    parser.add_argument("--calibration-seed", type=int, default=9000)
    parser.add_argument("--residual-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--enable-residual-gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--residual-mode", choices=["instant", "ewma"], default="ewma")
    parser.add_argument("--residual-ewma-alpha", type=float, default=0.90)
    parser.add_argument("--residual-warmup-steps", type=int, default=12)
    parser.add_argument("--residual-positive-weight", type=float, default=1.0)
    parser.add_argument("--residual-slope-weight", type=float, default=10.0)
    parser.add_argument("--residual-absolute-weight", type=float, default=0.0)
    parser.add_argument("--rls-forgetting", type=float, default=0.995)
    parser.add_argument("--rls-p0", type=float, default=1000.0)
    parser.add_argument("--rls-err-clip", type=float, default=80.0)
    parser.add_argument("--enable-risk-guard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-current-glucose", type=float, default=250.0)
    parser.add_argument("--max-projected-glucose", type=float, default=250.0)
    parser.add_argument("--trend-risk-horizon-steps", type=int, default=12)
    parser.add_argument("--device", default=None)
    parser.add_argument("--deterministic-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--write-rollouts", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    summaries: list[dict] = []
    calibration_summaries: list[dict] = []
    episode_summaries: list[dict] = []
    for patient_name in parse_patients(args.patients):
        run_dir = Path(args.runs_dir) / patient_run_name(patient_name, args.run_template)
        agent, config = load_agent(run_dir / args.model_name, device)
        run_options = load_run_options(run_dir)
        gate = ConditionalActionKnnGate.load(
            Path(args.memory_dir) / f"{patient_file_stem(patient_name)}_conditional_action_knn_memory.npz"
        )
        threshold, calibration_rows = calibrate_residual_threshold(
            agent,
            config,
            gate,
            patient_name,
            device,
            run_options,
            args,
        )
        calibration_summary = {
            "patient": patient_name,
            "residual_threshold": threshold,
            "calibration_condition": args.calibration_condition,
            **summarize_rows(calibration_rows),
        }
        calibration_summaries.append(calibration_summary)
        print(f"{patient_name}: residual_tau={threshold:.3f}")

        for condition_label, scenario, meal_announcement in parse_conditions(args.conditions):
            print(f"  Evaluating {condition_label}...")
            rows, ep_rows = rollout_with_residual_gate(
                agent=agent,
                config=config,
                gate=gate,
                patient_name=patient_name,
                condition_label=condition_label,
                scenario=scenario,
                meal_announcement=meal_announcement,
                seed=args.seed,
                episodes=args.episodes,
                steps_per_episode=args.steps_per_episode,
                device=device,
                run_options=run_options,
                deterministic_policy=args.deterministic_policy,
                residual_threshold=threshold,
                args=args,
            )
            summary = {
                "patient": patient_name,
                "condition": condition_label,
                "residual_threshold": threshold,
                **summarize_rows(rows),
            }
            summaries.append(summary)
            episode_summaries.extend(ep_rows)
            if args.write_rollouts:
                write_csv(output_dir / f"{patient_file_stem(patient_name)}_{condition_label}_rows.csv", rows)
            print(
                f"    familiar={summary['familiar_rate']:.3f} "
                f"risk60={summary['risk_any_60_rate']:.3f} "
                f"unsafe={summary['unsafe_familiar_rate']:.3f} "
                f"capture={summary['risk_capture_rate']:.3f}"
            )

    write_csv(output_dir / "residual_calibration_summary.csv", calibration_summaries)
    write_csv(output_dir / "residual_gate_summary.csv", summaries)
    write_csv(output_dir / "residual_gate_episode_summary.csv", episode_summaries)
    print(f"Saved residual-gate outputs under: {output_dir}")


if __name__ == "__main__":
    main()
