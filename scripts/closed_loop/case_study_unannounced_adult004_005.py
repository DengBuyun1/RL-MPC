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
    clinical_metrics,
    make_paper_simglucose_env,
    map_action_to_basal,
    patient_basal_rate,
    zone_reward,
)
from sac_mask_drl.history import HistoryBuffer
from sac_mask_drl.knn_ood_gate import (
    ConditionalActionKnnGate,
    extract_actor_history_embedding,
    kth_neighbor_indices_and_distances,
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


def local_neighbor_stats(gate: ConditionalActionKnnGate, embedding: np.ndarray, raw_action: np.ndarray) -> dict:
    indices, state_distances = kth_neighbor_indices_and_distances(
        embedding,
        gate.memory_state,
        k=gate.k_state,
        batch_size=1,
    )
    idx = indices[0]
    d_state = float(state_distances[0, -1])
    neighbor_actions = gate.memory_action[idx].reshape(gate.k_state, -1)
    raw = raw_action.reshape(1, -1)
    action_diff = np.abs(neighbor_actions - raw)
    if action_diff.shape[1] == 1:
        action_diff_scalar = action_diff[:, 0]
    else:
        action_diff_scalar = np.linalg.norm(action_diff, axis=1)
    d_action = float(np.quantile(action_diff_scalar, gate.action_quantile))
    return {
        "d_state": d_state,
        "tau_state": float(gate.state_threshold),
        "d_action": d_action,
        "tau_action": float(gate.action_threshold),
        "neighbor_state_distance_mean": float(np.mean(state_distances[0])),
        "neighbor_state_distance_min": float(np.min(state_distances[0])),
        "neighbor_state_distance_max": float(np.max(state_distances[0])),
        "neighbor_action_mean": float(np.mean(neighbor_actions[:, 0])),
        "neighbor_action_std": float(np.std(neighbor_actions[:, 0])),
        "neighbor_action_min": float(np.min(neighbor_actions[:, 0])),
        "neighbor_action_max": float(np.max(neighbor_actions[:, 0])),
        "neighbor_action_absdiff_mean": float(np.mean(action_diff_scalar)),
        "neighbor_action_absdiff_p50": float(np.quantile(action_diff_scalar, 0.50)),
        "neighbor_action_absdiff_p95": float(np.quantile(action_diff_scalar, 0.95)),
    }


def evaluate_condition(
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
) -> tuple[list[dict], list[dict]]:
    action_mapping = str(run_options.get("action_mapping", "paper_basal_centered"))
    max_basal_multiplier = float(run_options.get("max_basal_multiplier", 10.0))
    basal_delta_multiplier = float(run_options.get("basal_delta_multiplier", 1.0))
    state_mode = str(run_options.get("state_mode", "paper"))
    sensor_name = str(run_options.get("sensor_name", "GuardianRT"))

    all_rows: list[dict] = []
    episode_summaries: list[dict] = []

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

        obs, _ = env.reset(seed=episode_seed)
        glucose = float(obs[0])
        state = tracker.reset(glucose)
        history.reset(state)
        glucose_trace = [glucose]
        episode_rows: list[dict] = []
        total_reward = 0.0
        prev_glucose = glucose

        for step in range(steps_per_episode):
            history_tensor = history.tensor(device)
            embedding = extract_actor_history_embedding(agent, history_tensor)
            with torch.no_grad():
                dist = agent.policy_distribution(history_tensor)
                raw_action = agent.select_action(history_tensor, deterministic=deterministic_policy).astype(np.float32)
            drl_var = float(dist.var.detach().cpu().numpy().reshape(-1)[0])
            neighbor = local_neighbor_stats(gate, embedding, raw_action)
            familiar = (neighbor["d_state"] <= gate.state_threshold) and (neighbor["d_action"] <= gate.action_threshold)

            basal_action = map_action_to_basal(
                raw_action,
                basal_rate,
                max_basal,
                max_basal_multiplier,
                action_mapping=action_mapping,
                basal_delta_multiplier=basal_delta_multiplier,
            )
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

            row = {
                "patient": patient_name,
                "condition": condition_label,
                "scenario": scenario,
                "meal_announcement": meal_announcement,
                "episode": episode,
                "seed": episode_seed,
                "step": step,
                "time_min": step * sample_time,
                "glucose": glucose,
                "next_glucose": next_glucose,
                "glucose_rate": (glucose - prev_glucose) / max(sample_time, 1e-8),
                "raw_action": float(raw_action.reshape(-1)[0]),
                "basal": float(basal_action[0]),
                "bolus": bolus_rate,
                "meal": meal_grams,
                "actual_meal_rate": actual_meal_rate,
                "reward": reward,
                "drl_var": drl_var,
                "familiar": bool(familiar),
                "state_reject": bool(neighbor["d_state"] > gate.state_threshold),
                "action_reject": bool(neighbor["d_action"] > gate.action_threshold),
                "done": done,
                **neighbor,
            }
            episode_rows.append(row)

            prev_glucose = glucose
            glucose = next_glucose
            glucose_trace.append(next_glucose)
            total_reward += reward
            if done:
                break

        for row in episode_rows:
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
                    "risk_hypo_30": bool(fmin30 < 70.0) if not np.isnan(fmin30) else False,
                    "risk_hypo_60": bool(fmin60 < 70.0) if not np.isnan(fmin60) else False,
                    "risk_severe_hypo_60": bool(fmin60 < 54.0) if not np.isnan(fmin60) else False,
                    "risk_hyper_60": bool(fmax60 > 250.0) if not np.isnan(fmax60) else False,
                }
            )
        all_rows.extend(episode_rows)
        metrics = clinical_metrics(glucose_trace)
        episode_summaries.append(
            {
                "patient": patient_name,
                "condition": condition_label,
                "scenario": scenario,
                "meal_announcement": meal_announcement,
                "episode": episode,
                "seed": episode_seed,
                "steps": len(glucose_trace) - 1,
                "total_reward": total_reward,
                **metrics,
            }
        )
        env.close()

    return all_rows, episode_summaries


def summarize_rows(rows: list[dict]) -> dict:
    familiar = np.asarray([bool(r["familiar"]) for r in rows], dtype=bool)
    hypo60 = np.asarray([bool(r["risk_hypo_60"]) for r in rows], dtype=bool)
    hyper60 = np.asarray([bool(r["risk_hyper_60"]) for r in rows], dtype=bool)
    risk60 = hypo60 | hyper60
    if np.any(familiar):
        unsafe_familiar_rate = float(np.mean(risk60[familiar]))
    else:
        unsafe_familiar_rate = np.nan
    if np.any(risk60):
        risk_capture_rate = float(np.mean(~familiar[risk60]))
    else:
        risk_capture_rate = np.nan
    return {
        "n_steps": int(len(rows)),
        "familiar_rate": float(np.mean(familiar)) if len(familiar) else np.nan,
        "state_reject_rate": float(np.mean([bool(r["state_reject"]) for r in rows])) if rows else np.nan,
        "action_reject_rate": float(np.mean([bool(r["action_reject"]) for r in rows])) if rows else np.nan,
        "risk_hypo_60_rate": float(np.mean(hypo60)) if len(hypo60) else np.nan,
        "risk_hyper_60_rate": float(np.mean(hyper60)) if len(hyper60) else np.nan,
        "risk_any_60_rate": float(np.mean(risk60)) if len(risk60) else np.nan,
        "unsafe_familiar_rate": unsafe_familiar_rate,
        "risk_capture_rate": risk_capture_rate,
        "mean_d_state": float(np.mean([float(r["d_state"]) for r in rows])) if rows else np.nan,
        "mean_d_action": float(np.mean([float(r["d_action"]) for r in rows])) if rows else np.nan,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Case study for adult#004/#005 announced vs unannounced familiarity.")
    parser.add_argument("--patients", default="adult#004,adult#005")
    parser.add_argument("--memory-dir", required=True)
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--run-template", default="sac_baseline_fixed_{patient_dash}")
    parser.add_argument("--model-name", default="model_final.pth")
    parser.add_argument("--conditions", default="A:announced,A:unannounced")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps-per-episode", type=int, default=288)
    parser.add_argument("--seed", type=int, default=18000)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--deterministic-policy", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    all_summary_rows: list[dict] = []
    all_episode_rows: list[dict] = []
    for patient_name in parse_patients(args.patients):
        run_dir = Path(args.runs_dir) / patient_run_name(patient_name, args.run_template)
        agent, config = load_agent(run_dir / args.model_name, device)
        run_options = load_run_options(run_dir)
        gate = ConditionalActionKnnGate.load(
            Path(args.memory_dir) / f"{patient_file_stem(patient_name)}_conditional_action_knn_memory.npz"
        )
        print(
            f"Loaded {patient_name}: state_tau={gate.state_threshold:.4f} "
            f"action_tau={gate.action_threshold:.4f}"
        )
        for condition_label, scenario, meal_announcement in parse_conditions(args.conditions):
            print(f"  Evaluating {condition_label}...")
            rows, episode_rows = evaluate_condition(
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
            )
            summary = {
                "patient": patient_name,
                "condition": condition_label,
                **summarize_rows(rows),
            }
            all_summary_rows.append(summary)
            all_episode_rows.extend(episode_rows)
            write_csv(output_dir / f"{patient_file_stem(patient_name)}_{condition_label}_case_rows.csv", rows)
            print(
                f"    familiar={summary['familiar_rate']:.3f} "
                f"risk60={summary['risk_any_60_rate']:.3f} "
                f"unsafe_familiar={summary['unsafe_familiar_rate']:.3f}"
            )

    write_csv(output_dir / "case_study_summary.csv", all_summary_rows)
    write_csv(output_dir / "case_study_episode_summary.csv", all_episode_rows)
    print(f"Saved case-study outputs under: {output_dir}")


if __name__ == "__main__":
    main()
