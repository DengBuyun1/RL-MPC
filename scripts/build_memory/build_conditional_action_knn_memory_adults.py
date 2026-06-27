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
    build_episode_level_conditional_action_gate,
    extract_actor_history_embedding,
)


def parse_patients(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


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


def safe_memory_mask(rows: list[dict], args: argparse.Namespace) -> np.ndarray:
    mask = []
    for row in rows:
        glucose = float(row["glucose"])
        fmin30 = float(row["future_min_30"])
        fmin60 = float(row["future_min_60"])
        fmax60 = float(row["future_max_60"])
        is_safe = (
            args.safe_current_min <= glucose <= args.safe_current_max
            and fmin30 >= args.safe_future_min_30
            and fmin60 >= args.safe_future_min_60
            and fmax60 <= args.safe_future_max_60
            and not bool(row["episode_extreme"])
        )
        mask.append(is_safe)
    return np.asarray(mask, dtype=bool)


def rollout_memory(
    agent: MaskRecurrentSACAgent,
    config,
    patient_name: str,
    scenario: str,
    seed: int,
    episodes: int,
    steps_per_episode: int,
    device: torch.device,
    run_options: dict,
    deterministic_policy: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict], list[dict]]:
    action_mapping = str(run_options.get("action_mapping", "paper_basal_centered"))
    max_basal_multiplier = float(run_options.get("max_basal_multiplier", 10.0))
    basal_delta_multiplier = float(run_options.get("basal_delta_multiplier", 1.0))
    state_mode = str(run_options.get("state_mode", "paper"))
    meal_announcement = str(run_options.get("meal_announcement", "announced"))
    sensor_name = str(run_options.get("sensor_name", "GuardianRT"))

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    episode_ids: list[int] = []
    memory_rows: list[dict] = []
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

        obs, _ = env.reset(seed=episode_seed)
        glucose = float(obs[0])
        state = tracker.reset(glucose)
        history.reset(state)
        glucose_trace = [glucose]
        current_episode_rows: list[dict] = []
        total_reward = 0.0
        episode_extreme = False

        for step in range(steps_per_episode):
            history_tensor = history.tensor(device)
            embedding = extract_actor_history_embedding(agent, history_tensor)
            raw_action = agent.select_action(history_tensor, deterministic=deterministic_policy).astype(np.float32)
            memory_index = len(states)
            states.append(embedding[0])
            actions.append(raw_action.reshape(-1))
            episode_ids.append(episode)

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
            episode_extreme = episode_extreme or bool(next_glucose < 39.0 or next_glucose > 400.0)
            next_state = tracker.make_state(next_glucose, bolus_rate=bolus_rate, meal_grams=meal_grams)
            history.append(next_state, raw_action, reward)
            current_episode_rows.append(
                {
                    "memory_index": memory_index,
                    "patient": patient_name,
                    "scenario": scenario,
                    "episode": episode,
                    "seed": episode_seed,
                    "step": step,
                    "glucose": glucose,
                    "next_glucose": next_glucose,
                    "raw_action": float(raw_action.reshape(-1)[0]),
                    "basal": float(basal_action[0]),
                    "bolus": bolus_rate,
                    "meal": meal_grams,
                    "actual_meal_rate": actual_meal_rate,
                    "reward": reward,
                    "done": done,
                }
            )
            glucose = next_glucose
            glucose_trace.append(next_glucose)
            total_reward += reward
            if done:
                break

        for row in current_episode_rows:
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
                    "episode_extreme": episode_extreme,
                }
            )
        memory_rows.extend(current_episode_rows)

        metrics = clinical_metrics(glucose_trace)
        episode_rows.append(
            {
                "patient": patient_name,
                "scenario": scenario,
                "episode": episode,
                "seed": episode_seed,
                "steps": len(glucose_trace) - 1,
                "total_reward": total_reward,
                **metrics,
            }
        )
        env.close()

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(episode_ids, dtype=np.int32),
        memory_rows,
        episode_rows,
    )


def maybe_subsample(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ids: np.ndarray,
    rows: list[dict],
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    if max_samples <= 0 or len(states) <= max_samples:
        return states, actions, episode_ids, rows
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(states), size=max_samples, replace=False)
    indices.sort()
    selected_rows = [rows[int(i)] for i in indices]
    return states[indices], actions[indices], episode_ids[indices], selected_rows


def evaluate_same_patient(
    agent: MaskRecurrentSACAgent,
    config,
    gate: ConditionalActionKnnGate,
    patient_name: str,
    scenario: str,
    seed: int,
    episodes: int,
    steps_per_episode: int,
    device: torch.device,
    run_options: dict,
    deterministic_policy: bool,
) -> dict:
    states, actions, _episode_ids, _memory_rows, _rows = rollout_memory(
        agent,
        config,
        patient_name,
        scenario,
        seed,
        episodes,
        steps_per_episode,
        device,
        run_options,
        deterministic_policy,
    )
    familiar, state_scores, action_scores = gate.is_familiar(states, actions)
    return {
        "patient": patient_name,
        "scenario": scenario,
        "episodes": episodes,
        "n_steps": int(len(states)),
        "k_state": int(gate.k_state),
        "state_threshold": float(gate.state_threshold),
        "action_threshold": float(gate.action_threshold),
        "familiar_rate": float(np.mean(familiar)),
        "state_reject_rate": float(np.mean(state_scores > gate.state_threshold)),
        "action_reject_rate": float(np.mean(action_scores > gate.action_threshold)),
        "mean_state_distance": float(np.mean(state_scores)),
        "p95_state_distance": float(np.quantile(state_scores, 0.95)),
        "mean_action_distance": float(np.mean(action_scores)),
        "p95_action_distance": float(np.quantile(action_scores, 0.95)),
    }


def process_patient(patient_name: str, args: argparse.Namespace, device: torch.device) -> tuple[dict, dict]:
    run_dir = Path(args.runs_dir) / patient_run_name(patient_name, args.run_template)
    model_path = run_dir / args.model_name
    agent, config = load_agent(model_path, device)
    run_options = load_run_options(run_dir)
    states, actions, episode_ids, memory_rows, episode_rows = rollout_memory(
        agent=agent,
        config=config,
        patient_name=patient_name,
        scenario=args.memory_scenario,
        seed=args.seed,
        episodes=args.memory_episodes,
        steps_per_episode=args.steps_per_episode,
        device=device,
        run_options=run_options,
        deterministic_policy=args.deterministic_policy,
    )
    if args.safe_memory:
        full_count = len(states)
        keep = safe_memory_mask(memory_rows, args)
        states = states[keep]
        actions = actions[keep]
        episode_ids = episode_ids[keep]
        memory_rows = [row for row, keep_row in zip(memory_rows, keep) if keep_row]
        if len(states) < args.k_state:
            raise ValueError(f"{patient_name} safe memory has only {len(states)} samples, less than k_state={args.k_state}")
        print(f"  safe-memory kept {len(states)}/{full_count} samples ({len(states) / max(full_count, 1):.3f})")

    states, actions, episode_ids, memory_rows = maybe_subsample(
        states,
        actions,
        episode_ids,
        memory_rows,
        args.max_memory_samples,
        args.seed,
    )
    metadata = {
        "patient": patient_name,
        "model_path": str(model_path),
        "gate_type": "conditional_action",
        "memory_type": "safe_experienced" if args.safe_memory else "experienced",
        "memory_scenario": args.memory_scenario,
        "memory_episodes": args.memory_episodes,
        "steps_per_episode": args.steps_per_episode,
        "state_quantile": args.state_quantile,
        "action_quantile": args.action_quantile,
        "action_threshold_quantile": args.action_threshold_quantile,
        "action_source": "policy_action",
        "safe_current_min": args.safe_current_min,
        "safe_current_max": args.safe_current_max,
        "safe_future_min_30": args.safe_future_min_30,
        "safe_future_min_60": args.safe_future_min_60,
        "safe_future_max_60": args.safe_future_max_60,
    }
    gate, calib_state, calib_action = build_episode_level_conditional_action_gate(
        states,
        actions,
        episode_ids=episode_ids,
        k_state=args.k_state,
        state_quantile=args.state_quantile,
        action_quantile=args.action_quantile,
        action_threshold_quantile=args.action_threshold_quantile,
        batch_size=args.batch_size,
        metadata=metadata,
    )
    output_dir = Path(args.output_dir)
    stem = patient_file_stem(patient_name)
    gate_path = output_dir / f"{stem}_conditional_action_knn_memory.npz"
    gate.save(
        gate_path,
        calibration_state_distances=calib_state,
        calibration_action_distances=calib_action,
        memory_episode_ids=episode_ids,
    )
    write_csv(output_dir / f"{stem}_memory_episode_summary.csv", episode_rows)
    write_csv(output_dir / f"{stem}_memory_step_summary.csv", memory_rows)

    memory_summary = {
        "patient": patient_name,
        "model_path": str(model_path),
        "memory_path": str(gate_path),
        "n_memory": int(len(states)),
        "safe_memory": bool(args.safe_memory),
        "state_dim": int(states.shape[1]),
        "action_dim": int(actions.shape[1]),
        "k_state": int(args.k_state),
        "state_quantile": float(args.state_quantile),
        "action_quantile": float(args.action_quantile),
        "action_threshold_quantile": float(args.action_threshold_quantile),
        "state_threshold": float(gate.state_threshold),
        "action_threshold": float(gate.action_threshold),
        "calib_state_mean": float(np.mean(calib_state)),
        "calib_action_mean": float(np.mean(calib_action)),
    }
    same_patient_summary = evaluate_same_patient(
        agent,
        config,
        gate,
        patient_name=patient_name,
        scenario=args.eval_scenario,
        seed=args.eval_seed,
        episodes=args.eval_episodes,
        steps_per_episode=args.steps_per_episode,
        device=device,
        run_options=run_options,
        deterministic_policy=args.deterministic_policy,
    )
    return memory_summary, same_patient_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build conditional-action KNN familiarity memories.")
    parser.add_argument("--patients", default="adult#001,adult#002,adult#003,adult#004,adult#005")
    parser.add_argument("--runs-dir", required=True)
    parser.add_argument("--run-template", default="sac_baseline_fixed_{patient_dash}")
    parser.add_argument("--model-name", default="model_final.pth")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-scenario", default="train")
    parser.add_argument("--memory-episodes", type=int, default=20)
    parser.add_argument("--eval-scenario", default="A")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--steps-per-episode", type=int, default=288)
    parser.add_argument("--k-state", type=int, default=50)
    parser.add_argument("--state-quantile", type=float, default=0.95)
    parser.add_argument("--action-quantile", type=float, default=0.50)
    parser.add_argument("--action-threshold-quantile", type=float, default=0.95)
    parser.add_argument("--max-memory-samples", type=int, default=50000)
    parser.add_argument("--safe-memory", action="store_true")
    parser.add_argument("--safe-current-min", type=float, default=70.0)
    parser.add_argument("--safe-current-max", type=float, default=180.0)
    parser.add_argument("--safe-future-min-30", type=float, default=70.0)
    parser.add_argument("--safe-future-min-60", type=float, default=65.0)
    parser.add_argument("--safe-future-max-60", type=float, default=250.0)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--eval-seed", type=int, default=12000)
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

    memory_summaries: list[dict] = []
    same_patient_summaries: list[dict] = []
    for patient in parse_patients(args.patients):
        print(f"Processing {patient} on {device}...")
        memory_summary, same_summary = process_patient(patient, args, device)
        memory_summaries.append(memory_summary)
        same_patient_summaries.append(same_summary)
        print(
            f"{patient}: state_tau={memory_summary['state_threshold']:.4f} "
            f"action_tau={memory_summary['action_threshold']:.4f} "
            f"A_familiar={same_summary['familiar_rate']:.3f}"
        )

    write_csv(output_dir / "conditional_action_memory_summary.csv", memory_summaries)
    write_csv(output_dir / "conditional_action_same_patient_summary.csv", same_patient_summaries)
    print(f"Saved conditional-action KNN outputs under: {output_dir}")


if __name__ == "__main__":
    main()
