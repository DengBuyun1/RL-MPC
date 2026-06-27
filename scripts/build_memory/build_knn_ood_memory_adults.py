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
from sac_mask_drl.knn_ood_gate import KnnOodGate, build_episode_level_gate, build_gate, extract_actor_history_embedding
from sac_mask_drl.knn_ood_gate import build_state_action_features


def parse_patients(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_scenarios(value: str) -> list[str]:
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


def select_raw_action(agent: MaskRecurrentSACAgent, history: HistoryBuffer, device: torch.device, deterministic: bool) -> tuple[np.ndarray, float]:
    history_tensor = history.tensor(device)
    with torch.no_grad():
        dist = agent.policy_distribution(history_tensor)
        raw_action = agent.select_action(history_tensor, deterministic=deterministic).astype(np.float32)
    drl_var = float(dist.var.detach().cpu().numpy().reshape(-1)[0])
    return raw_action, drl_var


def rollout_embeddings(
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
    gate: KnnOodGate | None = None,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    embeddings: list[np.ndarray] = []
    rows: list[dict] = []
    episode_summaries: list[dict] = []

    action_mapping = str(run_options.get("action_mapping", "paper_basal_centered"))
    max_basal_multiplier = float(run_options.get("max_basal_multiplier", 10.0))
    basal_delta_multiplier = float(run_options.get("basal_delta_multiplier", 1.0))
    state_mode = str(run_options.get("state_mode", "paper"))
    meal_announcement = str(run_options.get("meal_announcement", "announced"))
    sensor_name = str(run_options.get("sensor_name", "GuardianRT"))

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
        total_reward = 0.0
        done = False

        for step in range(steps_per_episode):
            history_tensor = history.tensor(device)
            embedding = extract_actor_history_embedding(agent, history_tensor)
            embeddings.append(embedding[0])
            raw_action, drl_var = select_raw_action(agent, history, device, deterministic=deterministic_policy)

            familiar = ""
            knn_distance = ""
            if gate is not None:
                if gate.metadata.get("gate_type") == "state_action":
                    query = build_state_action_features(
                        embedding,
                        raw_action.reshape(1, -1),
                        action_mean=np.asarray(gate.metadata.get("action_mean", [0.0]), dtype=np.float32),
                        action_std=np.asarray(gate.metadata.get("action_std", [1.0]), dtype=np.float32),
                        action_beta=float(gate.metadata.get("action_beta", 1.0)),
                    )
                else:
                    query = embedding
                familiar_arr, distance_arr = gate.is_familiar(query)
                familiar = bool(familiar_arr[0])
                knn_distance = float(distance_arr[0])

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
            tracker.record_action(float(basal_action[0]), bolus_rate=bolus_rate)

            next_glucose = float(next_obs[0])
            reward = zone_reward(next_glucose)
            done = bool(terminated or truncated or next_glucose < 39.0 or next_glucose > 400.0)
            next_state = tracker.make_state(next_glucose, bolus_rate=bolus_rate, meal_grams=meal_grams)
            history.append(next_state, raw_action, reward)

            rows.append(
                {
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
                    "reward": reward,
                    "drl_var": drl_var,
                    "knn_distance": knn_distance,
                    "familiar": familiar,
                    "done": done,
                }
            )
            glucose = next_glucose
            glucose_trace.append(next_glucose)
            total_reward += reward
            if done:
                break

        metrics = clinical_metrics(glucose_trace)
        episode_summaries.append(
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

    return np.asarray(embeddings, dtype=np.float32), rows, episode_summaries


def maybe_subsample_with_episode_ids(
    embeddings: np.ndarray,
    episode_ids: np.ndarray,
    actions: np.ndarray,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if max_samples <= 0 or len(embeddings) <= max_samples:
        return embeddings, episode_ids, actions
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(embeddings), size=max_samples, replace=False)
    indices.sort()
    return embeddings[indices], episode_ids[indices], actions[indices]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_familiarity(rows: list[dict], gate: KnnOodGate) -> dict:
    distances = np.asarray([row["knn_distance"] for row in rows if row["knn_distance"] != ""], dtype=np.float32)
    familiar = np.asarray([bool(row["familiar"]) for row in rows if row["familiar"] != ""], dtype=np.bool_)
    summary = {
        "n_steps": int(len(rows)),
        "k": int(gate.k),
        "threshold": float(gate.threshold),
        "familiar_rate": float(np.mean(familiar)) if len(familiar) else np.nan,
        "ood_rate": float(1.0 - np.mean(familiar)) if len(familiar) else np.nan,
        "mean_knn_distance": float(np.mean(distances)) if len(distances) else np.nan,
        "p95_knn_distance": float(np.quantile(distances, 0.95)) if len(distances) else np.nan,
        "max_knn_distance": float(np.max(distances)) if len(distances) else np.nan,
    }
    return summary


def process_patient(patient_name: str, args: argparse.Namespace, device: torch.device) -> tuple[dict, list[dict]]:
    run_dir = Path(args.runs_dir) / patient_run_name(patient_name, args.run_template)
    model_path = run_dir / args.model_name
    if not model_path.exists():
        raise FileNotFoundError(f"missing model for {patient_name}: {model_path}")

    run_options = load_run_options(run_dir)
    agent, config = load_agent(model_path, device)
    stem = patient_file_stem(patient_name)

    memory_embeddings, memory_rows, memory_episode_rows = rollout_embeddings(
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
        gate=None,
    )
    memory_episode_ids = np.asarray([row["episode"] for row in memory_rows], dtype=np.int32)
    memory_actions = np.asarray([row["raw_action"] for row in memory_rows], dtype=np.float32).reshape(-1, 1)
    memory_embeddings, memory_episode_ids, memory_actions = maybe_subsample_with_episode_ids(
        memory_embeddings,
        memory_episode_ids,
        memory_actions,
        args.max_memory_samples,
        args.seed,
    )
    action_mean = np.mean(memory_actions, axis=0)
    action_std = np.std(memory_actions, axis=0)
    if args.gate_type == "state":
        gate_features = memory_embeddings
        memory_to_save = memory_embeddings
    elif args.gate_type == "state_action":
        gate_features = build_state_action_features(
            memory_embeddings,
            memory_actions,
            action_mean=action_mean,
            action_std=action_std,
            action_beta=args.action_beta,
        )
        memory_to_save = gate_features
    else:
        raise ValueError(f"unsupported gate_type: {args.gate_type}")
    metadata = {
        "patient": patient_name,
        "model_path": str(model_path),
        "memory_scenario": args.memory_scenario,
        "memory_episodes": args.memory_episodes,
        "steps_per_episode": args.steps_per_episode,
        "threshold_quantile": args.threshold_quantile,
        "threshold_mode": args.threshold_mode,
        "gate_type": args.gate_type,
        "action_beta": args.action_beta,
        "action_mean": action_mean.tolist(),
        "action_std": action_std.tolist(),
        "embedding": "agent.actor.encoder(history_32_steps), L2-normalized",
    }
    if args.threshold_mode == "point":
        gate, calibration_distances = build_gate(
            gate_features,
            k=args.k,
            quantile=args.threshold_quantile,
            batch_size=args.batch_size,
            metadata=metadata,
        )
    elif args.threshold_mode == "episode":
        gate, calibration_distances = build_episode_level_gate(
            gate_features,
            episode_ids=memory_episode_ids,
            k=args.k,
            quantile=args.threshold_quantile,
            batch_size=args.batch_size,
            metadata=metadata,
        )
    else:
        raise ValueError(f"unsupported threshold_mode: {args.threshold_mode}")

    output_dir = Path(args.output_dir)
    memory_path = output_dir / f"{stem}_knn_ood_memory.npz"
    gate.save(
        memory_path,
        calibration_distances=calibration_distances,
        memory_episode_ids=memory_episode_ids,
    )

    memory_summary = {
        "patient": patient_name,
        "model_path": str(model_path),
        "memory_path": str(memory_path),
        "memory_scenario": args.memory_scenario,
        "memory_episodes": args.memory_episodes,
        "n_memory": int(len(memory_embeddings)),
        "embedding_dim": int(memory_to_save.shape[1]),
        "k": int(args.k),
        "gate_type": args.gate_type,
        "action_beta": float(args.action_beta),
        "action_mean": float(action_mean[0]),
        "action_std": float(action_std[0]),
        "threshold_mode": args.threshold_mode,
        "threshold_quantile": float(args.threshold_quantile),
        "threshold": float(gate.threshold),
        "calib_mean_distance": float(np.mean(calibration_distances)),
        "calib_p95_distance": float(np.quantile(calibration_distances, 0.95)),
        "calib_max_distance": float(np.max(calibration_distances)),
    }
    write_csv(output_dir / f"{stem}_memory_episode_summary.csv", memory_episode_rows)

    eval_summaries: list[dict] = []
    if not args.skip_eval:
        for scenario in parse_scenarios(args.eval_scenarios):
            _, eval_rows, eval_episode_rows = rollout_embeddings(
                agent=agent,
                config=config,
                patient_name=patient_name,
                scenario=scenario,
                seed=args.eval_seed,
                episodes=args.eval_episodes,
                steps_per_episode=args.steps_per_episode,
                device=device,
                run_options=run_options,
                deterministic_policy=args.deterministic_policy,
                gate=gate,
            )
            write_csv(output_dir / f"{stem}_{scenario}_familiarity_rollouts.csv", eval_rows)
            write_csv(output_dir / f"{stem}_{scenario}_episode_summary.csv", eval_episode_rows)
            familiarity = summarize_familiarity(eval_rows, gate)
            eval_summaries.append(
                {
                    "patient": patient_name,
                    "scenario": scenario,
                    "eval_episodes": args.eval_episodes,
                    **familiarity,
                }
            )

    return memory_summary, eval_summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build KNN-OOD familiarity memories for personalized adult RL policies."
    )
    parser.add_argument(
        "--patients",
        default="adult#001,adult#002,adult#003,adult#004,adult#005",
        help="Comma-separated patient names.",
    )
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--run-template",
        default="personalized_{patient_dash}",
        help="Directory name template under --runs-dir. Available fields: {patient}, {patient_dash}, {idx3}, {idx}.",
    )
    parser.add_argument("--model-name", default="model_final.pth")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "runs" / "knn_ood_adult001_005"))
    parser.add_argument("--memory-scenario", default="train")
    parser.add_argument("--memory-episodes", type=int, default=20)
    parser.add_argument("--eval-scenarios", default="A")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--steps-per-episode", type=int, default=288)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--gate-type", choices=["state", "state_action"], default="state")
    parser.add_argument("--action-beta", type=float, default=1.0)
    parser.add_argument("--threshold-mode", choices=["point", "episode"], default="point")
    parser.add_argument("--threshold-quantile", type=float, default=0.95)
    parser.add_argument("--max-memory-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=9000)
    parser.add_argument("--eval-seed", type=int, default=12000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--deterministic-policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    memory_summaries: list[dict] = []
    eval_summaries: list[dict] = []
    for patient_name in parse_patients(args.patients):
        print(f"Processing {patient_name} on {device}...")
        memory_summary, patient_eval_summaries = process_patient(patient_name, args, device)
        memory_summaries.append(memory_summary)
        eval_summaries.extend(patient_eval_summaries)
        print(
            f"{patient_name}: n_memory={memory_summary['n_memory']} "
            f"threshold={memory_summary['threshold']:.4f}"
        )

    write_csv(output_dir / "memory_summary.csv", memory_summaries)
    write_csv(output_dir / "familiarity_summary.csv", eval_summaries)
    print(f"Saved KNN-OOD outputs under: {output_dir}")


if __name__ == "__main__":
    main()
