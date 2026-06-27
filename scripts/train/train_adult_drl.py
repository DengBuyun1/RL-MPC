from __future__ import annotations

import argparse
import csv
import json
import random
import signal
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

from sac_mask_drl import AgentConfig, MaskRecurrentSACAgent, MasksemblesConfig, RecurrentReplayBuffer, SACConfig
from sac_mask_drl.ap_env import (
    APStateTracker,
    apply_safety_guard,
    adult_patients,
    basal_multiplier_to_raw_action,
    clinical_metrics,
    make_simglucose_env,
    make_paper_simglucose_env,
    map_action_to_basal,
    patient_basal_rate,
    zone_reward,
)
from sac_mask_drl.history import HistoryBuffer


def select_patients(args):
    if args.patient:
        return [args.patient]
    return adult_patients()


def load_checkpoint_config(path: str | Path, device: torch.device | str = "cpu") -> AgentConfig:
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if "config" not in payload:
        raise KeyError(f"checkpoint has no config: {path}")
    return payload["config"]


def read_last_log_state(log_path: Path) -> tuple[int, int]:
    if not log_path.exists():
        return 0, 0
    last_row = None
    with log_path.open(newline="") as f:
        for row in csv.DictReader(f):
            last_row = row
    if not last_row:
        return 0, 0
    return int(float(last_row.get("total_steps") or 0)), int(float(last_row.get("episode") or 0))


def read_csv_header(log_path: Path) -> list[str] | None:
    if not log_path.exists():
        return None
    with log_path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return None
    return header or None


def filter_row(row: dict[str, object], fieldnames: list[str]) -> dict[str, object]:
    return {field: row.get(field, "") for field in fieldnames}


def should_update(total_steps: int, learning_starts: int, replay_size: int, batch_size: int, train_freq: int) -> bool:
    if total_steps < learning_starts or replay_size < batch_size:
        return False
    return total_steps % max(1, train_freq) == 0


def make_train_env(args, patient_name: str, seed: int):
    if args.env_protocol == "paper":
        return make_paper_simglucose_env(
            patient_name,
            "train",
            seed,
            args.steps_per_episode,
            meal_announcement=args.meal_announcement,
            sensor_name=args.sensor_name,
        )
    return make_simglucose_env(patient_name, "train", seed, args.steps_per_episode)


def train(args):
    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        print(f"Received signal {signum}; saving at the end of the current episode/step.")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    patients = select_patients(args)

    if args.resume_from and args.resume_use_checkpoint_config:
        config = load_checkpoint_config(args.resume_from, device="cpu")
        args.sequence_length = int(config.sequence_length)
        args.batch_size = int(config.batch_size)
        print(f"Loaded network config from checkpoint: {args.resume_from}")
    else:
        config = AgentConfig(
            observation_dim=4,
            action_dim=1,
            action_limit=1.0,
            gru_hidden_dim=args.gru_hidden_dim,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            masksembles=MasksemblesConfig(
                num_masks=args.num_masks,
                hidden_dim=args.hidden_dim,
                keep_prob=args.keep_prob,
                seed=args.seed,
            ),
            sac=SACConfig(
                gamma=args.gamma,
                alpha=args.alpha,
                auto_entropy=args.auto_entropy,
                actor_lr=args.lr,
                critic_lr=args.lr,
                alpha_lr=args.lr,
                log_std_min=args.log_std_min,
                log_std_max=args.log_std_max,
            ),
        )
    agent = MaskRecurrentSACAgent(config, device=device)
    if args.resume_from:
        agent.load(args.resume_from, load_optimizers=not args.no_resume_optimizers)
        print(f"Resumed model: {args.resume_from}")

    replay = RecurrentReplayBuffer(args.replay_size, 4, 1, args.sequence_length, device=device)
    history = HistoryBuffer(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.json").open("w") as f:
        json.dump(vars(args), f, indent=2)

    log_fields = [
        "episode",
        "total_steps",
        "patient",
        "reward",
        "steps",
        "titr",
        "tir",
        "tbr70",
        "tbr54",
        "tar180",
        "tar250",
        "mean_bg",
        "sd_bg",
        "actor_loss",
        "critic_loss",
        "alpha",
    ]
    log_path = output_dir / "training_log.csv"
    total_steps = 0
    episode = 0
    last_metrics = None
    if args.resume_from and (args.append_log or args.resume_log):
        counter_log_path = Path(args.resume_log) if args.resume_log else log_path
        total_steps, episode = read_last_log_state(counter_log_path)
        if total_steps > 0:
            print(f"Continuing counters from log {counter_log_path}: episode={episode}, total_steps={total_steps}")

    log_mode = "a" if args.resume_from and args.append_log and log_path.exists() else "w"
    writer_fields = read_csv_header(log_path) if log_mode == "a" else None
    if writer_fields is None:
        writer_fields = log_fields
    with log_path.open(log_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=writer_fields)
        if log_mode == "w":
            writer.writeheader()
        while total_steps < args.total_steps and not stop_requested:
            episode += 1
            patient_name = random.choice(patients)
            env, patient = make_train_env(args, patient_name, args.seed + episode)
            basal_rate = patient_basal_rate(patient)
            body_weight = float(patient._params["BW"])
            max_basal = float(env.action_space.high[0])
            sample_time = float(getattr(env, "sample_time", 5.0))
            tracker = APStateTracker(body_weight, basal_rate, sample_time=sample_time, state_mode=args.state_mode)

            obs, _ = env.reset(seed=args.seed + episode)
            glucose = float(obs[0])
            state = tracker.reset(glucose)
            history.reset(state)
            glucose_trace = [glucose]
            episode_reward = 0.0
            episode_steps = 0

            for _ in range(args.steps_per_episode):
                if total_steps < args.learning_starts:
                    if args.warmup_policy == "random":
                        raw_action = np.random.uniform(-1.0, 1.0, size=(1,)).astype(np.float32)
                    else:
                        if args.warmup_policy == "basal":
                            multiplier = 1.0
                        else:
                            multiplier = np.random.uniform(args.warmup_basal_low, args.warmup_basal_high)
                        raw_action = basal_multiplier_to_raw_action(
                            multiplier,
                            args.max_basal_multiplier,
                            action_mapping=args.action_mapping,
                            basal_delta_multiplier=args.basal_delta_multiplier,
                        )
                else:
                    raw_action = agent.select_action(history.tensor(device), deterministic=False).astype(np.float32)
                    if args.exploration_raw_clip > 0:
                        raw_action = np.clip(raw_action, -args.exploration_raw_clip, args.exploration_raw_clip).astype(np.float32)

                basal_action = map_action_to_basal(
                    raw_action,
                    basal_rate,
                    max_basal,
                    args.max_basal_multiplier,
                    action_mapping=args.action_mapping,
                    basal_delta_multiplier=args.basal_delta_multiplier,
                )
                if args.safety_guard:
                    basal_action = apply_safety_guard(
                        basal_action,
                        state,
                        hypo_threshold=args.safety_hypo_threshold,
                        predictive_threshold=args.safety_predictive_threshold,
                        max_iob=args.safety_max_iob,
                    )
                next_obs, _, terminated, truncated, info = env.step(basal_action)
                bolus_rate = float(info.get("bolus_rate", 0.0)) if isinstance(info, dict) else 0.0
                meal_grams = float(info.get("meal_grams", 0.0)) if isinstance(info, dict) else 0.0
                tracker.record_action(basal_action[0], bolus_rate=bolus_rate)

                next_glucose = float(next_obs[0])
                reward = zone_reward(next_glucose)
                done = bool(terminated or truncated or next_glucose < 39.0 or next_glucose > 400.0)
                next_state = tracker.make_state(next_glucose, bolus_rate=bolus_rate, meal_grams=meal_grams)

                replay.push(state, raw_action, reward, next_state, done)
                history.append(next_state, raw_action, reward)
                if should_update(total_steps, args.learning_starts, len(replay), args.batch_size, args.train_freq):
                    for _update in range(args.gradient_steps * args.updates_per_step):
                        last_metrics = agent.update(replay, args.batch_size)

                state = next_state
                glucose_trace.append(next_glucose)
                episode_reward += reward
                total_steps += 1
                episode_steps += 1

                if done or total_steps >= args.total_steps or stop_requested:
                    break

            replay.finish_episode()
            env.close()
            metrics = clinical_metrics(glucose_trace)
            log_row = {
                "episode": episode,
                "total_steps": total_steps,
                "patient": patient_name,
                "reward": round(episode_reward, 6),
                "steps": episode_steps,
                **{key: round(value, 6) for key, value in metrics.items()},
                "actor_loss": "" if last_metrics is None else round(float(last_metrics["actor_loss"]), 6),
                "critic_loss": "" if last_metrics is None else round(float(last_metrics["critic_loss"]), 6),
                "alpha": "" if last_metrics is None else round(float(last_metrics["alpha"]), 6),
            }
            writer.writerow(filter_row(log_row, writer_fields))
            f.flush()

            if episode % args.log_interval == 0 or episode == 1:
                train_info = ""
                if last_metrics is not None:
                    train_info = f" actor={last_metrics['actor_loss']:.3f} critic={last_metrics['critic_loss']:.3f}"
                print(
                    f"ep={episode} steps={total_steps}/{args.total_steps} patient={patient_name} "
                    f"reward={episode_reward:.2f} TIR={metrics['tir']:.1f}% TBR70={metrics['tbr70']:.1f}%{train_info}"
                )

            if args.checkpoint_interval > 0 and episode % args.checkpoint_interval == 0:
                agent.save(output_dir / f"checkpoint_ep{episode:05d}.pth")

            if stop_requested:
                break

    if stop_requested:
        interrupted_path = output_dir / f"checkpoint_interrupted_ep{episode:05d}_steps{total_steps}.pth"
        agent.save(interrupted_path)
        print(f"Saved interrupted checkpoint: {interrupted_path}")
        print(f"Saved log: {log_path}")
        return

    agent.save(output_dir / "model_final.pth")
    print(f"Saved model: {output_dir / 'model_final.pth'}")
    print(f"Saved log: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train paper-style recurrent SAC-RL on adult simglucose patients; Masksembles is controlled by --num-masks.")
    parser.add_argument("--patient", default=None, help="Train one adult patient, e.g. adult#001. Omit for all 10 adults.")
    parser.add_argument(
        "--train-preset",
        choices=["quick", "dev", "standard", "paper"],
        default="paper",
        help="Training length preset used when --total-steps is omitted.",
    )
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--steps-per-episode", type=int, default=288)
    parser.add_argument("--learning-starts", type=int, default=2500)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--train-freq", type=int, default=1, help="Run gradient updates every N environment steps.")
    parser.add_argument("--gradient-steps", type=int, default=1, help="Number of SAC updates each time train-freq is reached.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=1_000_000)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--gru-hidden-dim", type=int, default=48)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-masks", type=int, default=5)
    parser.add_argument("--keep-prob", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.992)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--auto-entropy", action="store_true")
    parser.add_argument("--log-std-min", type=float, default=-5.0)
    parser.add_argument("--log-std-max", type=float, default=-1.5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--env-protocol", choices=["paper", "legacy"], default="paper")
    parser.add_argument("--state-mode", choices=["paper", "legacy"], default="paper")
    parser.add_argument("--meal-announcement", choices=["announced", "unannounced"], default="announced")
    parser.add_argument("--sensor-name", choices=["GuardianRT", "Dexcom", "Navigator"], default="GuardianRT")
    parser.add_argument("--max-basal-multiplier", type=float, default=10.0)
    parser.add_argument("--action-mapping", choices=["zero_to_max", "paper_basal_centered", "basal_delta"], default="paper_basal_centered")
    parser.add_argument("--basal-delta-multiplier", type=float, default=1.0)
    parser.add_argument("--warmup-policy", choices=["basal_noise", "basal", "random"], default="basal_noise")
    parser.add_argument("--warmup-basal-low", type=float, default=0.7)
    parser.add_argument("--warmup-basal-high", type=float, default=1.3)
    parser.add_argument("--exploration-raw-clip", type=float, default=0.25, help="Clip stochastic environment actions during training; <=0 disables.")
    parser.add_argument("--safety-guard", action="store_true")
    parser.add_argument("--safety-hypo-threshold", type=float, default=80.0)
    parser.add_argument("--safety-predictive-threshold", type=float, default=75.0)
    parser.add_argument("--safety-max-iob", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--resume-from", default=None, help="Path to a checkpoint .pth to continue training from.")
    parser.add_argument("--resume-log", default=None, help="Optional previous training_log.csv used only to continue counters.")
    parser.add_argument("--resume-use-checkpoint-config", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-resume-optimizers", action="store_true", help="Load network weights but reinitialize optimizers.")
    parser.add_argument("--append-log", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs" / "runs" / "adult_mask_drl"))
    args = parser.parse_args()
    preset_steps = {
        "quick": 50_000,
        "dev": 200_000,
        "standard": 300_000,
        "paper": 1_500_000,
    }
    if args.total_steps is None:
        args.total_steps = preset_steps[args.train_preset]
    return args


if __name__ == "__main__":
    train(parse_args())
