from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ppo_mpc.envs import PPOGlucoseEnv, STATE_DIM  # noqa: E402
from ppo_mpc.ppo import PPOAgent, PPOConfig, RolloutBuffer  # noqa: E402
from ppo_mpc.provenance import (  # noqa: E402
    assert_runtime_matches_lock,
    runtime_fingerprint,
    runtime_identity,
    sha256_file,
    source_tree_sha256,
)


def parse_patients(value: str) -> list[str]:
    if value.lower() == "adult":
        return [f"adult#{idx:03d}" for idx in range(1, 11)]
    return [item.strip() for item in value.split(",") if item.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)

class EnvCycler:
    def __init__(self, args: argparse.Namespace, patients: list[str]):
        self.args = args
        self.patients = patients
        self.episode_index = -1
        self.env: PPOGlucoseEnv | None = None

    def reset(self):
        if self.env is not None:
            self.env.close()
        self.episode_index += 1
        patient = self.patients[self.episode_index % len(self.patients)]
        seed = self.args.seed + self.episode_index * 997 + int(patient.split("#")[-1])
        self.env = PPOGlucoseEnv(
            patient_name=patient,
            scenario=self.args.scenario,
            seed=seed,
            max_steps=self.args.steps,
            meal_announcement=self.args.meal_announcement,
            sensor_name=self.args.sensor_name,
            max_basal_multiplier=self.args.max_basal_multiplier,
        )
        return self.env.reset(seed=seed), patient

    def step(self, action):
        assert self.env is not None
        return self.env.step(action)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train standalone PPO on local simglucose.")
    parser.add_argument("--patients", default="adult#001")
    parser.add_argument("--scenario", default="train")
    parser.add_argument("--meal-announcement", choices=["announced", "unannounced"], default="announced")
    parser.add_argument("--sensor-name", choices=["GuardianRT", "Dexcom", "Navigator"], default="GuardianRT")
    parser.add_argument("--steps", type=int, default=288)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--total-steps", type=int, default=200000)
    parser.add_argument("--steps-per-update", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=10)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--value-lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-basal-multiplier", type=float, default=5.0)
    parser.add_argument("--output-dir", default="runs/ppo")
    parser.add_argument("--save-every-updates", type=int, default=10)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--requirements-lock",
        default=str(ROOT / "requirements-lock-server-py310-cu126.txt"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requirements_lock = Path(args.requirements_lock)
    if not requirements_lock.is_absolute():
        requirements_lock = ROOT / requirements_lock
    try:
        runtime_lock_audit = assert_runtime_matches_lock(requirements_lock)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        **vars(args),
        "action_distribution": "tanh_normal_v2",
        "observation_schema": "causal_iob_units_v2",
        "training_protocol_version": "ppo_train_v3_locked_runtime",
        "source_tree_sha256": source_tree_sha256(str(ROOT)),
        "requirements_lock_path": str(requirements_lock.resolve()),
        "requirements_lock_sha256": sha256_file(requirements_lock),
        "runtime_lock_verified": True,
        "runtime_lock_audit": runtime_lock_audit,
        "runtime_fingerprint": runtime_fingerprint(),
        "runtime_identity": runtime_identity(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }
    with (output_dir / "config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    patients = parse_patients(args.patients)
    config = PPOConfig(
        state_dim=STATE_DIM,
        hidden_dim=args.hidden_dim,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        policy_lr=args.policy_lr,
        value_lr=args.value_lr,
        entropy_coef=args.entropy_coef,
    )
    agent = PPOAgent(config=config, device=args.device)
    cycler = EnvCycler(args, patients)
    state, patient = cycler.reset()
    buffer = RolloutBuffer()
    logs: list[dict] = []
    total_steps = 0
    episode_reward = 0.0
    episode_steps = 0
    episode_index = 0
    update_index = 0

    while total_steps < args.total_steps:
        buffer.clear()
        rollout_rewards = []
        while len(buffer) < args.steps_per_update and total_steps < args.total_steps:
            action, log_prob, value = agent.act(state, deterministic=False)
            next_state, reward, done, truncated, info = cycler.step(action)
            terminal = bool(done or truncated)
            buffer.add(state, action, log_prob, reward, terminal, value)
            state = next_state
            episode_reward += float(reward)
            episode_steps += 1
            total_steps += 1
            rollout_rewards.append(float(reward))
            if terminal:
                logs.append(
                    {
                        "update": update_index,
                        "total_steps": total_steps,
                        "episode": episode_index,
                        "patient": patient,
                        "episode_steps": episode_steps,
                        "episode_reward": episode_reward,
                    }
                )
                state, patient = cycler.reset()
                episode_reward = 0.0
                episode_steps = 0
                episode_index += 1

        last_value = 0.0 if not buffer.dones or buffer.dones[-1] else agent.value(state)
        train_metrics = agent.update(buffer, last_value)
        update_index += 1
        row = {
            "update": update_index,
            "total_steps": total_steps,
            "rollout_reward_mean": float(np.mean(rollout_rewards)) if rollout_rewards else 0.0,
            **train_metrics,
        }
        logs.append(row)
        write_csv(output_dir / "training_log.csv", logs)
        if update_index % args.save_every_updates == 0:
            agent.save(
                output_dir / f"checkpoint_update_{update_index:04d}.pt",
                extra=run_config,
            )
        print(
            f"update={update_index:04d} steps={total_steps:7d} "
            f"reward={row['rollout_reward_mean']:.3f} "
            f"pi={row.get('policy_loss', 0.0):.3f} v={row.get('value_loss', 0.0):.3f}"
        )

    agent.save(output_dir / "model_final.pt", extra=run_config)
    write_csv(output_dir / "training_log.csv", logs)
    print(f"Saved PPO model under: {output_dir}")


if __name__ == "__main__":
    main()
