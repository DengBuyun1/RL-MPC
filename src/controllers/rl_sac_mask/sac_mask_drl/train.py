from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .agent import MaskRecurrentSACAgent
from .config import AgentConfig, MasksemblesConfig, SACConfig
from .history import HistoryBuffer
from .replay_buffer import RecurrentReplayBuffer


def train_gymnasium_env(args) -> None:
    import gymnasium as gym

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    env = gym.make(args.env_id)
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    action_limit = float(np.max(np.abs(env.action_space.high)))

    config = AgentConfig(
        observation_dim=obs_dim,
        action_dim=action_dim,
        action_limit=action_limit,
        sequence_length=args.sequence_length,
        gru_hidden_dim=args.gru_hidden_dim,
        batch_size=args.batch_size,
        masksembles=MasksemblesConfig(
            num_masks=args.num_masks,
            hidden_dim=args.hidden_dim,
            keep_prob=args.keep_prob,
            seed=args.seed,
        ),
        sac=SACConfig(gamma=args.gamma, alpha=args.alpha, auto_entropy=args.auto_entropy),
    )
    agent = MaskRecurrentSACAgent(config, device=device)
    replay = RecurrentReplayBuffer(
        capacity=args.replay_size,
        observation_dim=obs_dim,
        action_dim=action_dim,
        sequence_length=args.sequence_length,
        device=device,
    )
    history = HistoryBuffer(config)
    rng = np.random.default_rng(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    for episode in range(1, args.episodes + 1):
        observation, _ = env.reset(seed=args.seed + episode)
        observation = np.asarray(observation, dtype=np.float32).reshape(obs_dim)
        history.reset(observation)
        episode_reward = 0.0

        for _ in range(args.max_steps):
            if total_steps < args.learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.select_action(history.tensor(device), deterministic=False)
            action = np.asarray(action, dtype=np.float32).reshape(action_dim)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            next_observation = np.asarray(next_observation, dtype=np.float32).reshape(obs_dim)
            replay.push(observation, action, float(reward), next_observation, done)
            history.append(next_observation, action, float(reward))

            if len(replay) >= args.learning_starts:
                for _ in range(args.updates_per_step):
                    metrics = agent.update(replay, args.batch_size)
            else:
                metrics = None

            observation = next_observation
            episode_reward += float(reward)
            total_steps += 1
            if done:
                break

        replay.finish_episode()
        if episode % args.log_interval == 0 or episode == 1:
            suffix = ""
            if metrics:
                suffix = f" actor={metrics['actor_loss']:.3f} critic={metrics['critic_loss']:.3f} alpha={metrics['alpha']:.3f}"
            print(f"episode={episode} reward={episode_reward:.3f} steps={total_steps}{suffix}")
        if args.checkpoint_interval > 0 and episode % args.checkpoint_interval == 0:
            agent.save(output_dir / f"checkpoint_ep{episode:04d}.pth")

    agent.save(output_dir / "model_final.pth")
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train recurrent Masksembles SAC on a Gymnasium environment.")
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=288)
    parser.add_argument("--learning-starts", type=int, default=2500)
    parser.add_argument("--updates-per-step", type=int, default=1)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=0)
    default_output = Path(__file__).resolve().parents[4] / "outputs" / "runs" / "mask_recurrent_sac"
    parser.add_argument("--output-dir", default=str(default_output))
    return parser.parse_args()


if __name__ == "__main__":
    train_gymnasium_env(parse_args())
