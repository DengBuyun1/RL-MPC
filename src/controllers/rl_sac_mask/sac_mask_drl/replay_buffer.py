from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from .networks import build_history_inputs, shift_actions_rewards


@dataclass
class Transition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool


@dataclass
class RecurrentBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor

    def history_inputs(self) -> torch.Tensor:
        prev_actions, prev_rewards = shift_actions_rewards(self.actions, self.rewards)
        return build_history_inputs(self.observations, prev_actions, prev_rewards)

    def next_history_inputs(self) -> torch.Tensor:
        return build_history_inputs(self.next_observations, self.actions, self.rewards)

    @property
    def last_actions(self) -> torch.Tensor:
        return self.actions[:, -1]

    @property
    def last_rewards(self) -> torch.Tensor:
        return self.rewards[:, -1]

    @property
    def last_dones(self) -> torch.Tensor:
        return self.dones[:, -1]


class RecurrentReplayBuffer:
    """Episode replay buffer that samples fixed-length sequences for recurrent SAC."""

    def __init__(
        self,
        capacity: int,
        observation_dim: int,
        action_dim: int,
        sequence_length: int,
        device: torch.device | str = "cpu",
    ):
        if capacity < sequence_length:
            raise ValueError("capacity must be >= sequence_length")
        self.capacity = int(capacity)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.sequence_length = int(sequence_length)
        self.device = torch.device(device)
        self.episodes: List[List[Transition]] = []
        self.current_episode: List[Transition] = []
        self.size = 0

    def push(self, observation, action, reward, next_observation, done) -> None:
        transition = Transition(
            observation=np.asarray(observation, dtype=np.float32).reshape(self.observation_dim),
            action=np.asarray(action, dtype=np.float32).reshape(self.action_dim),
            reward=float(reward),
            next_observation=np.asarray(next_observation, dtype=np.float32).reshape(self.observation_dim),
            done=bool(done),
        )
        self.current_episode.append(transition)
        self.size += 1

        if done:
            self.finish_episode()
        self._trim()

    def finish_episode(self) -> None:
        if self.current_episode:
            self.episodes.append(self.current_episode)
            self.current_episode = []
            self._trim()

    def sample(self, batch_size: int) -> RecurrentBatch:
        eligible = [ep for ep in self.episodes if len(ep) >= self.sequence_length]
        if len(self.current_episode) >= self.sequence_length:
            eligible.append(self.current_episode)
        if not eligible:
            raise ValueError("not enough sequence data in replay buffer")

        obs, actions, rewards, next_obs, dones = [], [], [], [], []
        for _ in range(batch_size):
            ep = eligible[np.random.randint(len(eligible))]
            start = np.random.randint(0, len(ep) - self.sequence_length + 1)
            seq = ep[start : start + self.sequence_length]
            obs.append([t.observation for t in seq])
            actions.append([t.action for t in seq])
            rewards.append([[t.reward] for t in seq])
            next_obs.append([t.next_observation for t in seq])
            dones.append([[float(t.done)] for t in seq])

        return RecurrentBatch(
            observations=torch.as_tensor(np.asarray(obs), dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device),
            rewards=torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=self.device),
            next_observations=torch.as_tensor(np.asarray(next_obs), dtype=torch.float32, device=self.device),
            dones=torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=self.device),
        )

    def _trim(self) -> None:
        while self.size > self.capacity and self.episodes:
            removed = self.episodes.pop(0)
            self.size -= len(removed)

    def __len__(self) -> int:
        return self.size
