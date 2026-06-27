from __future__ import annotations

from collections import deque

import numpy as np
import torch

from .config import AgentConfig
from .networks import build_history_inputs


class HistoryBuffer:
    """Rolling history helper for evaluating recurrent policies one step at a time."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.observations = deque(maxlen=config.sequence_length)
        self.previous_actions = deque(maxlen=config.sequence_length)
        self.previous_rewards = deque(maxlen=config.sequence_length)

    def reset(self, observation) -> None:
        self.observations.clear()
        self.previous_actions.clear()
        self.previous_rewards.clear()
        zero_action = np.zeros(self.config.action_dim, dtype=np.float32)
        for _ in range(self.config.sequence_length):
            self.observations.append(np.asarray(observation, dtype=np.float32).reshape(self.config.observation_dim))
            self.previous_actions.append(zero_action.copy())
            self.previous_rewards.append(np.zeros(1, dtype=np.float32))

    def append(self, next_observation, action, reward) -> None:
        self.observations.append(np.asarray(next_observation, dtype=np.float32).reshape(self.config.observation_dim))
        self.previous_actions.append(np.asarray(action, dtype=np.float32).reshape(self.config.action_dim))
        self.previous_rewards.append(np.asarray([reward], dtype=np.float32))

    def tensor(self, device: torch.device | str = "cpu") -> torch.Tensor:
        observations = torch.as_tensor(np.asarray(self.observations), dtype=torch.float32, device=device).unsqueeze(0)
        actions = torch.as_tensor(np.asarray(self.previous_actions), dtype=torch.float32, device=device).unsqueeze(0)
        rewards = torch.as_tensor(np.asarray(self.previous_rewards), dtype=torch.float32, device=device).unsqueeze(0)
        return build_history_inputs(observations, actions, rewards)
