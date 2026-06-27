from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .config import AgentConfig
from .masksembles import MasksemblesLayer


def build_history_inputs(
    observations: torch.Tensor,
    previous_actions: torch.Tensor,
    previous_rewards: torch.Tensor,
) -> torch.Tensor:
    """Concatenate observation, previous action, and previous reward histories."""

    if previous_rewards.dim() == 2:
        previous_rewards = previous_rewards.unsqueeze(-1)
    return torch.cat([observations, previous_actions, previous_rewards], dim=-1)


def shift_actions_rewards(actions: torch.Tensor, rewards: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build previous-action/reward streams for recurrent policy inputs."""

    if rewards.dim() == 2:
        rewards = rewards.unsqueeze(-1)
    zeros_a = torch.zeros_like(actions[:, :1])
    zeros_r = torch.zeros_like(rewards[:, :1])
    prev_actions = torch.cat([zeros_a, actions[:, :-1]], dim=1)
    prev_rewards = torch.cat([zeros_r, rewards[:, :-1]], dim=1)
    return prev_actions, prev_rewards


class RecurrentEncoder(nn.Module):
    """GRU encoder for the POMDP history used by the HyCPAP DRL policy."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)

    def forward(self, history_inputs: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(history_inputs)
        return hidden[-1]


class MaskedMLP(nn.Module):
    """Shared MLP with fixed masks inserted between layers."""

    def __init__(self, input_dim: int, output_dim: int, config: AgentConfig, seed_offset: int = 0):
        super().__init__()
        mask_cfg = config.masksembles
        self.num_masks = mask_cfg.num_masks
        hidden_dim = mask_cfg.hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.mask1 = MasksemblesLayer(
            mask_cfg.num_masks,
            hidden_dim,
            mask_cfg.keep_prob,
            seed=mask_cfg.seed + seed_offset,
        )
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mask2 = MasksemblesLayer(
            mask_cfg.num_masks,
            hidden_dim,
            mask_cfg.keep_prob,
            seed=mask_cfg.seed + seed_offset + 10_000,
        )
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = F.relu(self.fc1(x))
            x = x.unsqueeze(1).expand(-1, self.num_masks, -1)
        elif x.dim() == 3:
            bsz, num_masks, input_dim = x.shape
            if num_masks != self.num_masks:
                raise ValueError(f"expected {self.num_masks} masks, got {num_masks}")
            x = F.relu(self.fc1(x.reshape(bsz * num_masks, input_dim))).reshape(bsz, num_masks, -1)
        else:
            raise ValueError(f"expected [batch, features] or [batch, masks, features], got {tuple(x.shape)}")
        x = self.mask1(x)
        bsz, num_masks, hidden_dim = x.shape
        x = F.relu(self.fc2(x.reshape(bsz * num_masks, hidden_dim)))
        x = x.reshape(bsz, num_masks, hidden_dim)
        x = self.mask2(x)
        return self.out(x.reshape(bsz * num_masks, hidden_dim)).reshape(bsz, num_masks, -1)


class MaskRecurrentActor(nn.Module):
    """Masksembles recurrent Gaussian actor."""

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        input_dim = config.observation_dim + config.action_dim + 1
        self.encoder = RecurrentEncoder(input_dim, config.gru_hidden_dim)
        self.backbone = MaskedMLP(
            config.gru_hidden_dim,
            2 * config.action_dim,
            config,
            seed_offset=0,
        )

    @property
    def num_masks(self) -> int:
        return self.config.masksembles.num_masks

    @property
    def action_limit(self) -> float:
        return self.config.action_limit

    def forward(self, history_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(history_inputs)
        params = self.backbone(features)
        mean, log_std = torch.chunk(params, 2, dim=-1)
        log_std = torch.clamp(log_std, self.config.sac.log_std_min, self.config.sac.log_std_max)
        return mean, log_std

    def sample_subpolicies(self, history_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.forward(history_inputs)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        squashed = torch.tanh(z)
        action = squashed * self.action_limit
        log_prob = normal.log_prob(z)
        log_prob -= torch.log(self.action_limit * (1.0 - squashed.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        deterministic_action = torch.tanh(mean) * self.action_limit
        return action, log_prob, deterministic_action

    def ensemble_distribution(self, history_inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return subpolicy and moment-matched ensemble Gaussian parameters."""

        mean, log_std = self.forward(history_inputs)
        std = log_std.exp()
        tanh_mean = torch.tanh(mean)
        action_mean = tanh_mean * self.action_limit
        action_var = ((1.0 - tanh_mean.pow(2)) * self.action_limit).pow(2) * std.pow(2)
        ensemble_mean = action_mean.mean(dim=1)
        ensemble_var = (action_var + action_mean.pow(2)).mean(dim=1) - ensemble_mean.pow(2)
        ensemble_var = ensemble_var.clamp_min(1e-10)
        return {
            "subpolicy_mean": action_mean,
            "subpolicy_var": action_var.clamp_min(1e-10),
            "ensemble_mean": ensemble_mean,
            "ensemble_var": ensemble_var,
        }

    def sample_ensemble(self, history_inputs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, dict[str, torch.Tensor]]:
        dist = self.ensemble_distribution(history_inputs)
        mean = dist["ensemble_mean"]
        var = dist["ensemble_var"]
        if deterministic:
            action = mean
        else:
            action = mean + torch.randn_like(mean) * torch.sqrt(var)
            action = action.clamp(-self.action_limit, self.action_limit)
        return action, dist


class MaskRecurrentCritic(nn.Module):
    """Twin recurrent Q-functions with Masksembles heads."""

    def __init__(self, config: AgentConfig):
        super().__init__()
        self.config = config
        input_dim = config.observation_dim + config.action_dim + 1
        q_input_dim = config.gru_hidden_dim + config.action_dim
        self.encoder_q1 = RecurrentEncoder(input_dim, config.gru_hidden_dim)
        self.encoder_q2 = RecurrentEncoder(input_dim, config.gru_hidden_dim)
        self.q1 = MaskedMLP(q_input_dim, 1, config, seed_offset=20_000)
        self.q2 = MaskedMLP(q_input_dim, 1, config, seed_offset=30_000)

    @property
    def num_masks(self) -> int:
        return self.config.masksembles.num_masks

    def forward(self, history_inputs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if actions.dim() == 2:
            actions = actions.unsqueeze(1).expand(-1, self.num_masks, -1)
        if actions.dim() != 3:
            raise ValueError(f"expected actions with shape [batch, masks, action_dim], got {tuple(actions.shape)}")

        features_q1 = self.encoder_q1(history_inputs)
        features_q2 = self.encoder_q2(history_inputs)
        q1_input = torch.cat([features_q1.unsqueeze(1).expand(-1, self.num_masks, -1), actions], dim=-1)
        q2_input = torch.cat([features_q2.unsqueeze(1).expand(-1, self.num_masks, -1), actions], dim=-1)

        return self.q1(q1_input), self.q2(q2_input)
