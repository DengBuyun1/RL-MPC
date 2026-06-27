from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from .config import AgentConfig
from .hybrid_placeholder import GaussianPolicyDistribution
from .networks import MaskRecurrentActor, MaskRecurrentCritic, build_history_inputs
from .replay_buffer import RecurrentBatch, RecurrentReplayBuffer


class MaskRecurrentSACAgent:
    """HyCPAP-style recurrent SAC with Masksembles uncertainty estimation."""

    def __init__(self, config: AgentConfig, device: torch.device | str = "cpu"):
        self.config = config
        self.device = torch.device(device)
        self.actor = MaskRecurrentActor(config).to(self.device)
        self.critic = MaskRecurrentCritic(config).to(self.device)
        self.critic_target = MaskRecurrentCritic(config).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        sac_cfg = config.sac
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=sac_cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=sac_cfg.critic_lr)
        self.auto_entropy = bool(sac_cfg.auto_entropy)
        if self.auto_entropy:
            target_entropy = sac_cfg.target_entropy
            if target_entropy is None:
                target_entropy = -float(config.action_dim)
            self.target_entropy = float(target_entropy)
            self.log_alpha = torch.tensor(np.log(sac_cfg.alpha), dtype=torch.float32, device=self.device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=sac_cfg.alpha_lr)
        else:
            self.target_entropy = None
            self.log_alpha = None
            self.alpha_optimizer = None
            self.fixed_alpha = float(sac_cfg.alpha)

    @property
    def alpha(self) -> torch.Tensor:
        if self.auto_entropy:
            return self.log_alpha.exp()
        return torch.tensor(self.fixed_alpha, dtype=torch.float32, device=self.device)

    def build_history_from_arrays(
        self,
        observations: np.ndarray,
        previous_actions: np.ndarray,
        previous_rewards: np.ndarray,
    ) -> torch.Tensor:
        observations_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device).unsqueeze(0)
        actions_t = torch.as_tensor(previous_actions, dtype=torch.float32, device=self.device).unsqueeze(0)
        rewards_t = torch.as_tensor(previous_rewards, dtype=torch.float32, device=self.device).unsqueeze(0)
        if rewards_t.dim() == 2:
            rewards_t = rewards_t.unsqueeze(-1)
        return build_history_inputs(observations_t, actions_t, rewards_t)

    @torch.no_grad()
    def select_action(self, history_inputs: torch.Tensor, deterministic: bool = False) -> np.ndarray:
        history_inputs = history_inputs.to(self.device)
        action, _ = self.actor.sample_ensemble(history_inputs, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def policy_distribution(self, history_inputs: torch.Tensor) -> GaussianPolicyDistribution:
        dist = self.actor.ensemble_distribution(history_inputs.to(self.device))
        return GaussianPolicyDistribution(mean=dist["ensemble_mean"], var=dist["ensemble_var"])

    def update(self, replay_buffer: RecurrentReplayBuffer, batch_size: int | None = None) -> Dict[str, float]:
        batch = replay_buffer.sample(batch_size or self.config.batch_size)
        return self.update_batch(batch)

    def update_batch(
        self,
        batch: RecurrentBatch,
        sample_weights: torch.Tensor | None = None,
        proximal_anchor: Dict[str, torch.Tensor] | None = None,
        proximal_coef: float = 0.0,
    ) -> Dict[str, float]:
        sac_cfg = self.config.sac
        history = batch.history_inputs()
        next_history = batch.next_history_inputs()
        actions = batch.last_actions
        rewards = batch.last_rewards
        dones = batch.last_dones
        weights = self._loss_weights(sample_weights, actions.shape[0])

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample_subpolicies(next_history)
            target_q1, target_q2 = self.critic_target(next_history, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_probs
            backup = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * sac_cfg.gamma * target_q

        current_q1, current_q2 = self.critic(history, actions)
        critic_loss = ((current_q1 - backup).pow(2) + (current_q2 - backup).pow(2)) * weights
        critic_loss = critic_loss.mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        sampled_actions, log_probs, _ = self.actor.sample_subpolicies(history)
        q1_pi, q2_pi = self.critic(history, sampled_actions)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = ((self.alpha.detach() * log_probs - min_q_pi) * weights).mean()
        prox_loss = torch.tensor(0.0, device=self.device)
        if proximal_anchor is not None and proximal_coef > 0.0:
            prox_loss = self._actor_proximal_loss(proximal_anchor)
            actor_loss = actor_loss + proximal_coef * prox_loss
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.tensor(0.0, device=self.device)
        if self.auto_entropy:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self._soft_update_targets(sac_cfg.tau)
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "alpha_loss": float(alpha_loss.detach().cpu()),
            "alpha": float(self.alpha.detach().cpu()),
            "proximal_loss": float(prox_loss.detach().cpu()),
        }

    def actor_anchor(self) -> Dict[str, torch.Tensor]:
        return {name: param.detach().clone() for name, param in self.actor.named_parameters()}

    def _loss_weights(self, sample_weights: torch.Tensor | None, batch_size: int) -> torch.Tensor:
        if sample_weights is None:
            return torch.ones((batch_size, 1, 1), dtype=torch.float32, device=self.device)
        weights = sample_weights.to(self.device, dtype=torch.float32).reshape(batch_size, 1, 1)
        weights = weights.clamp_min(1e-8)
        return weights / weights.mean().clamp_min(1e-8)

    def _actor_proximal_loss(self, anchor: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = torch.tensor(0.0, device=self.device)
        count = 0
        for name, param in self.actor.named_parameters():
            if name not in anchor:
                continue
            diff = param - anchor[name].to(self.device)
            total = total + diff.pow(2).sum()
            count += diff.numel()
        if count == 0:
            return total
        return total / count

    def _soft_update_targets(self, tau: float) -> None:
        with torch.no_grad():
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.mul_(1.0 - tau)
                target_param.data.add_(tau * param.data)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "config": self.config,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "auto_entropy": self.auto_entropy,
        }
        if self.auto_entropy:
            payload["log_alpha"] = self.log_alpha.detach().cpu()
            payload["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(payload, path)

    def load(self, path: str | Path, load_optimizers: bool = True) -> None:
        try:
            payload = torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(payload["actor"])
        self.critic.load_state_dict(payload["critic"])
        self.critic_target.load_state_dict(payload["critic_target"])
        if load_optimizers:
            self.actor_optimizer.load_state_dict(payload["actor_optimizer"])
            self.critic_optimizer.load_state_dict(payload["critic_optimizer"])
            if self.auto_entropy and "alpha_optimizer" in payload:
                self.log_alpha.data.copy_(payload["log_alpha"].to(self.device))
                self.alpha_optimizer.load_state_dict(payload["alpha_optimizer"])
