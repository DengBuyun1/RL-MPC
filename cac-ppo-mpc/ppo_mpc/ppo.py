from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    state_dim: int = 7
    action_dim: int = 1
    hidden_dim: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.20
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    entropy_coef: float = 0.01
    value_coef: float = 0.50
    max_grad_norm: float = 0.50
    update_epochs: int = 10
    minibatch_size: int = 256
    init_log_std: float = -0.5
    action_distribution: str = "tanh_normal_v2"
    observation_schema: str = "causal_iob_units_v2"
    squash_epsilon: float = 1e-6


class ActorCritic(nn.Module):
    def __init__(self, config: PPOConfig):
        super().__init__()
        self.config = config
        self.actor = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(config.state_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.log_std = nn.Parameter(
            torch.full((config.action_dim,), float(config.init_log_std), dtype=torch.float32)
        )

    def distribution(self, states: torch.Tensor) -> Normal:
        mean = self.actor(states)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def squash(self, raw_actions: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_actions)

    def bound_actions(self, actions: torch.Tensor) -> torch.Tensor:
        eps = float(self.config.squash_epsilon)
        return torch.clamp(actions, -1.0 + eps, 1.0 - eps)

    def unsquash(self, actions: torch.Tensor) -> torch.Tensor:
        return torch.atanh(self.bound_actions(actions))

    def squashed_log_prob(self, distribution, raw_actions=None, actions=None):
        if actions is None:
            if raw_actions is None:
                raise ValueError("raw_actions or actions must be provided")
            actions = self.squash(raw_actions)
        squashed = self.bound_actions(actions)
        # Reconstruct the effective pre-tanh value from the exact bounded
        # float action stored in the rollout.  This keeps collection and PPO
        # update likelihoods identical even when float32 tanh saturates.
        raw_actions = self.unsquash(squashed)
        eps = float(self.config.squash_epsilon)
        log_det = torch.log(torch.clamp(1.0 - squashed.pow(2), min=eps))
        return (distribution.log_prob(raw_actions) - log_det).sum(dim=-1)

    def sample_action(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.distribution(states)
        raw_actions = distribution.mean if deterministic else distribution.rsample()
        actions = self.bound_actions(self.squash(raw_actions))
        log_prob = self.squashed_log_prob(distribution, actions=actions)
        return actions, log_prob

    def value(self, states: torch.Tensor) -> torch.Tensor:
        return self.critic(states).squeeze(-1)


class RolloutBuffer:
    def __init__(self):
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[float] = []
        self.values: list[float] = []

    def add(self, state, action, log_prob, reward, done, value) -> None:
        self.states.append(np.asarray(state, dtype=np.float32))
        self.actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.dones.append(float(done))
        self.values.append(float(value))

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    def __init__(self, config: PPOConfig | None = None, device: str | torch.device | None = None):
        self.config = config or PPOConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = ActorCritic(self.config).to(self.device)
        self.policy_optimizer = torch.optim.Adam(
            list(self.model.actor.parameters()) + [self.model.log_std],
            lr=self.config.policy_lr,
        )
        self.value_optimizer = torch.optim.Adam(
            self.model.critic.parameters(), lr=self.config.value_lr
        )

    @torch.no_grad()
    def act(self, state, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t, log_prob = self.model.sample_action(state_t, deterministic=deterministic)
        value = self.model.value(state_t)
        return (
            action_t.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    @torch.no_grad()
    def value(self, state) -> float:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.model.value(state_t).item())

    def _advantages_and_returns(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
        values: np.ndarray,
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_advantage = 0.0
        for t in reversed(range(len(rewards))):
            next_nonterminal = 1.0 - dones[t]
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            delta = rewards[t] + self.config.gamma * next_value * next_nonterminal - values[t]
            last_advantage = (
                delta + self.config.gamma * self.config.gae_lambda * next_nonterminal * last_advantage
            )
            advantages[t] = last_advantage
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages.astype(np.float32), returns.astype(np.float32)

    def update(self, buffer: RolloutBuffer, last_value: float) -> dict[str, float]:
        states = np.asarray(buffer.states, dtype=np.float32)
        actions = np.asarray(buffer.actions, dtype=np.float32)
        old_log_probs = np.asarray(buffer.log_probs, dtype=np.float32)
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        values = np.asarray(buffer.values, dtype=np.float32)
        advantages, returns = self._advantages_and_returns(rewards, dones, values, last_value)

        states_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        n = states_t.shape[0]
        batch_size = min(self.config.minibatch_size, n)
        metrics: dict[str, float] = {}
        for _epoch in range(self.config.update_epochs):
            permutation = torch.randperm(n, device=self.device)
            for start in range(0, n, batch_size):
                idx = permutation[start : start + batch_size]
                dist = self.model.distribution(states_t[idx])
                new_log_probs = self.model.squashed_log_prob(dist, actions=actions_t[idx])
                _entropy_actions, entropy_log_probs = self.model.sample_action(
                    states_t[idx], deterministic=False
                )
                entropy = -entropy_log_probs.mean()
                ratio = torch.exp(new_log_probs - old_log_probs_t[idx])
                unclipped = ratio * advantages_t[idx]
                clipped = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * advantages_t[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()

                value_pred = self.model.value(states_t[idx])
                value_loss = ((value_pred - returns_t[idx]) ** 2).mean()
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy
                )

                self.policy_optimizer.zero_grad(set_to_none=True)
                self.value_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.policy_optimizer.step()
                self.value_optimizer.step()

                metrics = {
                    "policy_loss": float(policy_loss.detach().cpu().item()),
                    "value_loss": float(value_loss.detach().cpu().item()),
                    "entropy": float(entropy.detach().cpu().item()),
                    "approx_kl": float((old_log_probs_t[idx] - new_log_probs).mean().detach().cpu().item()),
                }
        return metrics

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        payload = {
            "config": asdict(self.config),
            "model_state": self.model.state_dict(),
            "extra": extra or {},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "PPOAgent":
        try:
            payload = torch.load(path, map_location=device or "cpu", weights_only=False)
        except TypeError:
            payload = torch.load(path, map_location=device or "cpu")
        saved_config = payload.get("config", {})
        config = PPOConfig(**saved_config)
        agent = cls(config=config, device=device)
        agent.model.load_state_dict(payload["model_state"])
        agent.model.eval()
        agent.checkpoint_action_distribution = saved_config.get(
            "action_distribution", "legacy_clipped_normal_v0"
        )
        agent.checkpoint_observation_schema = saved_config.get(
            "observation_schema", "legacy_iob_proxy_v1"
        )
        agent.checkpoint_extra = payload.get("extra", {})
        return agent
