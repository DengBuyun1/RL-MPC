from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .agent import MaskRecurrentSACAgent
from .replay_buffer import RecurrentBatch, RecurrentReplayBuffer


def transition_features(batch: RecurrentBatch) -> torch.Tensor:
    """Use the last transition in each sequence for density-ratio classification."""

    return torch.cat(
        [
            batch.observations[:, -1],
            batch.actions[:, -1],
            batch.rewards[:, -1],
            batch.next_observations[:, -1],
            batch.dones[:, -1],
        ],
        dim=-1,
    )


def normalized_ess(weights: torch.Tensor) -> torch.Tensor:
    """Normalized effective sample size, matching the dESS form in the supplement."""

    weights = weights.reshape(-1).clamp_min(1e-8)
    numerator = weights.sum().pow(2)
    denominator = weights.numel() * weights.pow(2).sum().clamp_min(1e-8)
    return (numerator / denominator).clamp(0.0, 1.0)


class LogisticDensityRatio(nn.Module):
    """Logistic classifier used to estimate train-to-new density ratios."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)

    @torch.no_grad()
    def beta_train_to_new(self, train_features: torch.Tensor) -> torch.Tensor:
        # Labels are train=1 and new=0. With equal class priors,
        # p_new(x)/p_train(x) = (1 - D(x)) / D(x).
        p_train = torch.sigmoid(self.forward(train_features)).clamp(1e-6, 1.0 - 1e-6)
        return ((1.0 - p_train) / p_train).clamp(0.0, 100.0)


@dataclass
class MetaAdaptationConfig:
    classifier_steps: int = 200
    classifier_lr: float = 3e-4
    fine_tune_steps: int = 1_000
    replay_reuse_steps: int = 1_000
    proximal_scale: float = 0.5


class MetaHyCPAPAdapter:
    """Implements the paper's data-limited DRL adaptation loop without MPC."""

    def __init__(self, agent: MaskRecurrentSACAgent, config: MetaAdaptationConfig | None = None):
        self.agent = agent
        self.config = config or MetaAdaptationConfig()
        self.classifier: LogisticDensityRatio | None = None

    def fit_density_ratio_classifier(
        self,
        train_buffer: RecurrentReplayBuffer,
        new_buffer: RecurrentReplayBuffer,
        batch_size: int,
    ) -> LogisticDensityRatio:
        train_batch = train_buffer.sample(batch_size)
        feature_dim = transition_features(train_batch).shape[-1]
        classifier = LogisticDensityRatio(feature_dim).to(self.agent.device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=self.config.classifier_lr)

        for _ in range(self.config.classifier_steps):
            train_batch = train_buffer.sample(batch_size)
            new_batch = new_buffer.sample(batch_size)
            train_features = transition_features(train_batch)
            new_features = transition_features(new_batch)
            features = torch.cat([train_features, new_features], dim=0)
            labels = torch.cat(
                [
                    torch.ones(train_features.shape[0], device=self.agent.device),
                    torch.zeros(new_features.shape[0], device=self.agent.device),
                ],
                dim=0,
            )
            logits = classifier(features)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        self.classifier = classifier
        return classifier

    def fine_tune(self, new_buffer: RecurrentReplayBuffer, batch_size: int) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for _ in range(self.config.fine_tune_steps):
            metrics = self.agent.update(new_buffer, batch_size=batch_size)
        return metrics

    def reuse_training_data(
        self,
        train_buffer: RecurrentReplayBuffer,
        new_buffer: RecurrentReplayBuffer,
        batch_size: int,
    ) -> Dict[str, float]:
        if self.classifier is None:
            self.fit_density_ratio_classifier(train_buffer, new_buffer, batch_size)

        anchor = self.agent.actor_anchor()
        metrics: Dict[str, float] = {}
        for _ in range(self.config.replay_reuse_steps):
            batch = train_buffer.sample(batch_size)
            features = transition_features(batch)
            beta = self.classifier.beta_train_to_new(features)
            dess = normalized_ess(beta)
            proximal_coef = self.config.proximal_scale * float(1.0 - dess.detach().cpu())
            metrics = self.agent.update_batch(
                batch,
                sample_weights=beta,
                proximal_anchor=anchor,
                proximal_coef=proximal_coef,
            )
            metrics["dESS"] = float(dess.detach().cpu())
            metrics["beta_mean"] = float(beta.mean().detach().cpu())
            metrics["proximal_coef"] = proximal_coef
        return metrics
