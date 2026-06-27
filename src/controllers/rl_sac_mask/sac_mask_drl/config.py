from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class MasksemblesConfig:
    """Fixed-mask ensemble settings used by actor and critic networks."""

    num_masks: int = 5
    hidden_dim: int = 256
    keep_prob: float = 0.5
    seed: int = 0


@dataclass(frozen=True)
class SACConfig:
    """Soft Actor-Critic hyperparameters."""

    gamma: float = 0.992
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    alpha: float = 0.2
    auto_entropy: bool = False
    target_entropy: Optional[float] = None
    log_std_min: float = -20.0
    log_std_max: float = 2.0


@dataclass(frozen=True)
class AgentConfig:
    """Network and replay settings matching the HyCPAP DRL block."""

    observation_dim: int
    action_dim: int = 1
    action_limit: float = 1.0
    gru_hidden_dim: int = 48
    sequence_length: int = 32
    batch_size: int = 256
    masksembles: MasksemblesConfig = MasksemblesConfig()
    sac: SACConfig = SACConfig()
