"""PPO + ARX-RLS-Zone-MPC package for simglucose experiments."""

from __future__ import annotations

from typing import Any

__all__ = ["PPOAgent", "PPOConfig"]


def __getattr__(name: str) -> Any:
    """Keep provenance/metrics utilities importable without eagerly loading torch."""
    if name in __all__:
        from .ppo import PPOAgent, PPOConfig

        return {"PPOAgent": PPOAgent, "PPOConfig": PPOConfig}[name]
    raise AttributeError(name)
