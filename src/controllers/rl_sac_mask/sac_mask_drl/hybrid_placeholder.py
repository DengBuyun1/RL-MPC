from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass
class GaussianPolicyDistribution:
    """Independent Gaussian action distribution."""

    mean: torch.Tensor
    var: torch.Tensor

    @property
    def std(self) -> torch.Tensor:
        return torch.sqrt(self.var.clamp_min(1e-10))


class MPCPriorProvider(Protocol):
    """Interface reserved for the paper's prior Zone-MPC policy."""

    def distribution(self, observation: torch.Tensor, history_inputs: torch.Tensor) -> GaussianPolicyDistribution:
        ...


def gaussian_product(
    drl_policy: GaussianPolicyDistribution,
    mpc_policy: GaussianPolicyDistribution,
) -> GaussianPolicyDistribution:
    """Equation-style Gaussian product for the future MPC fusion block."""

    drl_var = drl_policy.var.clamp_min(1e-10)
    mpc_var = mpc_policy.var.clamp_min(1e-10)
    mean = (drl_policy.mean * mpc_var + mpc_policy.mean * drl_var) / (drl_var + mpc_var)
    var = (drl_var * mpc_var) / (drl_var + mpc_var)
    return GaussianPolicyDistribution(mean=mean, var=var.clamp_min(1e-10))


class MissingMPCPrior:
    """Explicit placeholder until the adaptive periodic Zone-MPC is implemented."""

    def distribution(self, observation: torch.Tensor, history_inputs: torch.Tensor) -> GaussianPolicyDistribution:
        raise NotImplementedError(
            "MPC fusion is intentionally reserved. Implement adaptive periodic Zone-MPC "
            "and return its Gaussian prior distribution psi(a|s) here."
        )
