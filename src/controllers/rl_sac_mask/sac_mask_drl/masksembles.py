from __future__ import annotations

import torch
import torch.nn as nn


def make_fixed_masks(
    num_masks: int,
    features: int,
    keep_prob: float,
    seed: int = 0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create deterministic binary masks with inverted-dropout scaling."""

    if num_masks < 1:
        raise ValueError("num_masks must be >= 1")
    if features < 1:
        raise ValueError("features must be >= 1")
    if not 0.0 < keep_prob <= 1.0:
        raise ValueError("keep_prob must be in (0, 1]")

    if num_masks == 1:
        masks = torch.ones((1, features), dtype=torch.float32)
        if device is not None:
            masks = masks.to(device)
        return masks

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    masks = torch.bernoulli(torch.full((num_masks, features), keep_prob), generator=generator)

    # Avoid degenerate submodels when a random mask drops every unit.
    empty_rows = masks.sum(dim=1) == 0
    if empty_rows.any():
        indices = torch.arange(num_masks)[empty_rows] % features
        masks[empty_rows, indices] = 1.0

    masks = masks / keep_prob
    if device is not None:
        masks = masks.to(device)
    return masks


class MasksemblesLayer(nn.Module):
    """Apply a fixed mask bank to a tensor shaped [batch, masks, features]."""

    def __init__(self, num_masks: int, features: int, keep_prob: float, seed: int = 0):
        super().__init__()
        masks = make_fixed_masks(num_masks, features, keep_prob, seed=seed)
        self.register_buffer("masks", masks)

    @property
    def num_masks(self) -> int:
        return int(self.masks.shape[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [batch, masks, features], got {tuple(x.shape)}")
        if x.shape[1] != self.num_masks:
            raise ValueError(f"expected {self.num_masks} masks, got {x.shape[1]}")
        return x * self.masks.unsqueeze(0)
