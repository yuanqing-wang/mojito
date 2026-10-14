"""Finite scalar quantization (Mentzer et al., 2023,
https://arxiv.org/abs/2309.15505)."""

from typing import Sequence

import torch
from torch import nn


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient."""
    return z + (z.round() - z).detach()


class FSQ(nn.Module):
    """Quantize each of ``len(levels)`` channels to ``levels[i]`` values.

    The implicit codebook has ``prod(levels)`` entries; no codebook
    parameters, commitment loss or collapse.

    Parameters
    ----------
    levels : Sequence[int]
        Number of quantization levels per channel.
    """

    def __init__(self, levels: Sequence[int]):
        super().__init__()
        levels = torch.tensor(levels, dtype=torch.long)
        basis = torch.cumprod(
            torch.cat([torch.ones(1, dtype=torch.long), levels[:-1]]), 0
        )
        self.register_buffer("levels", levels, persistent=False)
        self.register_buffer("basis", basis, persistent=False)

    @property
    def dim(self) -> int:
        return len(self.levels)

    @property
    def codebook_size(self) -> int:
        return int(self.levels.prod())

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        half = (self.levels - 1) * (1 + eps) / 2
        offset = (self.levels % 2 == 0) * 0.5
        shift = torch.atanh(offset / half)
        return torch.tanh(z + shift) * half - offset

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize ``(..., dim)`` to codes in ``[-1, 1]``."""
        return round_ste(self.bound(z)) / (self.levels // 2)

    def to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        width = self.levels // 2
        digits = (codes * width + width).round().long()
        return (digits * self.basis).sum(-1)

    def to_codes(self, indices: torch.Tensor) -> torch.Tensor:
        width = self.levels // 2
        digits = (indices[..., None] // self.basis) % self.levels
        return (digits - width) / width
