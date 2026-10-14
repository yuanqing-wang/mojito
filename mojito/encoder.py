"""Graph transformer encoder with structural attention bias."""

import torch
from torch import nn

from .chem import NUM_BOND_TYPES
from .layers import BiasedAttention


class Encoder(nn.Module):
    """Graphormer-style encoder: atoms attend to all atoms in the same
    molecule, with per-head biases from bond type and topological
    distance (permutation-invariant, unlike Laplacian PEs).

    Parameters
    ----------
    num_atom_types : int
        Atom-vocabulary size.
    hidden_features : int
        Model width.
    out_features : int
        Output width (the quantizer's dimension).
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    max_distance : int
        Distances beyond this share one bias.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        num_atom_types: int,
        hidden_features: int,
        out_features: int,
        depth: int = 4,
        num_heads: int = 8,
        max_distance: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_distance = max_distance
        self.embed = nn.Embedding(num_atom_types, hidden_features)
        self.bond_bias = nn.Embedding(NUM_BOND_TYPES, num_heads)
        self.distance_bias = nn.Embedding(max_distance + 2, num_heads)
        self.layers = nn.ModuleList(
            BiasedAttention(hidden_features, num_heads, dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(hidden_features)
        self.out = nn.Linear(hidden_features, out_features)

    def forward(
        self,
        atoms: torch.Tensor,
        bonds: torch.Tensor,
        distance: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Map ``(B, N)`` atoms to ``(B, N, out_features)``."""
        distance = distance.clamp(max=self.max_distance + 1)
        bias = self.bond_bias(bonds) + self.distance_bias(distance)
        bias = bias.permute(0, 3, 1, 2)
        h = self.embed(atoms)
        for layer in self.layers:
            h = layer(h, mask, bias)
        return self.out(self.norm(h))
