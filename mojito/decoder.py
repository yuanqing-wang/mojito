"""Set decoder: recovers atoms and all bonds from tokens alone."""

import torch
from torch import nn

from .chem import NUM_BOND_TYPES
from .layers import BiasedAttention


class Decoder(nn.Module):
    """Transformer over the unordered token set, then a symmetric pair
    head over all atom pairs. The decoder never sees the graph.

    Parameters
    ----------
    in_features : int
        Token-code width.
    hidden_features : int
        Model width.
    num_atom_types : int
        Atom-vocabulary size.
    depth : int
        Number of transformer blocks.
    num_heads : int
        Number of attention heads.
    pair_features : int
        Width of the pairwise head.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_atom_types: int,
        depth: int = 4,
        num_heads: int = 8,
        pair_features: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fc_in = nn.Linear(in_features, hidden_features)
        self.layers = nn.ModuleList(
            BiasedAttention(hidden_features, num_heads, dropout)
            for _ in range(depth)
        )
        self.norm = nn.LayerNorm(hidden_features)
        self.atom_head = nn.Linear(hidden_features, num_atom_types)
        self.pair_in = nn.Linear(hidden_features, pair_features)
        self.pair_head = nn.Sequential(
            nn.Linear(2 * pair_features, pair_features),
            nn.GELU(),
            nn.Linear(pair_features, NUM_BOND_TYPES),
        )

    def forward(
        self, codes: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        atom_logits : torch.Tensor
            ``(B, N, num_atom_types)``.
        bond_logits : torch.Tensor
            ``(B, N, N, NUM_BOND_TYPES)``, symmetric in the atom axes.
        """
        h = self.fc_in(codes)
        for layer in self.layers:
            h = layer(h, mask)
        h = self.norm(h)
        p = self.pair_in(h)
        pi, pj = p[:, :, None], p[:, None, :]
        pair = torch.cat([pi + pj, pi * pj], dim=-1)
        return self.atom_head(h), self.pair_head(pair)
