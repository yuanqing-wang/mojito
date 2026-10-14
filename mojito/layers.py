"""Transformer block with additive per-head attention bias."""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


class BiasedAttention(nn.Module):
    """Pre-LN transformer block over padded sets.

    Parameters
    ----------
    hidden_features : int
        Model width; must be divisible by ``num_heads``.
    num_heads : int
        Number of attention heads.
    dropout : float
        Dropout on attention weights and the feed-forward path.
    """

    def __init__(
        self,
        hidden_features: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert hidden_features % num_heads == 0
        self.num_heads = num_heads
        self.dropout = dropout
        self.norm0 = nn.LayerNorm(hidden_features)
        self.qkv = nn.Linear(hidden_features, 3 * hidden_features)
        self.proj = nn.Linear(hidden_features, hidden_features)
        self.norm1 = nn.LayerNorm(hidden_features)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_features, 4 * hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_features, hidden_features),
        )

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        h : torch.Tensor
            ``(B, N, D)`` node features.
        mask : torch.Tensor
            ``(B, N)`` bool, ``True`` on real nodes.
        bias : torch.Tensor, optional
            ``(B, H, N, N)`` additive attention logits.
        """
        b, n, d = h.shape
        q, k, v = (
            self.qkv(self.norm0(h))
            .view(b, n, 3, self.num_heads, d // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        attn = torch.zeros(b, 1, n, n, device=h.device, dtype=h.dtype)
        if bias is not None:
            attn = attn + bias
        attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        x = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn,
            dropout_p=self.dropout if self.training else 0.0,
        )
        h = h + self.proj(x.transpose(1, 2).reshape(b, n, d))
        return h + self.ffn(self.norm1(h))
