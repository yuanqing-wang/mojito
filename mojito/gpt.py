"""Minimal decoder-only transformer for molecular language modeling.

Small and dependency-light (nanoGPT-style), so a smoke run fits on CPU.
Tokenizer-agnostic: it sees integer ids and never touches chemistry.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1


class Block(nn.Module):
    """Pre-LN transformer block with causal self-attention."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.MultiheadAttention(
            cfg.n_embd, cfg.n_head, dropout=cfg.dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        x = x + self.attn(h, h, h, attn_mask=mask, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    """Decoder-only language model.

    Parameters
    ----------
    cfg : GPTConfig
        Model and vocabulary sizes.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok.weight  # weight tying

    def forward(self, idx):
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos))
        mask = torch.triu(
            torch.full((t, t), float("-inf"), device=idx.device), 1
        )
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln(x))

    def loss(self, idx, targets, ignore_index):
        """Mean token cross-entropy (natural log)."""
        logits = self(idx)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=ignore_index,
        )

    @torch.no_grad()
    def generate(self, idx, max_new, eos, temperature=1.0):
        """Autoregressively sample ids until ``eos`` or ``max_new``."""
        for _ in range(max_new):
            cond = idx[:, -self.cfg.block_size:]
            logits = self(cond)[:, -1, :] / temperature
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
            if (nxt == eos).all():
                break
        return idx
