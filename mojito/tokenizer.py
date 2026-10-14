"""Encoder -> FSQ -> decoder molecular tokenizer."""

import torch
import torch.nn.functional as F
from torch import nn

from .data import Batch
from .metrics import pair_mask


class Tokenizer(nn.Module):
    """One discrete token per heavy atom; the token set alone must
    reconstruct the molecular graph.

    Parameters
    ----------
    encoder : nn.Module
        Maps a ``Batch`` to ``(B, N, quantizer.dim)``.
    quantizer : nn.Module
        FSQ-like module with ``to_indices`` / ``to_codes``.
    decoder : nn.Module
        Maps ``(codes, mask)`` to ``(atom_logits, bond_logits)``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        quantizer: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

    def quantize(self, batch: Batch) -> torch.Tensor:
        z = self.encoder(batch.atoms, batch.bonds, batch.distance, batch.mask)
        return self.quantizer(z)

    def encode(self, batch: Batch) -> torch.Tensor:
        """``(B, N)`` token indices."""
        return self.quantizer.to_indices(self.quantize(batch))

    def decode(self, indices: torch.Tensor, mask: torch.Tensor):
        return self.decoder(self.quantizer.to_codes(indices), mask)

    def forward(self, batch: Batch):
        """Differentiable reconstruction; returns
        ``(atom_logits, bond_logits, indices)``."""
        codes = self.quantize(batch)
        atom_logits, bond_logits = self.decoder(codes, batch.mask)
        return atom_logits, bond_logits, self.quantizer.to_indices(codes)


def reconstruction_loss(
    atom_logits: torch.Tensor,
    bond_logits: torch.Tensor,
    batch: Batch,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-entropy over real atoms and over unique real atom pairs."""
    loss_atom = F.cross_entropy(
        atom_logits[batch.mask], batch.atoms[batch.mask]
    )
    pairs = pair_mask(batch.mask)
    loss_bond = F.cross_entropy(bond_logits[pairs], batch.bonds[pairs])
    return loss_atom, loss_bond
