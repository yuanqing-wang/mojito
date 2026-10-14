"""Reconstruction and codebook metrics, accumulated as sums so that
dataset-level values are exact (not means of batch means)."""

from collections import defaultdict

import torch


def pair_mask(mask: torch.Tensor) -> torch.Tensor:
    """``(B, N, N)`` mask of unique real pairs ``i < j``."""
    n = mask.shape[1]
    upper = torch.ones(n, n, dtype=torch.bool, device=mask.device).triu(1)
    return mask[:, :, None] & mask[:, None, :] & upper


def collisions(
    indices: torch.Tensor, symmetry: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """``(B,)`` bool: two non-equivalent atoms share a token.

    A permutation-equivariant decoder maps equal tokens to equal rows,
    so a collision makes exact reconstruction impossible; the
    non-collision rate upper-bounds exact match.
    """
    same = indices[:, :, None] == indices[:, None, :]
    differ = symmetry[:, :, None] != symmetry[:, None, :]
    return (same & differ & pair_mask(mask)).flatten(1).any(1)


def batch_counts(atom_logits, bond_logits, indices, batch) -> dict:
    """Sufficient statistics for ``summarize``."""
    pairs = pair_mask(batch.mask)
    atom_hat = atom_logits.argmax(-1)
    bond_hat = bond_logits.argmax(-1)
    atom_ok = (atom_hat == batch.atoms) | ~batch.mask
    bond_ok = (bond_hat == batch.bonds) | ~pairs
    true_bond = (batch.bonds > 0) & pairs
    pred_bond = (bond_hat > 0) & pairs
    exact = atom_ok.all(1) & bond_ok.flatten(1).all(1)
    return {
        "atoms": batch.mask.sum(),
        "atoms_correct": (atom_ok & batch.mask).sum(),
        "bonds_true": true_bond.sum(),
        "bonds_pred": pred_bond.sum(),
        "bonds_tp": (true_bond & (bond_hat == batch.bonds)).sum(),
        "molecules": torch.tensor(len(exact)),
        "exact": exact.sum(),
        "collisions": collisions(indices, batch.symmetry, batch.mask).sum(),
        "usage": torch.bincount(indices[batch.mask]),
    }


class Accumulator:
    """Sum ``batch_counts`` dicts over an epoch."""

    def __init__(self):
        self.sums = defaultdict(int)

    def update(self, counts: dict):
        for k, v in counts.items():
            v = v.detach().cpu()
            if k == "usage":
                old = self.sums[k]
                if isinstance(old, int):
                    old = torch.zeros(0, dtype=torch.long)
                size = max(len(old), len(v))
                new = torch.zeros(size, dtype=torch.long)
                new[:len(old)] += old
                new[:len(v)] += v
                self.sums[k] = new
            else:
                self.sums[k] += int(v)

    def summarize(self, codebook_size: int) -> dict:
        s = self.sums
        precision = s["bonds_tp"] / max(s["bonds_pred"], 1)
        recall = s["bonds_tp"] / max(s["bonds_true"], 1)
        p = s["usage"].float() / s["usage"].sum()
        p = p[p > 0]
        return {
            "atom_acc": s["atoms_correct"] / s["atoms"],
            "bond_precision": precision,
            "bond_recall": recall,
            "bond_f1": 2 * precision * recall
            / max(precision + recall, 1e-12),
            "exact": s["exact"] / s["molecules"],
            "exact_upper_bound": 1 - s["collisions"] / s["molecules"],
            "codes_used": len(p),
            "code_utilization": len(p) / codebook_size,
            "code_perplexity": float(torch.exp(-(p * p.log()).sum())),
        }
