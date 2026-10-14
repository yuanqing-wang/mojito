"""Datasets, leakage-free splits and dense batching."""

from collections import Counter
from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd
import torch

from .chem import AtomKey, MolGraph, atom_key, canonical, parse, to_graph


class Batch(NamedTuple):
    """Dense, zero-padded batch of ``B`` molecules with ``N`` atoms max.

    ``mask`` is ``(B, N)`` and ``True`` on real atoms; the other fields
    mirror ``MolGraph`` with a leading batch axis.
    """

    atoms: torch.Tensor
    bonds: torch.Tensor
    distance: torch.Tensor
    symmetry: torch.Tensor
    mask: torch.Tensor

    def to(self, device) -> "Batch":
        return Batch(*(t.to(device) for t in self))


def collate(graphs: Sequence[MolGraph]) -> Batch:
    n = max(len(g.atoms) for g in graphs)
    b = len(graphs)
    atoms = torch.zeros(b, n, dtype=torch.long)
    bonds = torch.zeros(b, n, n, dtype=torch.long)
    distance = torch.zeros(b, n, n, dtype=torch.long)
    symmetry = torch.full((b, n), -1, dtype=torch.long)
    mask = torch.zeros(b, n, dtype=torch.bool)
    for k, g in enumerate(graphs):
        m = len(g.atoms)
        atoms[k, :m] = torch.from_numpy(g.atoms)
        bonds[k, :m, :m] = torch.from_numpy(g.bonds)
        distance[k, :m, :m] = torch.from_numpy(g.distance)
        symmetry[k, :m] = torch.from_numpy(g.symmetry)
        mask[k, :m] = True
    return Batch(atoms, bonds, distance, symmetry, mask)


def read_smiles(path: str, column: str = "smiles") -> list[str]:
    return [s.strip() for s in pd.read_csv(path)[column]]


def split(
    smiles: Sequence[str],
    fractions: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 0,
) -> tuple[list[str], list[str], list[str]]:
    """Deduplicate on canonical SMILES, then split at random.

    Deduplication precedes splitting so no molecule appears in two
    splits. Unparseable SMILES are dropped.
    """
    unique = sorted(
        {canonical(m) for m in map(parse, smiles) if m is not None}
    )
    order = np.random.default_rng(seed).permutation(len(unique))
    n_train = int(fractions[0] * len(unique))
    n_valid = int(fractions[1] * len(unique))
    pick = [unique[i] for i in order]
    return (
        pick[:n_train],
        pick[n_train:n_train + n_valid],
        pick[n_train + n_valid:],
    )


def build_vocab(smiles: Sequence[str]) -> list[AtomKey]:
    """Atom keys seen in ``smiles``, most frequent first."""
    counts = Counter(
        atom_key(a) for m in map(parse, smiles) for a in m.GetAtoms()
    )
    return [k for k, _ in counts.most_common()]


def featurize(
    smiles: Sequence[str], keys: Sequence[AtomKey]
) -> tuple[list[MolGraph], int]:
    """Featurize ``smiles``; returns graphs and the number dropped as
    out-of-vocabulary."""
    vocab = {k: i for i, k in enumerate(keys)}
    graphs, dropped = [], 0
    for mol in map(parse, smiles):
        try:
            graphs.append(to_graph(mol, vocab))
        except KeyError:
            dropped += 1
    return graphs, dropped
