"""Measure the block-cut tokenizer on a SMILES corpus.

Reports held-out exact round-trip, permutation-invariance (random atom
orderings), and units/molecule. Usage:

    python scripts/blocktree_stats.py DATA.csv [--n 20000] [--perms 5]
"""

import argparse
import random
import statistics as st

import pandas as pd
from rdkit import Chem

from mojito.blocktree import (
    canonical_decompose,
    canonical_smiles,
    reconstruct,
)


def run(path, n, perms, seed):
    smi = pd.read_csv(path)["smiles"].str.strip().tolist()
    random.Random(seed).shuffle(smi)
    held = smi[100_000:100_000 + n] if len(smi) > 100_000 + n else smi[:n]
    lossless = invariant = 0
    units, incid = [], []
    for s in held:
        dec = canonical_decompose(s)
        if dec is None:
            continue
        rebuilt = reconstruct(dec)
        if rebuilt is not None and (
            canonical_smiles(rebuilt) == canonical_smiles(s)
        ):
            lossless += 1
        m = Chem.MolFromSmiles(s)
        shuffles = (
            Chem.MolToSmiles(m, doRandom=True, canonical=False)
            for _ in range(perms)
        )
        invariant += all(
            canonical_decompose(x) == dec for x in shuffles
        )
        units.append(len(dec.blocks))
        incid.append(len(dec.blocks)
                     + sum(len(b.cuts) for b in dec.blocks))
    k = len(held)
    print(f"molecules: {k}")
    print(f"exact round-trip:        {lossless}/{k} = {lossless / k:.3%}")
    print(f"permutation-invariant:   {invariant}/{k} = {invariant / k:.3%}")
    print(f"blocks/molecule:         {st.mean(units):.1f}")
    print(f"blocks+incidences/mol:   {st.mean(incid):.1f}  (pre-merge)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data")
    p.add_argument("--n", type=int, default=20_000)
    p.add_argument("--perms", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(a.data, a.n, a.perms, a.seed)
