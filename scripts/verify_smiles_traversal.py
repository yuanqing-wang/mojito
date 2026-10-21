"""Verify the ported traversal reproduces RDKit's canonical atom order.

mojito/smiles_program.py ports RDKit's Canon.cpp (dfsFindCycles +
dfsBuildStack). This checks the port against ground truth: RDKit's own
``_smilesAtomOutputOrder`` after ``MolToSmiles``.
"""

import argparse
import random

import pandas as pd
from rdkit import Chem

from mojito.smiles_program import _traverse


def atom_order(m):
    return [ev[1] for ev in _traverse(m) if ev[0] == "atom"]


def rdkit_order(m):
    Chem.MolToSmiles(m)
    raw = m.GetProp("_smilesAtomOutputOrder")
    return [int(x) for x in raw[1:-1].rstrip(",").split(",") if x != ""]


def run(args):
    smis = pd.read_csv(args.data)["smiles"].str.strip().tolist()
    random.Random(args.seed).shuffle(smis)
    held = smis[100_000:100_000 + args.n]
    match = tot = 0
    for s in held:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        tot += 1
        # RDKit canonicalizes the aromatic mol; the port traverses it too
        match += atom_order(m) == rdkit_order(m)
    print(f"atom order matches RDKit: {match}/{tot} = {match / tot:.3%}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data")
    p.add_argument("--n", type=int, default=3000)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())
