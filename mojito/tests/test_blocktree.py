"""Correctness of the block-cut tree tokenizer: losslessness and
permutation-invariance on hand-picked hard cases."""

import random

import pytest
from rdkit import Chem

from mojito.blocktree import (
    canonical_decompose,
    canonical_smiles,
    decompose,
    reconstruct,
)

CASES = [
    "CCO",                                   # chain only
    "c1ccccc1",                              # one aromatic ring
    "CC(C)Cc1ccccc1",                        # symmetric fragment (isopropyl)
    "c1ccccc1-c1ccccc1",                     # biphenyl single-bond block
    "C1CCC2(CC1)CCCCC2",                     # spiro
    "c1ccc2ccccc2c1",                        # fused (naphthalene)
    "O=C(O)c1ccc(cc1)C(C)C(=O)O",            # multi-cut, symmetric ring
    "CN(C)C(=O)c1ccc(Cl)cc1",               # amide + substituents
    "C[NH3+]",                               # charged
    "OC(=O)c1ccc2[nH]ccc2c1",                # fused heteroaromatic
]


@pytest.mark.parametrize("smiles", CASES)
def test_round_trip_exact(smiles):
    rebuilt = reconstruct(decompose(smiles))
    assert rebuilt is not None
    assert canonical_smiles(rebuilt) == canonical_smiles(smiles)


@pytest.mark.parametrize("smiles", CASES)
def test_canonical_round_trip_exact(smiles):
    rebuilt = reconstruct(canonical_decompose(smiles))
    assert rebuilt is not None
    assert canonical_smiles(rebuilt) == canonical_smiles(smiles)


@pytest.mark.parametrize("smiles", CASES)
def test_permutation_invariant(smiles):
    """Same molecule, 8 random atom orderings -> identical decomposition."""
    m = Chem.MolFromSmiles(smiles)
    base = canonical_decompose(smiles)
    rng = random.Random(0)
    for _ in range(8):
        shuffled = Chem.MolToSmiles(m, doRandom=True, canonical=False)
        assert canonical_decompose(shuffled) == base


def test_invalid_smiles_returns_none():
    assert decompose("not_a_molecule") is None
    assert canonical_decompose("xyz!!!") is None


def test_blocks_are_biconnected_units():
    # isopropylbenzene: benzene ring + 3 single-bond blocks
    dec = canonical_decompose("CC(C)Cc1ccccc1")
    sizes = sorted(frag.count(":") for frag, _ in dec.blocks)
    # 6-membered ring has one cut (the ipso carbon); chain bonds have
    # one or two cuts each -- all blocks carry at least one tag
    assert all(s >= 1 for s in sizes)


@pytest.mark.parametrize("smiles", CASES)
def test_string_round_trip(smiles):
    from mojito.serialize import to_string, from_string
    rebuilt = from_string(to_string(smiles))
    assert rebuilt is not None
    assert canonical_smiles(rebuilt) == canonical_smiles(smiles)


@pytest.mark.parametrize("smiles", CASES)
def test_string_permutation_invariant(smiles):
    from mojito.serialize import to_string
    m = Chem.MolFromSmiles(smiles)
    base = to_string(smiles)
    for _ in range(8):
        shuffled = Chem.MolToSmiles(m, doRandom=True, canonical=False)
        assert to_string(shuffled) == base
