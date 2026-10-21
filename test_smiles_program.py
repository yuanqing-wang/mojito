"""Tests for smiles_program: round-trip, permutation-invariance, and
fidelity to RDKit's canonical atom order."""

import random

import pytest
from rdkit import Chem

import smiles_program as sp

CASES = [
    "C",                                  # single atom
    "CCO",                                # chain
    "CC(C)C",                             # branch
    "c1ccccc1",                           # aromatic ring
    "c1ccccc1O",                          # ring + substituent
    "CC(C)Cc1ccccc1",                     # symmetric branch
    "c1ccc2ccccc2c1",                     # fused rings (naphthalene)
    "C1CCC2(CC1)CCCCC2",                  # spiro
    "O=C(O)c1ccc(cc1)C(C)C(=O)O",         # multi-ring, carboxyl
    "CN(C)C(=O)c1ccc(Cl)cc1",            # amide + halogen
    "C[NH3+]",                            # positive charge
    "O=C([O-])c1ccc[nH]1",                # negative charge + heteroarom
    "OC(=O)c1ccc2[nH]ccc2c1",             # fused heteroaromatic
    "CCOC(=O)c1ccccc1N",                  # ester + amine
    "CC.CCO",                             # multiple fragments
    "C1CC1C2CC2",                         # two small rings joined
]


def rdkit_atom_order(mol):
    Chem.MolToSmiles(mol)
    raw = mol.GetProp("_smilesAtomOutputOrder")
    return [int(x) for x in raw[1:-1].rstrip(",").split(",") if x != ""]


@pytest.mark.parametrize("smiles", CASES)
def test_round_trip(smiles):
    rebuilt = sp.decode(sp.encode(Chem.MolFromSmiles(smiles)))
    assert rebuilt is not None
    assert sp.canonical(rebuilt) == sp.canonical(smiles)


@pytest.mark.parametrize("smiles", CASES)
def test_permutation_invariant(smiles):
    """Random atom orderings of the same molecule encode identically."""
    mol = Chem.MolFromSmiles(smiles)
    base = sp.encode(mol)
    for _ in range(16):
        shuffled = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
        assert sp.encode(Chem.MolFromSmiles(shuffled)) == base


@pytest.mark.parametrize("smiles", CASES)
def test_atom_order_matches_rdkit(smiles):
    """The traversal reproduces RDKit's _smilesAtomOutputOrder exactly."""
    mol = Chem.MolFromSmiles(smiles)
    events = sp.CanonicalTraversal().events(mol)
    order = [e.idx for e in events if isinstance(e, sp.Atom)]
    assert order == rdkit_atom_order(mol)


def test_empty_molecule():
    assert sp.encode(Chem.MolFromSmiles("")) == ""


def test_decode_invalid_returns_none():
    assert sp.decode("[CH5]") is None          # impossible valence
    assert sp.decode("[C]=1[C]1[C]1") is None  # malformed ring digits


def test_swappable_component_api():
    prog = sp.SmilesProgram(traversal=sp.CanonicalTraversal(),
                            atoms=sp.AtomCodec(), bonds=sp.BondCodec())
    for smiles in CASES:
        mol = Chem.MolFromSmiles(smiles)
        assert sp.canonical(prog.decode(prog.encode(mol))) \
            == sp.canonical(smiles)


def test_round_trip_on_random_sample():
    """A broader fuzz over randomized orderings of every case."""
    rng = random.Random(0)
    for smiles in CASES:
        mol = Chem.MolFromSmiles(smiles)
        for _ in range(5):
            variant = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            rebuilt = sp.decode(sp.encode(Chem.MolFromSmiles(variant)))
            assert rebuilt is not None
            assert sp.canonical(rebuilt) == sp.canonical(smiles)
