"""Lossless conversion between RDKit molecules and integer graphs.

Molecules are kekulized and stripped of stereochemistry. Each heavy atom
is keyed by ``(symbol, formal charge, total H count)`` so that the graph
fully determines the molecule (no valence inference on decoding).
"""

from typing import NamedTuple, Optional, Sequence

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

BOND_TYPES = (
    None,
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
)
NUM_BOND_TYPES = len(BOND_TYPES)
MAX_DISTANCE = 255

AtomKey = tuple[str, int, int]


class MolGraph(NamedTuple):
    """Integer graph of a molecule.

    Attributes
    ----------
    atoms : np.ndarray
        ``(N,)`` atom-vocabulary indices.
    bonds : np.ndarray
        ``(N, N)`` bond-type indices into ``BOND_TYPES``; 0 is no bond.
    distance : np.ndarray
        ``(N, N)`` topological distances, clipped to ``MAX_DISTANCE``.
    symmetry : np.ndarray
        ``(N,)`` symmetry-class labels; equal labels mark atoms that are
        (up to RDKit's canonical ranking) graph-equivalent.
    """

    atoms: np.ndarray
    bonds: np.ndarray
    distance: np.ndarray
    symmetry: np.ndarray


def parse(smiles: str) -> Optional[Chem.Mol]:
    """Parse, kekulize and strip stereo; ``None`` if invalid."""
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    return mol


def canonical(mol: Chem.Mol) -> str:
    """Canonical, stereo-free, aromatic SMILES (independent of the
    input's Kekule form)."""
    mol = Chem.Mol(mol)
    Chem.SanitizeMol(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)


def atom_key(atom: Chem.Atom) -> AtomKey:
    return atom.GetSymbol(), atom.GetFormalCharge(), atom.GetTotalNumHs()


def to_graph(mol: Chem.Mol, vocab: dict[AtomKey, int]) -> MolGraph:
    """Featurize a kekulized molecule.

    Raises
    ------
    KeyError
        If an atom key is not in ``vocab``.
    """
    n = mol.GetNumAtoms()
    atoms = np.array(
        [vocab[atom_key(a)] for a in mol.GetAtoms()], dtype=np.int64
    )
    bonds = np.zeros((n, n), dtype=np.int64)
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bonds[i, j] = bonds[j, i] = BOND_TYPES.index(b.GetBondType())
    distance = np.minimum(Chem.GetDistanceMatrix(mol), MAX_DISTANCE)
    symmetry = np.array(
        Chem.CanonicalRankAtoms(mol, breakTies=False), dtype=np.int64
    )
    return MolGraph(atoms, bonds, distance.astype(np.int64), symmetry)


def to_mol(
    atoms: np.ndarray, bonds: np.ndarray, keys: Sequence[AtomKey]
) -> Optional[Chem.Mol]:
    """Inverse of ``to_graph``; ``None`` if the graph is not chemical."""
    mol = Chem.RWMol()
    for idx in atoms:
        symbol, charge, num_hs = keys[int(idx)]
        atom = Chem.Atom(symbol)
        atom.SetFormalCharge(charge)
        atom.SetNumExplicitHs(num_hs)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for i, j in zip(*np.triu_indices(len(atoms), k=1)):
        if bonds[i, j]:
            mol.AddBond(int(i), int(j), BOND_TYPES[int(bonds[i, j])])
    try:
        Chem.SanitizeMol(mol)
    except (Chem.rdchem.MolSanitizeException, RuntimeError):
        return None
    return mol.GetMol()
