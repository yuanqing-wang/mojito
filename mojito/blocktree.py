"""Canonical, lossless block-cut tree tokenizer for molecules.

A molecule's 2D graph is decomposed into its **block-cut tree**: the
biconnected components (ring systems and single acyclic bonds) joined at
articulation atoms. The tree is unique, so no spanning-tree choice is
made (unlike junction-tree methods). Each block is stored as a canonical
fragment; each articulation is stored once; the gluing between them is
recorded by the fragment-local orbit (canonical rank) of the shared
atom, never a raw atom index. Hence the representation is

* **permutation-invariant** -- nothing depends on input atom order;
* **lossless** -- the original molecule is recovered exactly;
* **compact** -- one unit per ring system or chain atom (~11 on ZINC),
  not one per atom.

This module is the symbolic backbone. A learned codebook over block
fragments (discretized GNN embeddings) can replace the fragment string
with an integer, with the string kept as a fallback, without affecting
losslessness: identity lives in the tree and the orbits, not the code.
"""

from typing import NamedTuple, Optional, Union

import networkx as nx
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

Mollike = Union[str, Chem.Mol]


class Block(NamedTuple):
    """One biconnected component.

    Attributes
    ----------
    frag : str
        Canonical kekulized SMILES of the isolated component.
    cuts : tuple[int, ...]
        Articulation-atom ids (molecule-level) in this block, each tagged
        in ``frag`` by atom-map number ``id + 1``.
    """

    frag: str
    cuts: tuple[int, ...]


class Decomposition(NamedTuple):
    """Block-cut decomposition of one molecule.

    Attributes
    ----------
    blocks : tuple[Block, ...]
        Biconnected components.
    cut_atoms : dict[int, tuple[str, int, int]]
        For each articulation atom: (symbol, formal charge, total Hs).
        Its hydrogen count is molecule-level, not visible inside any
        single fragment, so it is stored once here.
    """

    blocks: tuple[Block, ...]
    cut_atoms: dict


def _parse(mol: Mollike) -> Optional[Chem.Mol]:
    m = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
    if m is None:
        return None
    m = Chem.Mol(m)
    Chem.RemoveStereochemistry(m)
    Chem.Kekulize(m, clearAromaticFlags=True)
    return m


def _graph(m: Chem.Mol) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(m.GetNumAtoms()))
    for b in m.GetBonds():
        g.add_edge(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
    return g


def decompose(mol: Mollike) -> Optional[Decomposition]:
    """Decompose a molecule into its canonical block-cut representation.

    Parameters
    ----------
    mol : str or rdkit.Chem.Mol
        SMILES or molecule; stereochemistry is dropped.

    Returns
    -------
    Decomposition or None
        ``None`` if the SMILES does not parse.
    """
    m = _parse(mol)
    if m is None:
        return None
    g = _graph(m)
    cuts = set(nx.articulation_points(g))
    cut_atoms = {
        a: (
            m.GetAtomWithIdx(a).GetSymbol(),
            m.GetAtomWithIdx(a).GetFormalCharge(),
            m.GetAtomWithIdx(a).GetTotalNumHs(),
        )
        for a in cuts
    }
    blocks = []
    for atoms in (set(c) for c in nx.biconnected_components(g)):
        bonds = [
            b.GetIdx()
            for b in m.GetBonds()
            if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms
        ]
        amap: dict = {}
        sub = Chem.PathToSubmol(m, bonds, atomMap=amap)  # molidx -> subidx
        block_cuts = tuple(sorted(a for a in atoms if a in cuts))
        for a in block_cuts:
            sub.GetAtomWithIdx(amap[a]).SetAtomMapNum(a + 1)
        frag = Chem.MolToSmiles(sub, kekuleSmiles=True, canonical=True)
        blocks.append(Block(frag, block_cuts))
    return Decomposition(tuple(blocks), cut_atoms)


def canonical_decompose(mol: Mollike) -> Optional[Decomposition]:
    """Decompose with order-invariant articulation ids.

    Articulation atoms are tagged by their rank in the molecule's
    canonical atom order (invariant to input atom numbering), and blocks
    are sorted canonically. Two different atom orderings of the same
    molecule therefore give byte-identical output.
    """
    m = _parse(mol)
    if m is None:
        return None
    rank = list(Chem.CanonicalRankAtoms(m, breakTies=True))
    g = _graph(m)
    cuts = set(nx.articulation_points(g))
    cut_atoms = {
        rank[a]: (
            m.GetAtomWithIdx(a).GetSymbol(),
            m.GetAtomWithIdx(a).GetFormalCharge(),
            m.GetAtomWithIdx(a).GetTotalNumHs(),
        )
        for a in cuts
    }
    blocks = []
    for atoms in (set(c) for c in nx.biconnected_components(g)):
        bonds = [
            b.GetIdx()
            for b in m.GetBonds()
            if b.GetBeginAtomIdx() in atoms and b.GetEndAtomIdx() in atoms
        ]
        amap: dict = {}
        sub = Chem.PathToSubmol(m, bonds, atomMap=amap)
        block_cuts = tuple(sorted(rank[a] for a in atoms if a in cuts))
        for a in atoms:
            if a in cuts:
                sub.GetAtomWithIdx(amap[a]).SetAtomMapNum(rank[a] + 1)
        frag = Chem.MolToSmiles(sub, kekuleSmiles=True, canonical=True)
        blocks.append(Block(frag, block_cuts))
    blocks.sort(key=lambda b: (b.cuts, b.frag))
    return Decomposition(tuple(blocks), cut_atoms)


def reconstruct(dec: Decomposition) -> Optional[Chem.Mol]:
    """Rebuild the molecule from a decomposition.

    Blocks are fused at shared articulation atoms, identified by the
    atom-map tags carried in the fragment SMILES. Hydrogen counts of
    articulation atoms are restored from ``cut_atoms``.

    Returns
    -------
    rdkit.Chem.Mol or None
        ``None`` if the fused graph is not a valid molecule.
    """
    out = Chem.RWMol()
    node: dict = {}  # articulation id -> out atom idx
    for frag, _ in dec.blocks:
        sub = Chem.MolFromSmiles(frag, sanitize=False)
        if sub is None:
            return None
        sub.UpdatePropertyCache(strict=False)
        local = {}
        for a in sub.GetAtoms():
            tag = a.GetAtomMapNum()
            if tag and tag in node:
                local[a.GetIdx()] = node[tag]
                continue
            na = Chem.Atom(a.GetAtomicNum())
            na.SetFormalCharge(a.GetFormalCharge())
            na.SetNoImplicit(True)
            na.SetNumExplicitHs(a.GetTotalNumHs())
            idx = out.AddAtom(na)
            local[a.GetIdx()] = idx
            if tag:
                node[tag] = idx
        for b in sub.GetBonds():
            i = local[b.GetBeginAtomIdx()]
            j = local[b.GetEndAtomIdx()]
            if out.GetBondBetweenAtoms(i, j) is None:
                out.AddBond(i, j, b.GetBondType())
    for tag, idx in node.items():
        _, _, num_hs = dec.cut_atoms[tag - 1]
        out.GetAtomWithIdx(idx).SetNumExplicitHs(num_hs)
    mol = out.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except (Chem.rdchem.MolSanitizeException, RuntimeError, ValueError):
        return None
    return mol


def canonical_smiles(mol: Mollike) -> Optional[str]:
    """Stereo-free canonical SMILES, or ``None`` if parsing fails."""
    m = Chem.MolFromSmiles(mol) if isinstance(mol, str) else mol
    if m is None:
        return None
    return Chem.MolToSmiles(m, isomericSmiles=False)
