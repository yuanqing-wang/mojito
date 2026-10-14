"""Canonical, reversible string serialization of the block-cut tree.

A molecule becomes a dot-joined list of canonical block fragments. Each
articulation atom appears in the >= 2 blocks that share it, tagged by a
small **attachment id** reused across those occurrences (like SMILES
ring-bond digits, or SAFE's fragment attachments). Attachment ids are
numbered by first appearance in the canonical block order, so the string
is permutation-invariant; the fragment list reconstructs the molecule
exactly (cut-atom hydrogens are inferred from valence on fusing).

This is the symbol stream that a BPE vocabulary is learned over
(``scripts/bpe_compare.py``): block granularity means BPE merges whole
chemical motifs, not characters.
"""

import re
from typing import Optional

from rdkit import Chem, RDLogger

from .blocktree import Mollike, canonical_decompose

RDLogger.DisableLog("rdApp.*")

_SEP = " . "
_ATOM = re.compile(
    r"\[[^\]]+\]|Br?|Cl?|[BCNOPSFIbcnops]|[0-9]|[=#()/\\+\-@%]"
)


def to_string(mol: Mollike) -> Optional[str]:
    """Serialize ``mol`` to its canonical block string, or ``None``."""
    dec = canonical_decompose(mol)
    if dec is None:
        return None
    relabel: dict = {}
    frags = []
    for frag, cuts in dec.blocks:
        def repl(m, _relabel=relabel):
            old = int(m.group(1))
            _relabel.setdefault(old, len(_relabel))
            return f":{_relabel[old] + 1}]"
        frags.append(re.sub(r":(\d+)\]", repl, frag))
    return _SEP.join(frags)


def from_string(text: str) -> Optional[Chem.Mol]:
    """Rebuild the molecule from a block string, or ``None``."""
    out = Chem.RWMol()
    node: dict = {}
    for frag in text.split(_SEP):
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
            if not tag:  # interior atom: freeze its hydrogens
                na.SetNoImplicit(True)
                na.SetNumExplicitHs(a.GetTotalNumHs())
            idx = out.AddAtom(na)
            local[a.GetIdx()] = idx
            if tag:
                node[tag] = idx  # cut atom: infer Hs from valence later
        for b in sub.GetBonds():
            i, j = local[b.GetBeginAtomIdx()], local[b.GetEndAtomIdx()]
            if out.GetBondBetweenAtoms(i, j) is None:
                out.AddBond(i, j, b.GetBondType())
    mol = out.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except (Chem.rdchem.MolSanitizeException, RuntimeError, ValueError):
        return None
    return mol


def tokens(text: str) -> list:
    """Split a block string into atomic symbols for BPE.

    Fragment separators and the SMILES atoms/bonds/digits are the base
    units; BPE then merges frequent adjacent pairs.
    """
    out = []
    for i, frag in enumerate(text.split(_SEP)):
        if i:
            out.append(".")
        out.extend(_ATOM.findall(frag))
    return out
