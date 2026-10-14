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


def ringcollapse_tokens(mol: Mollike) -> Optional[list]:
    """Ring-collapsed token list: each ring system is one token, acyclic
    atoms keep SMILES' implicit linear connectivity.

    Invariant: ring systems become isotope-marked dummy atoms so RDKit's
    canonical SMILES orders them deterministically; ring identity is the
    canonical ring SMILES; each exocyclic bond carries the attachment
    atom's orbit (canonical rank within its ring). Lossless accounting
    only -- reconstruction is future work; this measures token count.
    """
    import networkx as nx
    from .blocktree import _graph, _parse
    m = _parse(mol)
    if m is None:
        return None
    g = _graph(m)
    comps = [set(c) for c in nx.biconnected_components(g) if len(c) > 2]
    ring_id, node_of, rw, nid = {}, {}, Chem.RWMol(), {}
    ranks = list(Chem.CanonicalRankAtoms(m, breakTies=False))
    for i, c in enumerate(comps):
        bonds = [b.GetIdx() for b in m.GetBonds()
                 if b.GetBeginAtomIdx() in c and b.GetEndAtomIdx() in c]
        sub = Chem.PathToSubmol(m, bonds)
        ring_id[i] = "R:" + Chem.MolToSmiles(sub, kekuleSmiles=True)
        d = Chem.Atom(0)
        d.SetIsotope(i + 1)
        nid[("R", i)] = rw.AddAtom(d)
        for a in c:
            node_of[a] = ("R", i)
    for a in range(m.GetNumAtoms()):
        if a not in node_of:
            at = m.GetAtomWithIdx(a)
            na = Chem.Atom(at.GetAtomicNum())
            na.SetFormalCharge(at.GetFormalCharge())
            nid[("A", a)] = rw.AddAtom(na)
            node_of[a] = ("A", a)
    seen, orbit_toks = set(), []
    for b in m.GetBonds():
        u, v = node_of[b.GetBeginAtomIdx()], node_of[b.GetEndAtomIdx()]
        if u == v:
            continue
        key = tuple(sorted((nid[u], nid[v])))
        if key in seen:
            continue
        seen.add(key)
        rw.AddBond(nid[u], nid[v], b.GetBondType())
        for end, atom in ((u, b.GetBeginAtomIdx()), (v, b.GetEndAtomIdx())):
            if end[0] == "R":
                orbit_toks.append(f"o{ranks[atom] % 32}")
    red = Chem.MolToSmiles(rw.GetMol(), canonical=True)
    out = []
    for t in _ATOM.findall(red):
        m_iso = re.fullmatch(r"\[(\d+)\*\]", t)
        out.append(ring_id[int(m_iso.group(1)) - 1] if m_iso else t)
    return out + orbit_toks


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
