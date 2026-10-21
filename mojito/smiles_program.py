"""SMILES as an explicit, editable program.

A transparent encoder (molecular graph -> string) and decoder (string ->
molecular graph), using RDKit only for the graph (atoms, bonds,
canonical ranks), never for the string. This makes the grammar's actual
decisions -- traversal order, ring-closure numbering, branch notation,
atom spelling -- visible as code an LLM can mutate, instead of hidden
inside ``MolToSmiles``.

The default here is a deliberately simple, fully-explicit dialect:
kekulized (no aromatic lowercase), stereo dropped, every atom bracketed
as ``[<symbol><Hn><charge>]``. It is verbose on purpose -- shortening
(organic-subset shorthand, implicit hydrogens) is a *rule* the search
can add, not a baked-in assumption. Lossless and permutation-invariant
(canonical-rank traversal).
"""

from typing import Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_BOND_SYMBOL = {
    Chem.BondType.SINGLE: "", Chem.BondType.DOUBLE: "=",
    Chem.BondType.TRIPLE: "#",
}
_SYMBOL_BOND = {"": Chem.BondType.SINGLE, "-": Chem.BondType.SINGLE,
                "=": Chem.BondType.DOUBLE, "#": Chem.BondType.TRIPLE}


def _atom_token(atom: Chem.Atom) -> str:
    sym = atom.GetSymbol()
    nh = atom.GetTotalNumHs()
    h = f"H{nh}" if nh > 1 else ("H" if nh == 1 else "")
    c = atom.GetFormalCharge()
    charge = "" if c == 0 else ("+" if c > 0 else "-") + (
        str(abs(c)) if abs(c) > 1 else "")
    return f"[{sym}{h}{charge}]"


def _ring_digit(n: int) -> str:
    return str(n) if n < 10 else f"%{n:02d}"


def encode(mol) -> Optional[str]:
    """Molecular graph -> SMILES-dialect string via canonical DFS."""
    m = Chem.Mol(mol)
    Chem.RemoveStereochemistry(m)
    Chem.Kekulize(m, clearAromaticFlags=True)
    n = m.GetNumAtoms()
    if n == 0:
        return ""
    rank = list(Chem.CanonicalRankAtoms(m, breakTies=True))
    order = sorted(range(n), key=lambda a: rank[a])

    # classify edges by DFS: tree edges vs ring-closure (back) edges
    adj = {a: sorted((b.GetOtherAtomIdx(a) for b in
                      m.GetAtomWithIdx(a).GetBonds()), key=lambda x: rank[x])
           for a in range(n)}
    visited, parent, closures = set(), {}, {a: [] for a in range(n)}
    next_digit = [1]

    import sys
    sys.setrecursionlimit(10000)
    used = set()  # undirected edges already placed (tree or ring)

    def classify(a, par):
        """Recursive DFS: each edge is used once -- as a tree edge or,
        if it closes back to an ancestor, as a ring closure."""
        visited.add(a)
        parent[a] = par
        for b in adj[a]:
            e = frozenset((a, b))
            if e in used:
                continue
            used.add(e)
            if b not in visited:
                classify(b, a)
            else:  # back edge -> ring closure
                d = next_digit[0]
                next_digit[0] += 1
                bond = m.GetBondBetweenAtoms(a, b).GetBondType()
                closures[a].append((d, bond))
                closures[b].append((d, bond))

    for a in order:
        if a not in visited:
            classify(a, -1)

    # recursive emit in canonical child order
    out = []

    def emit(a):
        out.append(_atom_token(m.GetAtomWithIdx(a)))
        for d, bond in closures[a]:
            out.append(_BOND_SYMBOL[bond] + _ring_digit(d))
        children = [b for b in adj[a]
                    if parent.get(b) == a and b != parent[a]]
        for i, b in enumerate(children):
            bt = m.GetBondBetweenAtoms(a, b).GetBondType()
            branch = i < len(children) - 1
            out.append("(" if branch else "")
            out.append(_BOND_SYMBOL[bt])
            emit(b)
            out.append(")" if branch else "")

    roots = [a for a in order if parent.get(a) == -1]
    for i, r in enumerate(roots):
        if i:
            out.append(".")
        emit(r)
    return "".join(out)


def decode(text: str) -> Optional[Chem.Mol]:
    """SMILES-dialect string -> molecular graph."""
    import re
    tok = re.findall(r"\[[^\]]+\]|%\d\d|[0-9]|[=#()./-]", text)
    mol = Chem.RWMol()
    stack, prev, bond, open_ring = [], None, "", {}
    try:
        for t in tok:
            if t == "(":
                stack.append(prev)
            elif t == ")":
                prev = stack.pop()
            elif t == ".":
                prev = None
            elif t in ("=", "#", "-"):
                bond = t
            elif t[0] == "[":
                inner = t[1:-1]
                sym = re.match(r"[A-Z][a-z]?", inner).group()
                rest = inner[len(sym):]
                hm = re.search(r"H(\d*)", rest)
                nh = (int(hm.group(1)) if hm.group(1) else 1) if hm else 0
                cm = re.search(r"([+-])(\d*)", rest)
                ch = 0
                if cm:
                    ch = (1 if cm.group(1) == "+" else -1) * (
                        int(cm.group(2)) if cm.group(2) else 1)
                atom = Chem.Atom(sym)
                atom.SetNoImplicit(True)
                atom.SetNumExplicitHs(nh)
                atom.SetFormalCharge(ch)
                idx = mol.AddAtom(atom)
                if prev is not None:
                    mol.AddBond(prev, idx, _SYMBOL_BOND[bond])
                prev, bond = idx, ""
            else:  # ring digit
                d = int(t[1:]) if t[0] == "%" else int(t)
                if d in open_ring:
                    a0, b0 = open_ring.pop(d)
                    mol.AddBond(a0, prev, _SYMBOL_BOND[b0 or bond])
                else:
                    open_ring[d] = (prev, bond)
                bond = ""
        m = mol.GetMol()
        Chem.SanitizeMol(m)
        return m
    except Exception:
        return None


def canonical(mol_or_smiles) -> Optional[str]:
    m = (Chem.MolFromSmiles(mol_or_smiles)
         if isinstance(mol_or_smiles, str) else mol_or_smiles)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None
