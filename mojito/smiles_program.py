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


# RDKit Canon.cpp traversal constants (Code/GraphMol/Canon.h)
_MAX_NATOMS = 5000
_MAX_BONDTYPE = 32
_MAX_CYCLES = 1024
_WHITE, _GREY, _BLACK = 0, 1, 2


def _traverse(m):
    """Faithful port of RDKit's Canon.cpp dfsFindCycles + dfsBuildStack.

    Two passes over the graph in canonical-rank order, with RDKit's
    rank-biasing so ring-closure neighbours are visited first, then
    non-ring, then ring atoms. Returns, per start fragment, the emit
    order: a list of events ('atom' a) / ('ring' digit, bondtype) /
    ('branch_open',) / ('branch_close',) / ('bond', bondtype) / ('dot',).
    Reproduces RDKit's exact ``_smilesAtomOutputOrder``.
    """
    import sys
    sys.setrecursionlimit(100000)
    n = m.GetNumAtoms()
    rank = list(Chem.CanonicalRankAtoms(m, breakTies=True))
    ri = m.GetRingInfo()

    def in_ring(bond):
        return ri.NumBondRings(bond.GetIdx()) > 0

    def biased(base, bond, grey):
        # Canon.cpp dfsFindCycles "black magic" neighbour ordering.
        bt = int(bond.GetBondType())
        if grey:
            base -= (_MAX_BONDTYPE + 1) * _MAX_NATOMS * _MAX_NATOMS
            base += (_MAX_BONDTYPE - bt) * _MAX_NATOMS
        elif in_ring(bond):
            base += (_MAX_BONDTYPE - bt) * _MAX_NATOMS * _MAX_NATOMS
        return base

    ring_closures = {a: [] for a in range(n)}

    def find_cycles(a, in_bond, tcolors):
        tcolors[a] = _GREY
        poss = []
        for bond in m.GetAtomWithIdx(a).GetBonds():
            if bond.GetIdx() == in_bond:
                continue
            o = bond.GetOtherAtomIdx(a)
            poss.append((biased(rank[o], bond, tcolors[o] == _GREY),
                         o, bond.GetIdx()))
        for _, o, bi in sorted(poss, key=lambda x: x[0]):
            if tcolors[o] == _WHITE:
                find_cycles(o, bi, tcolors)
            elif tcolors[o] == _GREY:  # back edge -> ring closure
                ring_closures[o].append(bi)
                ring_closures[a].append(bi)
        tcolors[a] = _BLACK

    colors = [_WHITE] * n
    events = []
    digit_of = {}
    available = list(range(1, _MAX_CYCLES + 1))

    def build_stack(a, in_bond):
        seen = {a}
        events.append(("atom", a))
        colors[a] = _GREY
        closed = []
        for bi in ring_closures[a]:
            bond = m.GetBondWithIdx(bi)
            seen.add(bond.GetOtherAtomIdx(a))
            if bi in digit_of:                       # closing end: bond+digit
                events.append(("ring_close", digit_of[bi], bi))
                closed.append(digit_of[bi])
            else:                                    # opening end: digit only
                d = available.pop(0)
                digit_of[bi] = d
                events.append(("ring_open", d))
        for d in closed:
            available.append(d)
            available.sort()
        poss = []
        for bond in m.GetAtomWithIdx(a).GetBonds():
            if bond.GetIdx() == in_bond:
                continue
            o = bond.GetOtherAtomIdx(a)
            if colors[o] != _WHITE or o in seen:
                continue
            base = rank[o]
            if in_ring(bond):
                base += (_MAX_BONDTYPE - int(bond.GetBondType())) \
                    * _MAX_NATOMS * _MAX_NATOMS
            poss.append((base, o, bond.GetIdx()))
        poss.sort(key=lambda x: x[0])
        for i, (_, o, bi) in enumerate(poss):
            if colors[o] != _WHITE:
                continue
            branch = i < len(poss) - 1
            if branch:
                events.append(("branch_open",))
            events.append(("bond", bi))
            build_stack(o, bi)
            if branch:
                events.append(("branch_close",))
        colors[a] = _BLACK

    first = True
    while len(digit_of) + sum(1 for c in colors if c != _WHITE) < n \
            or any(c == _WHITE for c in colors):
        white = [a for a in range(n) if colors[a] == _WHITE]
        if not white:
            break
        start = min(white, key=lambda a: rank[a])
        if not first:
            events.append(("dot",))
        first = False
        find_cycles(start, -1, [_WHITE] * n)
        build_stack(start, -1)
    return events


def encode(mol) -> Optional[str]:
    """Molecular graph -> SMILES-dialect string.

    The traversal runs on the aromatic molecule (as RDKit does, so the
    atom order is RDKit-canonical); bond symbols are read from a Kekule
    copy by bond index for the explicit dialect.
    """
    arom = Chem.Mol(mol)
    Chem.RemoveStereochemistry(arom)
    Chem.SanitizeMol(arom)
    if arom.GetNumAtoms() == 0:
        return ""
    kek = Chem.Mol(arom)
    Chem.Kekulize(kek, clearAromaticFlags=True)

    def bond_symbol(bi):
        return _BOND_SYMBOL[kek.GetBondWithIdx(bi).GetBondType()]

    out = []
    for ev in _traverse(arom):
        if ev[0] == "atom":
            out.append(_atom_token(kek.GetAtomWithIdx(ev[1])))
        elif ev[0] == "ring_open":
            out.append(_ring_digit(ev[1]))
        elif ev[0] == "ring_close":
            out.append(bond_symbol(ev[2]) + _ring_digit(ev[1]))
        elif ev[0] == "bond":
            out.append(bond_symbol(ev[1]))
        elif ev[0] == "branch_open":
            out.append("(")
        elif ev[0] == "branch_close":
            out.append(")")
        elif ev[0] == "dot":
            out.append(".")
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
