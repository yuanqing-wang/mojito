"""SMILES as an explicit, editable program.

A transparent encoder (molecular graph -> string) and decoder (string ->
molecular graph) that uses RDKit only for the molecular graph (atoms,
bonds, canonical ranks), never ``MolToSmiles`` / ``MolFromSmiles`` for
the string. The grammar's decisions -- traversal order, ring-closure
numbering, branch notation, atom and bond spelling -- are visible as code
instead of hidden inside RDKit.

The design is modular so each decision is swappable:

* :class:`CanonicalTraversal` -- a faithful port of RDKit's canonical
  SMILES traversal (``Code/GraphMol/Canon.cpp``: ``dfsFindCycles`` +
  ``dfsBuildStack``). It emits an abstract :data:`Event` stream and
  reproduces RDKit's exact ``_smilesAtomOutputOrder``.
* :class:`AtomCodec` -- atom <-> token (the atom-spelling dialect).
* :class:`BondCodec` -- bond <-> symbol.
* :class:`SmilesProgram` -- wires a traversal and the two codecs into
  ``encode`` / ``decode``.

The default dialect is deliberately explicit: kekulized (no aromatic
lowercase), stereo dropped, every atom bracketed as
``[<symbol><Hn><charge>]``. It is verbose on purpose, so shortening
(organic-subset shorthand, implicit hydrogens) is a swap, not a baked-in
assumption. Lossless and permutation-invariant.
"""

import re
import sys
from typing import List, NamedTuple, Optional, Union

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------------------
# Event stream: the abstract output of a traversal, rendered by a codec.
# --------------------------------------------------------------------------

class Atom(NamedTuple):
    idx: int


class Bond(NamedTuple):
    bond_idx: int


class RingOpen(NamedTuple):
    digit: int


class RingClose(NamedTuple):
    digit: int
    bond_idx: int


class BranchOpen(NamedTuple):
    pass


class BranchClose(NamedTuple):
    pass


class Dot(NamedTuple):
    pass


Event = Union[Atom, Bond, RingOpen, RingClose, BranchOpen, BranchClose, Dot]


# --------------------------------------------------------------------------
# Bond codec: bond type <-> symbol.
# --------------------------------------------------------------------------

class BondCodec:
    """Translate bond types to/from SMILES bond symbols."""

    TO_SYMBOL = {
        Chem.BondType.SINGLE: "", Chem.BondType.DOUBLE: "=",
        Chem.BondType.TRIPLE: "#",
    }
    FROM_SYMBOL = {
        "": Chem.BondType.SINGLE, "-": Chem.BondType.SINGLE,
        "=": Chem.BondType.DOUBLE, "#": Chem.BondType.TRIPLE,
    }

    def symbol(self, bond: Chem.Bond) -> str:
        return self.TO_SYMBOL[bond.GetBondType()]

    def bond_type(self, symbol: str) -> Chem.BondType:
        return self.FROM_SYMBOL[symbol]


# --------------------------------------------------------------------------
# Atom codec: atom <-> token. Swap this to change the spelling dialect.
# --------------------------------------------------------------------------

class AtomCodec:
    """Fully-explicit bracket atoms: ``[<symbol><Hn><charge>]``."""

    _ATOM = re.compile(r"\[([A-Z][a-z]?)(H\d*)?([+-]\d*)?\]")

    def token(self, atom: Chem.Atom) -> str:
        nh = atom.GetTotalNumHs()
        h = f"H{nh}" if nh > 1 else ("H" if nh == 1 else "")
        c = atom.GetFormalCharge()
        charge = "" if c == 0 else (
            ("+" if c > 0 else "-") + (str(abs(c)) if abs(c) > 1 else ""))
        return f"[{atom.GetSymbol()}{h}{charge}]"

    def build(self, token: str) -> Chem.Atom:
        m = self._ATOM.fullmatch(token)
        if m is None:
            raise ValueError(f"bad atom token: {token}")
        sym, h, ch = m.groups()
        nh = (int(h[1:]) if len(h) > 1 else 1) if h else 0
        charge = 0
        if ch:
            charge = (1 if ch[0] == "+" else -1) * (
                int(ch[1:]) if len(ch) > 1 else 1)
        atom = Chem.Atom(sym)
        atom.SetNoImplicit(True)
        atom.SetNumExplicitHs(nh)
        atom.SetFormalCharge(charge)
        return atom


# --------------------------------------------------------------------------
# Traversal: graph -> event stream. Swap this to change traversal policy.
# --------------------------------------------------------------------------

class CanonicalTraversal:
    """Port of RDKit's canonical SMILES traversal (Canon.cpp).

    Two passes over the graph: ``_find_cycles`` locates ring-closure
    (back) edges, then ``_build_stack`` emits the event stream assigning
    ring digits. Neighbours are visited in canonical-rank order, biased
    (RDKit's "black magic") so ring-closure atoms come first, then
    non-ring atoms, then ring atoms. Start atom is the lowest canonical
    rank in each fragment. Reproduces RDKit's ``_smilesAtomOutputOrder``.
    """

    # Canon.h constants.
    MAX_NATOMS = 5000
    MAX_BONDTYPE = 32
    MAX_CYCLES = 1024
    _WHITE, _GREY, _BLACK = 0, 1, 2

    def events(self, mol: Chem.Mol) -> List[Event]:
        sys.setrecursionlimit(100000)
        self._mol = mol
        self._rank = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
        self._ring_info = mol.GetRingInfo()
        n = mol.GetNumAtoms()
        self._closures = {a: [] for a in range(n)}
        self._colors = [self._WHITE] * n
        self._digit_of = {}
        self._available = list(range(1, self.MAX_CYCLES + 1))
        out: List[Event] = []
        first = True
        while any(c == self._WHITE for c in self._colors):
            start = min((a for a in range(n)
                         if self._colors[a] == self._WHITE),
                        key=lambda a: self._rank[a])
            if not first:
                out.append(Dot())
            first = False
            self._find_cycles(start, -1, [self._WHITE] * n)
            self._build_stack(start, -1, out)
        return out

    def _in_ring(self, bond: Chem.Bond) -> bool:
        return self._ring_info.NumBondRings(bond.GetIdx()) > 0

    def _biased_rank(self, base: int, bond: Chem.Bond, grey: bool) -> int:
        bt = int(bond.GetBondType())
        if grey:  # ring-closure neighbour -> visit first
            base -= (self.MAX_BONDTYPE + 1) * self.MAX_NATOMS ** 2
            base += (self.MAX_BONDTYPE - bt) * self.MAX_NATOMS
        elif self._in_ring(bond):  # ring neighbour -> visit last
            base += (self.MAX_BONDTYPE - bt) * self.MAX_NATOMS ** 2
        return base

    def _find_cycles(self, a: int, in_bond: int, colors: list) -> None:
        colors[a] = self._GREY
        poss = []
        for bond in self._mol.GetAtomWithIdx(a).GetBonds():
            if bond.GetIdx() == in_bond:
                continue
            o = bond.GetOtherAtomIdx(a)
            poss.append((self._biased_rank(self._rank[o], bond,
                                           colors[o] == self._GREY),
                         o, bond.GetIdx()))
        for _, o, bi in sorted(poss, key=lambda x: x[0]):
            if colors[o] == self._WHITE:
                self._find_cycles(o, bi, colors)
            elif colors[o] == self._GREY:  # back edge -> ring closure
                self._closures[o].append(bi)
                self._closures[a].append(bi)
        colors[a] = self._BLACK

    def _build_stack(self, a: int, in_bond: int, out: List[Event]) -> None:
        seen = {a}
        out.append(Atom(a))
        self._colors[a] = self._GREY
        closed = []
        for bi in self._closures[a]:
            bond = self._mol.GetBondWithIdx(bi)
            seen.add(bond.GetOtherAtomIdx(a))
            if bi in self._digit_of:                 # closing end
                out.append(RingClose(self._digit_of[bi], bi))
                closed.append(self._digit_of[bi])
            else:                                    # opening end
                d = self._available.pop(0)
                self._digit_of[bi] = d
                out.append(RingOpen(d))
        for d in closed:                             # free closed digits
            self._available.append(d)
            self._available.sort()
        poss = []
        for bond in self._mol.GetAtomWithIdx(a).GetBonds():
            if bond.GetIdx() == in_bond:
                continue
            o = bond.GetOtherAtomIdx(a)
            if self._colors[o] != self._WHITE or o in seen:
                continue
            base = self._rank[o]
            if self._in_ring(bond):
                base += (self.MAX_BONDTYPE - int(bond.GetBondType())) \
                    * self.MAX_NATOMS ** 2
            poss.append((base, o, bond.GetIdx()))
        poss.sort(key=lambda x: x[0])
        for i, (_, o, bi) in enumerate(poss):
            if self._colors[o] != self._WHITE:
                continue
            branch = i < len(poss) - 1
            if branch:
                out.append(BranchOpen())
            out.append(Bond(bi))
            self._build_stack(o, bi, out)
            if branch:
                out.append(BranchClose())
        self._colors[a] = self._BLACK


# --------------------------------------------------------------------------
# Program: wire a traversal + codecs into encode / decode.
# --------------------------------------------------------------------------

class SmilesProgram:
    """A reversible SMILES-dialect codec built from swappable parts."""

    _TOKEN = re.compile(r"\[[^\]]+\]|%\d\d|[0-9]|[=#()./-]")

    def __init__(self, traversal=None, atoms=None, bonds=None):
        self.traversal = traversal or CanonicalTraversal()
        self.atoms = atoms or AtomCodec()
        self.bonds = bonds or BondCodec()

    def encode(self, mol: Chem.Mol) -> Optional[str]:
        """Molecular graph -> string.

        The traversal runs on the aromatic molecule (RDKit-canonical
        order); bond symbols are read from a Kekule copy by bond index.
        """
        arom = Chem.Mol(mol)
        Chem.RemoveStereochemistry(arom)
        Chem.SanitizeMol(arom)
        if arom.GetNumAtoms() == 0:
            return ""
        kek = Chem.Mol(arom)
        Chem.Kekulize(kek, clearAromaticFlags=True)
        out = []
        for ev in self.traversal.events(arom):
            if isinstance(ev, Atom):
                out.append(self.atoms.token(kek.GetAtomWithIdx(ev.idx)))
            elif isinstance(ev, RingOpen):
                out.append(_ring_digit(ev.digit))
            elif isinstance(ev, RingClose):
                sym = self.bonds.symbol(kek.GetBondWithIdx(ev.bond_idx))
                out.append(sym + _ring_digit(ev.digit))
            elif isinstance(ev, Bond):
                out.append(self.bonds.symbol(kek.GetBondWithIdx(ev.bond_idx)))
            elif isinstance(ev, BranchOpen):
                out.append("(")
            elif isinstance(ev, BranchClose):
                out.append(")")
            elif isinstance(ev, Dot):
                out.append(".")
        return "".join(out)

    def decode(self, text: str) -> Optional[Chem.Mol]:
        """String -> molecular graph (``None`` if invalid)."""
        mol = Chem.RWMol()
        stack, prev, bond, open_ring = [], None, "", {}
        try:
            for t in self._TOKEN.findall(text):
                if t == "(":
                    stack.append(prev)
                elif t == ")":
                    prev = stack.pop()
                elif t == ".":
                    prev = None
                elif t in ("=", "#", "-"):
                    bond = t
                elif t[0] == "[":
                    idx = mol.AddAtom(self.atoms.build(t))
                    if prev is not None:
                        mol.AddBond(prev, idx, self.bonds.bond_type(bond))
                    prev, bond = idx, ""
                else:  # ring digit
                    d = int(t[1:]) if t[0] == "%" else int(t)
                    if d in open_ring:
                        a0, b0 = open_ring.pop(d)
                        mol.AddBond(a0, prev,
                                    self.bonds.bond_type(b0 or bond))
                    else:
                        open_ring[d] = (prev, bond)
                    bond = ""
            m = mol.GetMol()
            Chem.SanitizeMol(m)
            return m
        except Exception:
            return None


def _ring_digit(n: int) -> str:
    return str(n) if n < 10 else f"%{n:02d}"


# --------------------------------------------------------------------------
# Module-level default API.
# --------------------------------------------------------------------------

_DEFAULT = SmilesProgram()


def encode(mol: Chem.Mol) -> Optional[str]:
    """Encode a molecule with the default dialect."""
    return _DEFAULT.encode(mol)


def decode(text: str) -> Optional[Chem.Mol]:
    """Decode a string with the default dialect."""
    return _DEFAULT.decode(text)


def canonical(mol_or_smiles) -> Optional[str]:
    """Stereo-free canonical SMILES, for round-trip comparison."""
    m = (Chem.MolFromSmiles(mol_or_smiles)
         if isinstance(mol_or_smiles, str) else mol_or_smiles)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None
