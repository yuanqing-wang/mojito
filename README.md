# smiles-program

SMILES as an explicit, editable Python program: a transparent
encoder (molecular graph → string) and decoder (string → molecular
graph), using RDKit only for the graph (atoms, bonds, canonical ranks),
never `MolToSmiles` / `MolFromSmiles` for the string.

The traversal is a faithful port of RDKit's canonical SMILES algorithm
(`Code/GraphMol/Canon.cpp`: `dfsFindCycles` + `dfsBuildStack`) — a
two-pass, three-colour DFS with RDKit's rank-biasing (ring-closure
neighbours first, then non-ring, then ring atoms). It reproduces RDKit's
exact canonical atom order (verified 100% against
`_smilesAtomOutputOrder` on held-out ZINC250k), with 100% round-trip and
100% permutation-invariance.

The output dialect is deliberately explicit — kekulized, stereo dropped,
every atom bracketed as `[<symbol><Hn><charge>]` — so that shortening
(organic-subset shorthand, implicit hydrogens) is a *rule* one can add,
not a baked-in assumption.

## Usage
```python
from rdkit import Chem
from smiles_program import encode, decode, canonical

mol = Chem.MolFromSmiles("CC(C)Cc1ccccc1")
s = encode(mol)                       # graph -> string
back = decode(s)                      # string -> graph
assert canonical(back) == canonical(mol)
```

Requires `rdkit`.
