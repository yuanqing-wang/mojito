# SMILES as an explicit program

`mojito/smiles_program.py` implements SMILES as a transparent
encoder/decoder, using RDKit only for the molecular graph (atoms, bonds,
canonical ranks) -- never `MolToSmiles`/`MolFromSmiles` for the string.
It is a **faithful port of RDKit's own algorithm** (`Code/GraphMol/
Canon.cpp`: `dfsFindCycles` + `dfsBuildStack`), so it reproduces RDKit's
exact canonical atom order -- verified 100% against RDKit's
`_smilesAtomOutputOrder` (`scripts/verify_smiles_traversal.py`). The
grammar's real decisions are now editable code:

- **traversal:** two-pass three-colour DFS from the lowest-canonical-rank
  root, with RDKit's rank-biasing (constants MAX_NATOMS=5000,
  MAX_BONDTYPE=32): ring-closure neighbours visited first, then non-ring,
  then ring atoms;
- **ring closures:** back-edges get the lowest available digit, reused as
  rings close (`%nn` past 9), bond symbol emitted at the closing end;
- **branches:** `( )` around all but the last child;
- **atoms:** fully-explicit `[<symbol><Hn><charge>]`, kekulized, stereo
  dropped -- verbose on purpose, so shortening is a *rule*, not a baked
  assumption;
- **bonds:** `=`/`#`, single implicit.

## Verified (10k held-out ZINC250k)
- atom order vs RDKit `_smilesAtomOutputOrder`: **100.00%** (faithful port)
- round-trip (decode(encode(m)) == m, canonical): **100.00%**
- permutation-invariant (3 random atom orders each): **100.00%**
- length: 43.8 program tokens/mol vs 38.2 for RDKit SMILES -- the ~15%
  overhead is the always-bracket, explicit-H dialect.

## Why this matters for rigour
Now the grammar *is* a program we control, not RDKit's black box:
- its **length is the MDL model term** (Kolmogorov complexity made
  concrete); its **output length is the corpus term**;
- losslessness is *checked by running it*, so an LLM may rewrite any part
  (traversal rule, ring numbering, atom spelling) and the round-trip gate
  certifies the result;
- the 15% dialect overhead is itself a target: organic-subset shorthand
  and implicit hydrogens are reversible rules the search can add.

This turns grammar design into genuine program synthesis over a verified
encoder/decoder, the honest FunSearch setup -- the next step is letting
the proposer edit *this program*, not post-process SMILES strings.
