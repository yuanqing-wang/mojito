# SMILES as an explicit program

`mojito/smiles_program.py` implements SMILES as a transparent
encoder/decoder, using RDKit only for the molecular graph (atoms, bonds,
canonical ranks) -- never `MolToSmiles`/`MolFromSmiles` for the string.
The grammar's real decisions become editable code:

- **traversal:** recursive DFS from the canonical-rank root, neighbours
  visited in canonical-rank order (-> permutation-invariant);
- **ring closures:** DFS back-edges get paired digits (`%nn` past 9);
- **branches:** `( )` around all but the last child;
- **atoms:** fully-explicit `[<symbol><Hn><charge>]`, kekulized, stereo
  dropped -- verbose on purpose, so shortening is a *rule*, not a baked
  assumption;
- **bonds:** `=`/`#`, single implicit.

## Verified (10k held-out ZINC250k)
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
