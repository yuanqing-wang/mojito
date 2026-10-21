# Grammar optimization: a denser SMILES-like code

Question: keep a linear, letter-based molecular string, but design a
less redundant grammar than canonical SMILES -- shorter sequences, same
losslessness. Does it beat canonical-SMILES-BPE?

## Result (50k train / 5k held-out ZINC250k, matched BPE vocab)

tokens/molecule (lower is better):

| vocab | DeepSMILES | canonical SMILES | SELFIES |
|---|---|---|---|
| 256  | 10.57 | 11.42 | 12.62 |
| 1024 |  7.46 |  8.10 |  8.83 |
| 4096 |  **5.75** |  6.07 |  6.69 |

bits/molecule (= tokens x log2 vocab):

| vocab | DeepSMILES | canonical SMILES | SELFIES |
|---|---|---|---|
| 256  | 84.5 | 91.3 | 101.0 |
| 4096 | **69.0** | 72.9 | 80.3 |

**DeepSMILES beats canonical SMILES at every vocabulary size on both
tokens/molecule and bits/molecule** -- ~5% at vocab 4096, up to ~8% at
small vocab -- while staying lossless and letter-based. It wins by
dropping SMILES' two redundancies: ring-closure digit *pairs* (one
ring-size digit instead) and parentheses (a `)` count). SELFIES is
worst: robustness makes it verbose.

## The ceiling (why the win is modest and bounded)

bits/molecule is bounded below by the **entropy of the molecule
distribution**, ~65 bits for ZINC (lowest bits/molecule reached at large
vocab). At vocab V the token floor is entropy / log2(V); at 4096 that is
65.5 / 12 = **5.46 tokens/molecule**. So:

- canonical SMILES 6.07 -> ~11% above the floor (redundant),
- DeepSMILES 5.75 -> ~5% above the floor (captures ~60% of the slack),
- hard floor 5.46 -> no lossless letter grammar can go below this.

A hand-designed grammar denser than DeepSMILES (e.g. also dropping bond
symbols inferable from valence, or aromaticity) could claw back the
remaining ~0.3 tokens/molecule, but that ~5% is the entire prize left.

## Takeaway

Grammar optimization is the one lever that genuinely beat canonical
SMILES in this project (unlike the block/ring structural tokenizers,
which lost on every information metric; see `blocktree.md`). The gain is
real but small and capped by entropy. **Practical recommendation: adopt
DeepSMILES-BPE as the tokenizer** -- a free ~5% shorter context at equal
vocab and losslessness -- rather than build a bespoke grammar for the
last ~5%. Figure: `docs/figures/grammar.pdf`.
