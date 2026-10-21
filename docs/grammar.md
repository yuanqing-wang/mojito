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

## The entropy floor (corrected)

The true floor is the **Shannon entropy** H of the molecule
distribution: the minimum average bits/molecule for any lossless code
(source-coding theorem). H is never known exactly -- every code is an
upper bound, and better models give tighter bounds:

| bound | bits/mol | what it is |
|---|---|---|
| uniform BPE code (tokens x log2 vocab) | 65.5 | loosest; assumes equiprobable tokens |
| undertrained GPT held-out NLL | <= 48 | tighter, still falling |
| true entropy H | **< 48** (likely ~30-40) | the real floor |

An earlier version of this doc and figure labeled the 65.5 uniform-code
value "entropy floor" -- that was wrong. 65.5 assumes every BPE token is
equally likely; a real LM predicts non-uniformly and reaches <= 48
bits/molecule (the generative run, still improving). So the true floor
is well below 65.5, the per-vocab token "floor" of 5.46 does not apply
under a real model, and there is more headroom for a denser grammar than
the ~5% first claimed. What is robust: DeepSMILES yields shorter
sequences (5.75 vs 6.07 tokens/mol), which helps transformer compute
(O(L^2)) regardless of entropy.

## Takeaway

Grammar optimization is the one lever that genuinely beat canonical
SMILES in this project (unlike the block/ring structural tokenizers,
which lost on every information metric; see `blocktree.md`). The gain is
real but small and capped by entropy. **Practical recommendation: adopt
DeepSMILES-BPE as the tokenizer** -- a free ~5% shorter context at equal
vocab and losslessness -- rather than build a bespoke grammar for the
last ~5%. Figure: `docs/figures/grammar.pdf`.
