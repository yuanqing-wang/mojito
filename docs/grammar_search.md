# FunSearch prototype: grammar search under two-part MDL

A working evolutionary search over reversible molecular grammars,
scored by minimum description length. It demonstrates the design in
`grammar_design.md` end to end on CPU and shows the two levers --
shorter description and bounded vocabulary -- optimized jointly.

## What it does
- **Grammar** = reversible codec: DeepSMILES on/off, then BPE with k
  merges. Lossless by construction; every candidate passes a 100%
  round-trip gate before scoring.
- **Fitness** = two-part MDL (bits/molecule, lower better):
  `L(corpus|grammar)` = held-out bits/mol under a bigram entropy proxy,
  plus `L(grammar)/n_train` = amortized BPE-table transmission cost (the
  vocab regularizer).
- **Search** = FunSearch skeleton: seed population, mutate the best,
  gate on round-trip, keep improvements. The programmatic proposer
  stands in for the LLM proposer of a full run.

## Result (20k train / 3k held-out ZINC250k)

Two-part MDL landscape (fitness; lower is better):

| k | SMILES | DeepSMILES |
|---|---|---|
| 0    | 93.3 | 82.5 |
| 256  |  -   | 56.1 |
| 512  | 58.8 | **55.1** |
| 1024 | 59.0 | 56.0 |
| 2048 | 61.7 | 59.0 |
| 4096 | 66.3 | 63.7 |
| 8192 | 73.6 |  -   |

The search **autonomously discovered**, starting from {SMILES,
SMILES+bpe8192, DeepSMILES} seeds:
1. **DeepSMILES beats SMILES** at every vocab (rule selection) -- ~3-4
   bits/mol.
2. A **vocabulary sweet spot at k=512** (~550 tokens), *not* the largest
   vocab: corpus bits keep falling with k, but the grammar-transmission
   penalty turns total MDL upward beyond ~512. A pure "fewer tokens"
   objective would have run k to the maximum (8192, 5.5 tokens/mol) and
   overfit the vocabulary; the MDL objective picks 8.5 tokens/mol at a
   554-token vocabulary instead.

Best grammar: **DeepSMILES + 512 BPE merges**, 55.1 bits/mol, 8.5
tokens/mol, vocab 554. Figure: `docs/figures/grammar_search.pdf`.

## Why it matters
This is the minimal proof that the loop works: an automated evaluator +
mutation finds a better grammar than canonical SMILES and the right
vocabulary size *jointly*, under one principled objective. Swapping the
programmatic proposer for an LLM (the only change for a full FunSearch
run) lets it propose *new reversible rules* beyond DeepSMILES -- which
is where gains past the ~5% DeepSMILES edge would come from, toward the
trained-LM entropy floor (<= 48 bits/mol).

## Honest limits
- The rule library here is tiny (DeepSMILES x BPE), so the search space
  is small; the win over DeepSMILES-BPE is the *vocab sweet spot*, not a
  new grammar. New grammar rules need the LLM proposer.
- The bigram proxy is cheap but looser than a trained LM; the frontier
  should be re-scored with a small LM before trusting absolute numbers
  (the proxy is for *ranking* grammars in the loop).
