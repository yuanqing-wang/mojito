# Validating the entropy proxy (and what it revealed)

The grammar search ranks grammars by a cheap bigram bits/molecule proxy.
Is that proxy a faithful stand-in for a trained LM? We trained a small
GPT (2-layer, 2000 steps) on five grammars and compared.

## Result (15k train / 2k held-out ZINC250k)

| grammar | bigram proxy | trained-LM | tokens/mol | vocab |
|---|---|---|---|---|
| smiles+bpe0   | 93.2 | 60.9 | 38.1 |   39 |
| deep+bpe0     | 82.5 | 60.5 | 40.4 |   42 |
| smiles+bpe512 | 58.9 | 67.2 |  9.3 |  551 |
| deep+bpe512   | 55.3 | 63.8 |  8.6 |  554 |
| deep+bpe2048  | 58.2 | 65.3 |  6.5 | 2090 |

Overall Spearman rho = **-0.500** -- they *anti*-correlate. But the single
number is misleading; decomposed by axis:

- **Grammar rule (DeepSMILES vs SMILES):** proxy and LM **agree**. At
  bpe0, deep < smiles on both (82.5<93.2 proxy, 60.5<60.9 LM); at bpe512,
  deep < smiles on both (55.3<58.9, 63.8<67.2). The proxy correctly ranks
  the rule.
- **Vocabulary size (k):** proxy and LM **disagree**. The proxy rewards
  bigger vocab (fewer tokens); the trained LM rewards *smaller* vocab.

## Why (the real lesson)
The disagreement is on vocab, and it is a **training-budget confound**: a
large vocab has more embedding/softmax parameters, each token is seen
fewer times, so in a fixed budget those embeddings stay undertrained and
held-out NLL rises. The bigram proxy has no training, so no such bias. At
*convergence* a lossless tokenizer's LM bits/mol approach the molecular
entropy regardless of tokenization, so converged LM bits barely rank
grammars by vocab either.

Conclusion -- neither bigram proxy nor (undertrained) LM bits is the
right objective for **vocab size**:
- the proxy is valid for **ranking grammar rules** (use it in the loop
  for rule selection);
- **vocab size** must be set by the MDL grammar-cost term (which is
  exactly what gave the k=512 sweet spot in `grammar_search.md`) plus
  downstream compute (tokens/mol, O(L^2)), **not** by model bits.

This vindicates the two-part MDL design: the grammar-cost term, not the
corpus-bits term, is what should -- and does -- govern vocabulary. It
also warns against using raw LM NLL to pick vocab under a fixed budget.

## Caveat
The LMs are undertrained (2000 steps, CPU); the confound would shrink
with more training, and a converged study (each grammar's LM trained to
convergence, ideally on GPU) is the rigorous follow-up. The qualitative
split -- agree on rules, budget-confounded on vocab -- is robust.
Figure: `docs/figures/proxy_validation.pdf`.
