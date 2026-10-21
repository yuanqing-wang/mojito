# Designing an LLM-optimized molecular language

Reframe: stop hand-building a tokenizer; **learn the molecular language
itself** as the solution to a minimum-description-length (MDL) problem,
with an LLM evolving the grammar in the loop. The grammar is optimized
for description length (length + entropy) under a hard losslessness
constraint. This is new: the literature has validity-grammars, learned
tokenizers, and LLM program search, but none combines them to *design a
molecular notation*.

## Where this sits in the literature

**Objective — MDL / optimal coding.**
- Grammar induction classically minimizes grammar size + data encoded
  under it (G + E); MDL is the right objective for "a shorter language."
  MDL-trained nets also generalize from less data
  ([Lan, Chemla, Katzir 2023](https://aclanthology.org/people/roni-katzir/)).
- The linguistic principle is **Zipf's law of abbreviation** — frequent
  units get shorter codes — which is itself a prediction of optimal
  coding, with coding efficiency eta = Lmin / L
  ([Petrini et al. 2022](https://www.arxiv.org/pdf/2208.10384),
  [Revisiting Optimality of Word Lengths](https://arxiv.org/pdf/2312.03897)).
  Our DeepSMILES result is a first data point: removing redundancy (ring
  digits, parens) moved tokens/mol 6.07 -> 5.75 and bits/mol 72.9 -> 69.

**Validity by grammar (so the code only emits real molecules).**
- Grammar VAE serializes CFG parse trees, valid by construction
  ([Kusner et al. 2017](https://proceedings.mlr.press/v70/kusner17a.html)).
- Molecular Hypergraph Grammar builds a hyperedge-replacement graph
  grammar carrying chemical constraints, learned from a molecule set
  ([Kajino 2019](https://proceedings.mlr.press/v97/kajino19a.html)).

**The graph-entropy floor.**
- **Structural entropy** (Li & Pan): the min bits to losslessly encode an
  unlabeled graph, realized by an *encoding tree* (a hierarchical vertex
  grouping); the best tree captures the most structure
  ([overview](https://ar5iv.labs.arxiv.org/html/2001.09637)). This is the
  graph-native analogue of Shannon entropy and connects directly to our
  canonical **block-cut tree** (`blocktree.py`): a decomposition is
  exactly such an encoding tree.

**Tokenizer learning (the weak, current state of the art).**
- Unigram-LM pruning to maximize corpus likelihood
  ([Kudo 2018](https://arxiv.org/pdf/2004.03720)); SMILES Pair Encoding
  ([Li & Fourches](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.0c01127));
  entropy-guided trie refinement, >50% token reduction, lossless, but
  *not* grammar-level and *not* LLM-driven
  ([J. Cheminf. 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12866345/)).

**Search mechanism — LLM in the loop.**
- **FunSearch**: frozen LLM proposes programs, an evaluator scores them,
  best-of fed back; discovered new math and better heuristics
  ([Romera-Paredes et al. 2024](https://deepmind.google/discover/blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/)).
- Eureka / reward-evolution loops mutate executable code with an LLM
  ([ICLR 2024]). Token boundaries are even learnable end-to-end by
  score-function RL
  ([Dauncey & Wattenhofer 2026](https://proceedings.mlr.press/v306/dauncey26a.html)).

Gap (searched, not found): no LLM-driven, MDL-objective search over
molecular *grammars* scored by length + entropy under losslessness.

## The design: three coupled pieces

### 1. Substrate — what the grammar serializes
Serialize the **canonical graph decomposition we already built** (the
block-cut tree: unique, permutation-invariant, 100% lossless). This
hands the grammar invariance and reconstructability *for free*, so the
LLM optimizes only the *code*, never correctness. Structural-entropy
theory says the decomposition (encoding tree) is itself a lever — so a
later stage can let the LLM optimize the decomposition too.

### 2. Grammar — a reversible codec as a composable program
Represent the grammar as an ordered pipeline of **individually
invertible rewrite rules** over the decomposition/string. DeepSMILES =
canonical-SMILES + {drop ring-closure matching, drop parens}; both rules
are invertible, so the composition is lossless by construction. The LLM
proposes new reversible rules, e.g.:
- omit bond symbols inferable from valence/aromaticity (restore on
  decode);
- Zipf-code frequent motifs/rings to single short symbols (a learned
  codebook with canonical-fragment fallback, as in `blocktree.md`);
- reorder the traversal to make the token stream more predictable
  (lower conditional entropy).

Losslessness is guaranteed *by the invertible-rule contract* — malformed
LLM edits fail a round-trip gate rather than silently corrupting, so the
search never needs to trust the LLM.

### 3. Objective — two-part minimum description length
Optimize **total description length = L(grammar) + L(corpus | grammar)**,
not corpus bits alone. Minimizing corpus bits by itself drives the
vocabulary to infinity (one token per molecule -> ~0 corpus bits, zero
generalization); the grammar term is the regularizer that bounds vocab.
"Shorter description" and "limit vocabulary" are therefore the *same*
objective done properly, not two goals.

Fitness of a candidate grammar on a held-out corpus:
1. **Hard gate:** 100% round-trip (encode->decode == input), else reject.
   Optionally also permutation-invariance (same code for reordered atoms).
2. **L(corpus | grammar):** bits/molecule = held-out cross-entropy under a
   *cheap, faithful* entropy model fit on the encoded corpus — an order-k
   Markov / n-gram LM (fast) as a proxy for the true LM NLL (the
   generative run showed a trained GPT reaches <= 48 bits/mol, far below
   the 65.5 uniform-code bound; the proxy must track that). Held-out NLL
   already self-regularizes vocab (rare tokens cost more -log p, extra
   embedding params overfit) -- which is why it, not tokens/mol, is fair.
3. **L(grammar):** bits to transmit the vocabulary/rules themselves. This
   is the half held-out NLL omits (it assumes a shared grammar), and it
   prevents codebook overfitting. The Zipf/Unigram criterion makes it
   concrete: a token earns its slot iff frequency x bits-saved >
   cost-to-introduce (~log2 corpus size) -- exactly Unigram-LM pruning.
4. **Secondary:** tokens/molecule (transformer compute is O(L^2)); and the
   downstream |V|-dependent model cost (embedding + softmax are O(|V|d)).

The vocab tradeoff is **non-monotonic** -- too small lengthens sequences,
too large undertrains tokens and inflates the model -- so the search must
*find* the sweet spot (molecules: ~hundreds to low thousands), not fix it.
Cross-tokenizer comparisons must hold vocab fixed OR compare total
description length, else they are apples to oranges (why our grammar
experiment compared at matched vocab).

## The optimization loop (FunSearch pattern)
```
population <- {canonical SMILES, DeepSMILES}           # seeds
repeat:
    parents  <- sample best-scoring grammars
    child    <- LLM(mutate/cross parents: add/edit a reversible rule)
    if not round_trip_ok(child, holdout):  continue    # hard gate
    score    <- -bits_per_mol(child, holdout)          # + penalties
    population.add(child, score)
return best
```
The LLM earns its place because the space of *chemically meaningful*
reversible rules (valence inference, ring perception, aromaticity,
motif coding) is semantic — random/greedy search won't propose them;
an LLM will.

## Experiments / falsifiable claims
- Does the evolved grammar beat DeepSMILES-BPE (5.75 tok / 69 bits) and
  approach the trained-LM floor (<= 48 bits/mol)? Plot description length
  vs search iteration.
- Ablations: graph substrate on/off; Zipf coding on/off; entropy proxy
  (n-gram) vs full-LM fitness; LLM vs random rule proposal.
- Generalization: evolve on ZINC, test on natural products / ChEMBL —
  does the grammar overfit one distribution?
- Downstream: does a GPT on the evolved language reach lower held-out
  NLL and >= validity vs SMILES at equal compute/sequence length?

## Honest risks
- **Bounded upside.** bits/mol is floored by molecular entropy; DeepSMILES
  already captures much of SMILES's slack. The win is measured against
  the *trained-LM* floor (<= 48), not the uniform bound, and may be
  single-digit percent. Worth it only if it also helps downstream.
- **Cost.** LLM-in-loop + per-candidate entropy fit is expensive; the
  n-gram proxy and the round-trip gate must be cheap and the proxy must
  correlate with true LM NLL (validate this first).
- **Invariance vs learnability tension.** A canonical (one-string)
  language removes the data-augmentation that helps SMILES generators
  ([Arus-Pous 2019]); the evolved grammar must be tested as a *generator*,
  not only as a compressor.
- **Reward hacking.** The LLM may exploit the proxy (e.g. degenerate
  codes that the n-gram scores well but a real LM does not); periodically
  validate the frontier with a real small LM.

## Why this is a paper
It turns "which molecular string format?" — chosen by convention, not
evidence ([Rethinking Molecular Text Representations](https://arxiv.org/pdf/2606.03057))
— into a *measured optimization* with a clear objective (MDL), a floor
(structural/Shannon entropy), a principled prior (Zipf), and an
automated designer (LLM/FunSearch). The tokenizer stops being the
contribution; the *method for discovering optimal molecular languages*
is.
