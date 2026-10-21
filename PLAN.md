# Autoresearch harness for a better molecular program

Goal: search the space of **reversible molecular codecs** (programs like
`smiles_program.py`) for one that minimises description length, with
losslessness as a hard constraint and an LLM proposing code edits
(FunSearch). The program is modular (`CanonicalTraversal`, `AtomCodec`,
`BondCodec`, post-rules), so each decision is an independent mutation
point.

## 0. What "better" means (objective)

Minimise **two-part minimum description length** over a held-out corpus:

    score(program) = L(corpus | program) + L(program)          [bits]

- `L(corpus | program)` = held-out bits/molecule of the encoded strings.
  In-loop: a cheap order-k n-gram model (fast). On the frontier: a small
  trained LM (the gold metric; the generative run showed a GPT reaches
  <= 48 bits/mol, far below the 65.5 uniform-code bound).
- `L(program)` = program size in bits (its source, plus any learned
  tables such as a BPE merge list). This is the parsimony/vocab
  regulariser; without it the search memorises the corpus.

Secondary (Pareto, not folded into one number): tokens/molecule
(transformer compute is O(L^2)), program complexity, and generalisation
to a shifted distribution. A "win" is an honest point on the Pareto
front (fewer bits, or equal bits at fewer tokens / simpler program), not
a single cherry-picked metric.

Reality check (established earlier, keep front of mind): bits/molecule is
bounded below by molecular entropy; canonical SMILES is already close to
it (DeepSMILES recovered ~5% of the slack, traversal reorder ~0%). So
the harness must (a) measure against the trained-LM floor, not the
uniform bound, and (b) reward Pareto improvements, since the pure-bits
prize is small.

## 1. Guidelines (how the loop runs)

- **Seed** the population with known-good programs: the current
  `smiles_program.py`, a DeepSMILES-style variant, and an organic-subset
  variant. Reproducing their known scores is the harness sanity check.
- **FunSearch island model**: keep several sub-populations for diversity;
  the LLM is shown the best programs of an island plus their metrics,
  example encodings, and recent gate failures, and returns a code edit.
- **One component per proposal** (traversal, atom codec, bond codec, or a
  post-rule), so a failure localises and diffs stay reviewable.
- **Evaluate** every proposal: apply -> hard gates -> if it passes, full
  metrics -> add to the island with its score. Cache by source hash.
- **Human checkpoint**: the loop proposes and ranks; adopting a frontier
  program into the repo is a reviewed commit, never automatic.
- **Budgeted and resumable**: fixed caps (below); state and every
  candidate's source + metrics logged to a run dir; a live dashboard
  (best-so-far MDL vs iteration, the Pareto front, frontier diffs).

## 2. Guardrails (where the rigor lives)

Hard gates (pass/fail, checked before any scoring):

1. **Lossless**: `decode(encode(m)) == m` (canonical) on a large held-out
   set -- a single failure rejects the program.
2. **Permutation-invariant**: `encode` identical across N random atom
   orderings of each molecule.
3. **Deterministic**: same input -> same output, seed-independent; no
   wall-clock / RNG / global-state dependence.
4. **Self-contained**: proposed code may not call `MolToSmiles`,
   `MolFromSmiles`, or any canonical-SMILES routine (that would launder
   RDKit's answer). Enforced by static check of the source.

Execution safety:

5. **Sandbox**: restricted builtins, whitelisted imports (`re`, graph
   access only), no filesystem/network, run in an isolated
   subprocess/container with CPU-time and memory limits; a crash or
   timeout rejects the candidate.
6. **No irreversible actions**: the loop writes only to its run dir and
   opens no PRs/commits on its own.

Anti-reward-hacking (the subtle ones):

7. **Don't trust the proxy alone**: periodically re-score the frontier
   with the trained LM; penalise programs whose proxy-vs-LM rank
   diverges (we measured the n-gram proxy ranks grammar *rules* well but
   mis-ranks vocab size -- so vocab is governed by `L(program)`, not by
   proxy bits).
8. **Generalisation gate**: evaluate on a distribution shift (train on
   ZINC, test on ChEMBL / natural products). A program that overfits one
   corpus's statistics is rejected even if its train MDL is lowest.
9. **Parsimony pressure**: `L(program)` and a program-length cap stop the
   LLM from hard-coding lookup tables that encode the data.
10. **Kill criteria**: stop an island on a gate-failure streak,
    divergence, or budget exhaustion; cap total candidates, LLM tokens,
    and GPU hours up front.

## 3. Mutations (the search space)

Each is a code edit to one modular component, re-gated after applying:

- **AtomCodec** (highest known headroom -- the ~15% explicit-dialect
  overhead): organic-subset shorthand (`C` not `[CH2]`), implicit-H
  inference, compact charge notation. Round-trip gate certifies losslessness.
- **BondCodec**: implicit single bonds, aromatic lowercase + perception,
  omit bond symbols inferable from valence.
- **CanonicalTraversal**: start-atom rule, neighbour-ordering/biasing,
  ring-digit assignment. (Headroom looked ~0 for a bigram; re-test under
  a trained LM and the tokens metric before discarding.)
- **Post-rules** (reversible token transforms, FunSearch code-gen):
  DeepSMILES-style ring/branch removal, motif/Zipf coding of frequent
  substructures, run-length encoding -- each a gated `encode/decode` pair.
- **Vocab layer**: BPE with `k` merges, `k` chosen by the `L(program)`
  term (we saw a clean MDL sweet spot at k~512).
- **Crossover**: compose components from two frontier programs (e.g. one
  program's atom codec with another's traversal).

The LLM proposer receives the component source, the metric deltas it
should target, example encodings, and failure feedback; it returns a diff.

## 4. Metrics (logged per candidate, charted over the run)

Gates (pass/fail): round-trip %, invariance %, determinism, self-contained.

Scored:
- **held-out bits/molecule** -- n-gram proxy (in-loop) and trained-LM
  (frontier); the primary term.
- **MDL total** = `L(program)` + amortised corpus bits.
- **tokens/molecule** and **vocabulary size** (compute + parameter cost).
- **program complexity** (source bytes / AST nodes) -- parsimony.
- **generalisation gap** = bits(shifted test) - bits(train dist).
- **proxy-vs-LM rank correlation** on the frontier (reward-hacking guard).
- **population diversity** and **eval time/candidate** (health).

Reporting: best-so-far MDL vs iteration; the Pareto front over (bits,
tokens, program-length); the frontier programs as readable diffs from
`smiles_program.py`, each with its gate results and metrics.

## 5. Phasing

- **P0 scaffold**: evaluator (gates + metrics), dataset splits
  (ZINC train/val + ChEMBL shift), logging, dashboard. Reuse the bigram
  proxy, BPE, and small-GPT code already written.
- **P1 sanity**: programmatic search over the fixed component library;
  must reproduce the DeepSMILES + k~512 MDL sweet spot.
- **P2 LLM proposer**: sandboxed code-gen edits, starting with AtomCodec
  shortening (the known ~15% lever). Gate, score, iterate.
- **P3 gold fitness**: trained-LM scoring + generalisation gate on the
  frontier (GPU); Pareto analysis; reward-hacking audit.
- **P4 write-up**: ablations, human review of frontier programs, adopt
  the best as a reviewed commit.

The contribution, if any, is the *method* -- autoresearch that discovers
and certifies lossless molecular languages under MDL -- plus an honest
Pareto map of how far canonical SMILES can be improved. Negative results
(the entropy floor is close) are part of the result, not a failure.
