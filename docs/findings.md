# Findings

## 1. FSQ atom tokenizer (this branch)

Settings: ZINC250k, canonical-SMILES dedup then 80/10/10 random split
(seed 0), atom vocabulary from train only. Featurization is lossless:
20,000/20,000 random ZINC molecules round-trip exactly (stereo removed).

Split: 198,243 / 24,780 / 24,781 molecules, 34 atom types, 0 OOV drops.

CPU smoke-scale run (`--hidden 128 --depth 4 --heads 8 --batch-size 64
--lr 1e-3 --warmup 300 --steps 6000`, FSQ levels 8-5-5-5 = 1000 codes,
seed 0; ~25 min on 4 CPU cores). Test set, best-validation checkpoint
(step 6000):

| atom acc | bond F1 | exact | collision bound | codes used | perplexity |
|---|---|---|---|---|---|
| 0.9998 | 0.925 | **0.185** | 0.891 | 945 / 1000 | 518 |

Curves: `docs/figures/fsq_cpu_run.pdf` (validation, every 500 steps;
single seed, no error bars). Still rising at the end — undertrained.

Reading: atoms are trivial; bonds are the bottleneck. Exact match is far
below the collision bound, and the bound itself is < 1, so this design
cannot be one-to-one by construction (equal tokens on non-equivalent
atoms are indistinguishable to an equivariant decoder). It also emits
one token per heavy atom (23.1 / molecule on ZINC) — no compression.
Keep as a learned baseline; not the main line.

## 2. Audit of `ring` (LLM + fragment tokens; latest branch, Dec 2025)

Lineage: `main -> att -> quantize -> jt -> ring`. `jt` dropped the GNN
and introduced a rule-based fragment tree (rings + rotatable-bond
pieces, DFS-serialized with `EDGE`/`BACK` tokens); `ring` LoRA-tunes
Qwen3-4B on SMolInstruct with these strings. Checked on 2,000 random
ZINC250k molecules (seed 0) against `origin/ring` (3146937):

| Issue | Evidence | Severity |
|---|---|---|
| `scripts/run.py` regex `(<SMILES>[^<;]*).(...)` has an unescaped `.` | `<SMILES>CCO</SMILES>` becomes `<SMILES>CC</SMILES>.` — the last character of every SMILES is dropped before tokenization | **Invalidates trained models** |
| `add(tokenizer)` is imported but never called in `run.py` | Mojito tokens never enter the vocabulary; the LLM sees BPE pieces of `<C1=CC=CC=C1> <EDGE 1-0> ...` | **Invalidates the comparison** |
| Token strings contain spaces (`<EDGE 1-0>`, `<EDGE [2, 1]:[1, 0]>`) but `string_to_molecule` splits on `" "` | `string_to_molecule(molecule_to_string(s))` raises `StopIteration` for 2000/2000 | No generated output can be decoded |
| Added vocabulary does not match emitted tokens | 23.8% of emitted tokens absent (indices start at 1 in the vocab, 0 in practice; list repr has spaces) | High |
| `SHARED_BONDS` added twice | 5,184 duplicates in 10,629 additional tokens | Low |
| List-level round trip is lossy | 1978/2000 exact; 4 exceptions; valence "repair" in `tree_to_molecule` silently edits H counts | Medium |
| Training labels | `labels = input_ids` incl. padding and prompt; `max_length=128` silently truncates | Medium |
| LoRA on `embed_tokens`, `lm_head` frozen | New-token output rows stay random; must use `modules_to_save` / trainable token indices | High once tokens are added |
| `eval.py` | first 100 ESOL test points, unparseable outputs dropped (should count as failures), no SMILES baseline | Not a result yet |

Conclusion: no number produced on `ring` so far measures the tokenizer.

Additional `ring` measurements (2,000 / 500 random ZINC250k molecules,
seed 0):

- Permutation invariance: identical tokens under 5 random atom orderings
  (`MolToSmiles(doRandom=True)`) for **4 / 499** molecules. Fragment
  ordering ties and attachment indices depend on input atom order.
- Length (mean per molecule): 23.1 heavy atoms, 44.3 canonical-SMILES
  characters, 38.2 regex SMILES tokens, **14.9 `ring` tokens**.

## 3. Direction

Invariance + injectivity = a canonical form. Proposed: certified
canonical labeling (colored graph, stereo kept) -> graph-BPE motif
merges with canonical tie-breaks and orbit-based attachment indices ->
minimum DFS code over the motif tree. Minimum DFS codes are
prefix-checkable (gSpan), enabling canonicity + valence masks at decode
time, hence a bijection between the LM's support and molecules and an
exact log p(molecule). Continuous information enters as a per-token
side channel added to input embeddings, never as the identity.
Closest prior work: Guo et al., ICLR 2026 (arXiv:2603.11099), Euler
serialization + BPE, "almost invariant", generation only a proof of
concept. Key risk: canonical strings are harder to learn than
randomized ones (Arus-Pous et al. 2019) — test this first.
