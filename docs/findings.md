# Findings

## 1. FSQ atom tokenizer (this branch)

Settings: ZINC250k, canonical-SMILES dedup then 80/10/10 random split
(seed 0), atom vocabulary from train only. Featurization is lossless:
20,000/20,000 random ZINC molecules round-trip exactly (stereo removed).

RESULTS_PLACEHOLDER

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
