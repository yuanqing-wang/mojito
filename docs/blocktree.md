# Block-cut tree tokenizer

A molecular tokenizer that is **permutation-invariant**, **lossless**,
and fragment-level, built on the graph-theoretic block-cut tree.

## Idea
Decompose the 2D molecular graph into its biconnected components:
- a **block** = a maximal ring system or a single acyclic bond;
- a **cut atom** = an articulation atom shared by >= 2 blocks.

The block-cut tree is **unique** (no spanning-tree choice, unlike
junction-tree methods), so decomposition is deterministic. The molecule
is reconstructed exactly from three symbolic facts:
1. the tree (which blocks, joined at which cut atoms),
2. each block's canonical fragment,
3. the gluing: which cut atom each block shares, by order-invariant id.

Articulation atoms are named by their rank in RDKit's canonical atom
order (invariant to input numbering), and blocks are sorted canonically,
so two atom orderings of the same molecule give byte-identical output.

## Why this meets the three requirements
- **Permutation-invariant:** nothing depends on input atom order.
- **Lossless (one-to-one):** gluing is symbolic, so a later learned
  codebook over block fragments can be lossy without the tokenizer
  being lossy -- identity lives in the tree and the cut ids, not in any
  embedding. This is why atom-level VQ (collision-bounded at ~0.89 on
  ZINC, see `findings.md`) cannot be one-to-one but this can.
- **Efficient:** ~12 blocks/molecule vs. 23 heavy atoms; the GNN runs
  over a ~12-node block graph.

## Measured on ZINC250k (held-out, seed 0)
`python scripts/blocktree_stats.py zinc250k.csv --n 20000`

| metric | value |
|---|---|
| exact round-trip | 100.000% (20000/20000) |
| permutation-invariant (5 random orderings each) | 100.000% |
| blocks / molecule | 12.2 |
| blocks + cut-incidences / molecule (naive, pre-merge) | 32.6 |

Stereochemistry is dropped (as in SMILES-tokenizer baselines).

## Status and next steps
Done: lossless + invariant symbolic backbone (`mojito/blocktree.py`),
32 unit tests on hard cases (spiro, fused, biphenyl, symmetric
fragments, charged, heteroaromatic).

Open, in priority order:
1. **Serialization + efficiency.** The naive 32.6 tokens/molecule is
   worse than SMILES; a tree-BPE merge of adjacent blocks must bring it
   below ~10. This is the make-or-break number.
2. **Learned block codebook.** GNN per-block embedding -> residual FSQ
   code; literal canonical fragment as fallback for rare blocks (98.6%
   of molecules covered by top-4000 blocks, see `findings.md`).
3. **Masked decoding.** Valence + canonicality masks so every emitted
   string is a valid canonical code => exact `log p(molecule)`.
4. **Prior art.** Closest is Guo et al. ICLR 2026 (Euler serialization
   + BPE, only "almost invariant"); JT-VAE/HierVAE (learned, lossy
   assembly). None is a certified canonical, lossless, bijective
   fragment tokenizer.

## Novelty risk
Junction trees (2018) and tree canonicalization are standard; the
contribution is the *combination* -- certified canonical + lossless +
compressed fragment tokenizer with exact likelihood as an LLM
vocabulary -- plus the honest demonstration that the GNN route is
elegant only once identity is moved off the atoms and onto the tree.
Canonical cut-atom ids currently rely on RDKit canonical ranking; an
AHU tree-canonicalization would remove even that dependency.
