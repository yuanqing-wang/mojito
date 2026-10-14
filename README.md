# mojito
Molecular Juxtaposition Tokenizer

One discrete token per heavy atom (FSQ, `prod(levels)` codes). The
decoder sees **only the token set** — never the graph — and must
reconstruct every atom and bond. Exact molecule-level reconstruction on
held-out molecules is the headline metric.

```
mojito/chem.py       lossless SMILES <-> integer graph (kekulized, no stereo)
mojito/data.py       dedup-then-split, vocab from train only, dense batches
mojito/encoder.py    graph transformer, bond-type + distance attention bias
mojito/quantizer.py  finite scalar quantization
mojito/decoder.py    set transformer + symmetric pairwise bond head
mojito/metrics.py    exact match, bond P/R/F1, code usage, collision bound
scripts/run.py       train / validate / test once on best checkpoint
scripts/plot.py      metrics.jsonl -> PDF
```

## Usage
```bash
pip install -e .[dev]
pytest -q mojito/tests
python scripts/run.py --data zinc250k.csv --out runs/base
python scripts/plot.py runs/base --out runs/base/curves.pdf
sbatch --account=$ACCOUNT scripts/run.sbatch zinc250k.csv runs/base
```
ZINC250k: `https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv`

## Metrics
- `exact`: fraction of molecules with all atoms and all bonds correct.
- `exact_upper_bound`: fraction with no *token collision* (two
  non-equivalent atoms sharing a code). The decoder is
  permutation-equivariant, so a collision makes exact reconstruction
  impossible; `exact <= exact_upper_bound`.
- `bond_f1`: over true bonds, type must match. Edge "accuracy" is not
  reported — ~90% of pairs are non-bonds.

See `docs/findings.md` for results and the audit of other branches.
