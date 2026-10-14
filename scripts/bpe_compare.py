"""Compare tokens/molecule after BPE: block-cut strings vs SMILES.

Both representations are reduced to the same SMILES-atom base units, so
the comparison isolates what the block-cut structure buys once BPE
merges frequent motifs. Trains BPE at several vocabulary sizes on a
train split and reports mean tokens/molecule on a disjoint held-out
split. Writes ``bpe_tokens.json`` and a vector PDF.

Usage:
    python scripts/bpe_compare.py DATA.csv --out docs/figures/bpe.pdf
"""

import argparse
import json
import re

import pandas as pd
from tokenizers import Tokenizer, models, trainers

from mojito.blocktree import canonical_smiles
from mojito.serialize import to_string, tokens as block_tokens

ATOM = re.compile(
    r"\[[^\]]+\]|Br?|Cl?|[BCNOPSFIbcnops]|[0-9]|[=#()/\\+\-@%.]"
)
VOCAB_SIZES = [256, 512, 1024, 2048, 4096]


def smiles_tokens(s):
    c = canonical_smiles(s)
    return ATOM.findall(c) if c else None


def token_lists(smis, which):
    """One atom-token list per molecule."""
    fn = (lambda s: block_tokens(to_string(s))) if which == "block" \
        else smiles_tokens
    out = []
    for s in smis:
        try:
            toks = fn(s)
        except Exception:
            toks = None
        if toks:
            out.append(toks)
    return out


def encode_chars(token_lists_all, charmap):
    """Map each atom token to one private-use char; BPE then merges
    across atoms (whitespace pre-tokenizers would forbid that)."""
    return ["".join(charmap[t] for t in toks) for toks in token_lists_all]


def train_bpe(lines, vocab_size):
    tok = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, show_progress=False)
    tok.train_from_iterator(lines, trainer)
    return tok


def mean_len(tok, lines):
    enc = tok.encode_batch(lines)
    return sum(len(e.tokens) for e in enc) / len(enc)


def run(path, out, n_train, n_eval, seed):
    smis = pd.read_csv(path)["smiles"].str.strip().tolist()
    import random
    random.Random(seed).shuffle(smis)
    train, held = smis[:n_train], smis[n_train:n_train + n_eval]
    results = {}
    for which in ("block", "smiles"):
        tr_tok = token_lists(train, which)
        ev_tok = token_lists(held, which)
        alphabet = sorted({t for toks in tr_tok + ev_tok for t in toks})
        charmap = {t: chr(0xE000 + i) for i, t in enumerate(alphabet)}
        tr = encode_chars(tr_tok, charmap)
        ev = encode_chars(ev_tok, charmap)
        base = sum(len(line.split()) for line in ev) / len(ev)
        row = {0: base}
        for v in VOCAB_SIZES:
            row[v] = mean_len(train_bpe(tr, v), ev)
            print(f"{which:6s} vocab {v:5d}: {row[v]:.2f} tokens/mol",
                  flush=True)
        results[which] = row
    with open("bpe_tokens.json", "w") as f:
        json.dump(results, f, indent=1)
    plot(results, out)


def plot(results, out):
    import os
    import warnings
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for p in font_manager.findSystemFonts([os.path.expanduser("~/.fonts")]):
        if "montserrat" in p.lower():
            font_manager.fontManager.addfont(p)
    if "Montserrat" in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = "Montserrat"
    else:
        warnings.warn("Montserrat not found")
    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    colors = {"block": "#0072B2", "smiles": "#D55E00"}
    labels = {"block": "block-cut (ours)", "smiles": "canonical SMILES"}
    for which, row in results.items():
        xs = sorted(k for k in row if k)
        ax.plot(xs, [row[k] for k in xs], "o-", color=colors[which],
                label=labels[which])
    ax.set_xscale("log", base=2)
    ax.set_xlabel("BPE vocabulary size")
    ax.set_ylabel("tokens / molecule")
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data")
    p.add_argument("--out", default="bpe.pdf")
    p.add_argument("--n-train", type=int, default=50_000)
    p.add_argument("--n-eval", type=int, default=5_000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    run(a.data, a.out, a.n_train, a.n_eval, a.seed)
