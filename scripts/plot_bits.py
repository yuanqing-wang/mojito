"""Two-panel figure: tokens/molecule and bits/molecule vs BPE vocab,
ring-collapsed vs SMILES, on held-out ZINC250k (50k train / 5k eval,
seed 0). Data from scripts/bpe_compare.py extended to vocab 32768.

bits/molecule = tokens/molecule * log2(vocab) is the fair metric:
tokens/molecule alone is reducible by enlarging the vocabulary, but all
lossless tokenizers converge to the molecular-entropy floor, so
description length is what "a better token" must mean.
"""

import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

VOCAB = [4096, 8192, 16384, 32768]
TOKENS = {
    "ringcollapse": [7.36, 6.03, 5.36, 4.93],
    "smiles": [6.07, 5.36, 4.80, 4.36],
}
BITS = {
    "ringcollapse": [88.3, 78.4, 75.0, 73.9],
    "smiles": [72.9, 69.6, 67.1, 65.5],
}
COLORS = {"ringcollapse": "#0072B2", "smiles": "#D55E00"}
LABELS = {"ringcollapse": "ring-collapsed (ours)", "smiles": "SMILES"}


def use_montserrat():
    for p in font_manager.findSystemFonts([os.path.expanduser("~/.fonts")]):
        if "montserrat" in p.lower():
            font_manager.fontManager.addfont(p)
    if "Montserrat" in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = "Montserrat"
    else:
        warnings.warn("Montserrat not found")
    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})


def main(out="docs/figures/bits.pdf"):
    use_montserrat()
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for which in ("ringcollapse", "smiles"):
        a.plot(VOCAB, TOKENS[which], "o-", color=COLORS[which],
               label=LABELS[which])
        b.plot(VOCAB, BITS[which], "o-", color=COLORS[which],
               label=LABELS[which])
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("BPE vocabulary size")
        ax.spines[["top", "right"]].set_visible(False)
    a.set_ylabel("tokens / molecule")
    a.set_ylim(bottom=0)
    b.set_ylabel("bits / molecule")
    b.set_ylim(bottom=0)
    a.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main()
