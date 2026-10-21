"""Plot grammar comparison: tokens/molecule and bits/molecule vs BPE
vocab for SMILES / DeepSMILES / SELFIES, with the entropy-floor
reference on the token panel. Reads grammar_tokens.json.
"""

import argparse
import json
import os
import warnings
from math import log2

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Entropy estimate (bits/molecule): the lowest bits/molecule reached at
# large vocab in earlier runs (SMILES-BPE at vocab 32768 ~= 65.5).
ENTROPY = 65.5
COLORS = {"smiles": "#D55E00", "deepsmiles": "#0072B2", "selfies": "#999999"}
LABELS = {"smiles": "canonical SMILES", "deepsmiles": "DeepSMILES",
          "selfies": "SELFIES"}


def use_montserrat():
    for p in font_manager.findSystemFonts([os.path.expanduser("~/.fonts")]):
        if "montserrat" in p.lower():
            font_manager.fontManager.addfont(p)
    if "Montserrat" in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = "Montserrat"
    else:
        warnings.warn("Montserrat not found")
    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})


def main(src, out):
    use_montserrat()
    data = json.load(open(src))
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for name, row in data.items():
        vs = sorted(int(k) for k in row if k != "0")
        a.plot(vs, [row[str(v)]["tok"] for v in vs], "o-",
               color=COLORS[name], label=LABELS[name])
        b.plot(vs, [row[str(v)]["bits"] for v in vs], "o-",
               color=COLORS[name], label=LABELS[name])
    vs = sorted(int(k) for k in next(iter(data.values())) if k != "0")
    a.plot(vs, [ENTROPY / log2(v) for v in vs], "k--", lw=1,
           label="entropy floor")
    b.axhline(ENTROPY, color="k", ls="--", lw=1, label="entropy")
    for ax in (a, b):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("BPE vocabulary size")
        ax.spines[["top", "right"]].set_visible(False)
    a.set_ylabel("tokens / molecule")
    a.set_ylim(bottom=0)
    b.set_ylabel("bits / molecule")
    b.set_ylim(bottom=0)
    a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", default="docs/figures/grammar_tokens.json")
    p.add_argument("--out", default="docs/figures/grammar.pdf")
    a = p.parse_args()
    main(a.src, a.out)
