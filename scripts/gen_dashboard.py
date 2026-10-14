"""Live dashboard for the generative runs: reads each run's
metrics.jsonl and renders validity, uniqueness, novelty and held-out
bits/molecule vs step for mojito vs SMILES. Re-run to refresh.

Usage: python scripts/gen_dashboard.py RUN_mojito RUN_smiles --out fig.pdf
"""

import argparse
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

COLORS = {"mojito": "#0072B2", "smiles": "#D55E00"}
PANELS = [
    ("validity", "Validity", (0, 1)),
    ("uniqueness", "Uniqueness", (0, 1)),
    ("novelty", "Novelty", (0, 1)),
    ("heldout_bits_per_mol", "Held-out bits/mol", None),
]


def use_montserrat():
    for p in font_manager.findSystemFonts([os.path.expanduser("~/.fonts")]):
        if "montserrat" in p.lower():
            font_manager.fontManager.addfont(p)
    if "Montserrat" in {f.name for f in font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = "Montserrat"
    else:
        warnings.warn("Montserrat not found")
    plt.rcParams.update({"font.size": 12, "pdf.fonttype": 42})


def read(run):
    path = os.path.join(run, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def main(runs, out):
    use_montserrat()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(11, 2.7))
    for run in runs:
        recs = read(run)
        if not recs:
            continue
        which = recs[-1]["tokenizer"]
        steps = [r["step"] for r in recs]
        for ax, (key, title, ylim) in zip(axes, PANELS):
            ax.plot(steps, [r[key] for r in recs], "o-",
                    color=COLORS.get(which, "#555"), label=which, ms=3)
    for ax, (key, title, ylim) in zip(axes, PANELS):
        ax.set_title(title)
        ax.set_xlabel("step")
        if ylim:
            ax.set_ylim(*ylim)
        if key == "heldout_bits_per_mol":
            ax.set_yscale("log")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="docs/figures/generative.pdf")
    a = p.parse_args()
    main(a.runs, a.out)
