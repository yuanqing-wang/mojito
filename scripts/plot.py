"""Plot training curves from ``metrics.jsonl`` to a vector PDF.

Usage: ``python scripts/plot.py RUN_DIR [RUN_DIR ...] --out fig.pdf``
"""

import argparse
import json
import os
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

# Okabe-Ito, colorblind-safe
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
PANELS = [
    ("valid_exact", "Exact reconstruction"),
    ("valid_bond_f1", "Bond F1"),
    ("valid_code_perplexity", "Code perplexity"),
]


def use_montserrat():
    for path in font_manager.findSystemFonts(
        [os.path.expanduser("~/.fonts")]
    ):
        if "montserrat" in path.lower():
            font_manager.fontManager.addfont(path)
    names = {f.name for f in font_manager.fontManager.ttflist}
    if "Montserrat" in names:
        plt.rcParams["font.family"] = "Montserrat"
    else:
        warnings.warn("Montserrat not found; using default font.")
    plt.rcParams.update(
        {"font.size": 12, "axes.titlesize": 12, "pdf.fonttype": 42}
    )


def read(run_dir):
    with open(os.path.join(run_dir, "metrics.jsonl")) as f:
        return [json.loads(line) for line in f]


def plot(run_dirs, out, width=7.0):
    use_montserrat()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(width, 2.6))
    for color, run_dir in zip(PALETTE, run_dirs):
        records = read(run_dir)
        steps = [r["step"] for r in records]
        label = os.path.basename(os.path.normpath(run_dir))
        for ax, (key, _) in zip(axes, PANELS):
            ax.plot(steps, [r[key] for r in records], color=color,
                    label=label)
        axes[0].plot(
            steps, [r["valid_exact_upper_bound"] for r in records],
            color=color, ls="--", label=f"{label} bound",
        )
    for ax, (key, title) in zip(axes, PANELS):
        ax.set_title(title)
        ax.set_xlabel("Step")
        if key != "valid_code_perplexity":
            ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("runs", nargs="+")
    p.add_argument("--out", default="reconstruction.pdf")
    a = p.parse_args()
    plot(a.runs, a.out)
