"""Run the FunSearch-style grammar search on ZINC and plot the result.

Streams each evaluated grammar as JSONL and writes a figure showing the
two-part MDL fitness landscape (corpus bits vs grammar/vocab cost) and
the sweet-spot vocabulary the search finds.

Usage: python scripts/run_grammar_search.py DATA.csv --out RUN
"""

import argparse
import json
import os
import random

import pandas as pd

from mojito.grammar_search import Grammar, evaluate, search


def plot(results, out):
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
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.4, 3.0))
    for deep, color, lab in [(False, "#D55E00", "SMILES"),
                             (True, "#0072B2", "DeepSMILES")]:
        pts = sorted((r for r in results if r["deep"] == deep),
                     key=lambda r: r["k"])
        if not pts:
            continue
        ks = [max(r["k"], 1) for r in pts]
        a.plot(ks, [r["fitness"] for r in pts], "o-", color=color, label=lab)
        b.plot([r["vocab"] for r in pts],
               [r["corpus_bits_per_mol"] for r in pts], "o-",
               color=color, label=lab)
    best = min(results, key=lambda r: r["fitness"])
    a.scatter([max(best["k"], 1)], [best["fitness"]], s=120,
              facecolors="none", edgecolors="k", zorder=5,
              label=f"best: {best['grammar']}")
    a.set_xscale("log", base=2)
    a.set_xlabel("BPE merges k")
    a.set_ylabel("two-part MDL (bits/mol)")
    b.set_xscale("log", base=2)
    b.set_xlabel("vocabulary size")
    b.set_ylabel("corpus bits/mol")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out)


def run(args):
    os.makedirs(args.out, exist_ok=True)
    smis = pd.read_csv(args.data)["smiles"].str.strip().tolist()
    random.Random(args.seed).shuffle(smis)
    train = smis[:args.n_train]
    held = smis[args.n_train:args.n_train + args.n_eval]
    ks = [0, 128, 256, 512, 1024, 2048, 4096, 8192]
    log = open(os.path.join(args.out, "search.jsonl"), "a")

    def record(r):
        log.write(json.dumps(r) + "\n")
        log.flush()
        print(json.dumps(r), flush=True)

    results, best = search(train, held, ks, generations=args.generations,
                           seed=args.seed, log=record,
                           use_llm=args.proposer == "llm")
    with open(os.path.join(args.out, "best.json"), "w") as f:
        json.dump(best, f, indent=1)
    plot(results, os.path.join(args.out, "grammar_search.pdf"))
    print("BEST", json.dumps(best), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-eval", type=int, default=3000)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--proposer", choices=["programmatic", "llm"],
                   default="programmatic",
                   help="llm uses the Anthropic API (needs API key)")
    run(p.parse_args())
