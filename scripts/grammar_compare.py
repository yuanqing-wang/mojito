"""Can a denser SMILES-like grammar beat canonical SMILES after BPE?

Compares canonical SMILES, DeepSMILES (drops ring-closure matching and
parentheses) and SELFIES on tokens/molecule and bits/molecule
(= tokens x log2 vocab) at matched BPE vocab sizes, on held-out
ZINC250k. The information-theoretic floor on tokens/molecule at vocab V
is entropy / log2(V); entropy is estimated from the lowest bits/molecule
reached. A grammar can only recover the redundancy slack above that
floor -- it cannot beat entropy.
"""

import argparse
import json
import re
from math import log2

import pandas as pd
import deepsmiles
import selfies as sf
from rdkit import Chem, RDLogger
from tokenizers import Tokenizer, models, trainers

RDLogger.DisableLog("rdApp.*")
ATOM = re.compile(
    r"\[[^\]]+\]|Br?|Cl?|[BCNOPSFIbcnops]|[0-9]|[=#()/\\+\-@%.]"
)
VOCAB = [256, 512, 1024, 2048, 4096]
_DS = deepsmiles.Converter(rings=True, branches=True)


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def smiles_toks(s):
    c = canon(s)
    return ATOM.findall(c) if c else None


def deep_toks(s):
    c = canon(s)
    if not c:
        return None
    try:
        return ATOM.findall(_DS.encode(c))
    except Exception:
        return None


def selfies_toks(s):
    c = canon(s)
    if not c:
        return None
    try:
        return list(sf.split_selfies(sf.encoder(c)))
    except Exception:
        return None


GRAMMARS = {
    "smiles": smiles_toks,
    "deepsmiles": deep_toks,
    "selfies": selfies_toks,
}


def token_lists(smis, fn):
    out = []
    for s in smis:
        t = fn(s)
        if t:
            out.append(t)
    return out


def encode_chars(lists):
    alpha = sorted({t for L in lists for t in L})
    cm = {t: chr(0xE000 + i) for i, t in enumerate(alpha)}
    return ["".join(cm[t] for t in L) for L in lists], len(alpha)


def run(path, n_train, n_eval, seed, out):
    import random
    smis = pd.read_csv(path)["smiles"].str.strip().tolist()
    random.Random(seed).shuffle(smis)
    train, held = smis[:n_train], smis[n_train:n_train + n_eval]
    results = {}
    print(f"{'grammar':12s}{'vocab':>7s}{'tok/mol':>9s}{'bits/mol':>10s}"
          f"{'alpha':>7s}", flush=True)
    for name, fn in GRAMMARS.items():
        tr, ev = token_lists(train, fn), token_lists(held, fn)
        alpha = sorted({t for L in tr + ev for t in L})
        cm = {t: chr(0xE000 + i) for i, t in enumerate(alpha)}
        etr = ["".join(cm[t] for t in L) for L in tr]
        eev = ["".join(cm[t] for t in L) for L in ev]
        base = sum(len(L) for L in ev) / len(ev)
        row = {"0": {"tok": base, "bits": None}}
        for V in VOCAB:
            tok = Tokenizer(models.BPE())
            tok.train_from_iterator(
                etr, trainers.BpeTrainer(vocab_size=V, show_progress=False))
            e = tok.encode_batch(eev)
            ml = sum(len(x.tokens) for x in e) / len(e)
            rv = tok.get_vocab_size()
            row[str(V)] = {"tok": ml, "bits": ml * log2(rv)}
            print(f"{name:12s}{V:>7d}{ml:>9.2f}{ml*log2(rv):>10.1f}"
                  f"{len(alpha):>7d}", flush=True)
        results[name] = row
    with open(out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data")
    p.add_argument("--n-train", type=int, default=50000)
    p.add_argument("--n-eval", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="docs/figures/grammar_tokens.json")
    a = p.parse_args()
    run(a.data, a.n_train, a.n_eval, a.seed, a.out)
