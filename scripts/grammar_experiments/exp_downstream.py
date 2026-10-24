"""Experiment #1: downstream utility of token units.

Does changing the token UNIT (atoms -> BPE -> chemical motifs) carry more
task-relevant signal? Tokenize each molecule three ways, build count
vectors over the top-K tokens, ridge-regress to ZINC's own logP / QED /
SAS, and compare held-out R^2. Higher R^2 at equal regressor = a better
inductive bias for the task -- the downstream fitness the harness would
optimise (objective 0b). This is not bounded by compression entropy.
"""
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS

RDLogger.DisableLog("rdApp.*")
ATOM = re.compile(r"\[[^\]]+\]|Br?|Cl?|[BCNOPSFIbcnops]|[0-9]|[=#()/\\+\-@%.]")


def atom_tokens(s):
    return ATOM.findall(s)


def brics_tokens(s):
    """Chemical motif tokens: BRICS fragments (whole mol if unbreakable)."""
    m = Chem.MolFromSmiles(s)
    if m is None:
        return []
    frags = list(BRICS.BRICSDecompose(m))
    return frags if frags else [Chem.MolToSmiles(m)]


def bpe_vocab(train_strs, k):
    from tokenizers import Tokenizer, models, trainers
    tr = [atom_tokens(s) for s in train_strs]
    alpha = sorted({t for L in tr for t in L})
    cm = {t: chr(0xE000 + i) for i, t in enumerate(alpha)}
    tok = Tokenizer(models.BPE())
    tok.train_from_iterator(
        ["".join(cm[t] for t in L) for L in tr],
        trainers.BpeTrainer(vocab_size=len(alpha) + k, show_progress=False))
    inv = {v: k2 for k2, v in cm.items()}

    def enc(s):
        e = tok.encode("".join(cm[t] for t in atom_tokens(s) if t in cm))
        return e.tokens
    return enc


def count_matrix(token_lists, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(token_lists), len(vocab)), dtype=np.float32)
    for r, toks in enumerate(token_lists):
        for t in toks:
            if t in idx:
                X[r, idx[t]] += 1.0
    return X


def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1), np.float32)])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1), np.float32)])
    d = Xtr.shape[1]
    A = Xtr.T @ Xtr + lam * np.eye(d, dtype=np.float32)
    w = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def run(path, n_train, n_eval, topk=1000):
    df = pd.read_csv(path)
    df["smiles"] = df["smiles"].str.strip()
    df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    tr_df, te_df = df.iloc[:n_train], df.iloc[n_train:n_train + n_eval]
    targets = ["logP", "qed", "SAS"]
    bpe_enc = bpe_vocab(tr_df["smiles"].tolist(), 512)
    tokenizers = {
        "atom": atom_tokens,
        "bpe512": bpe_enc,
        "brics_motif": brics_tokens,
    }
    print(f"{'tokenizer':12s}{'vocab':>7s}" +
          "".join(f"{t:>8s}" for t in targets), flush=True)
    for name, fn in tokenizers.items():
        tr_tok = [fn(s) for s in tr_df["smiles"]]
        te_tok = [fn(s) for s in te_df["smiles"]]
        cnt = Counter(t for L in tr_tok for t in L)
        vocab = [t for t, _ in cnt.most_common(topk)]
        Xtr = count_matrix(tr_tok, vocab)
        Xte = count_matrix(te_tok, vocab)
        r2s = []
        for tgt in targets:
            ytr = tr_df[tgt].to_numpy(np.float32)
            yte = te_df[tgt].to_numpy(np.float32)
            r2s.append(ridge_r2(Xtr, ytr, Xte, yte))
        print(f"{name:12s}{len(vocab):>7d}" +
              "".join(f"{r:>8.3f}" for r in r2s), flush=True)


if __name__ == "__main__":
    run(sys.argv[1], 15000, 3000)
