"""Structure-dependent downstream test: do token units help on
scaffold-split activity prediction?

logP/QED/SAS were composition-driven (atoms already strong). Activity
tasks (BBBP, BACE) are scaffold-dependent, with activity cliffs, so the
inductive bias of the token unit should matter more -- and the
scaffold split tests generalisation to unseen chemotypes. Compare
count-vector logistic regression across atom / BPE / BRICS tokenizations,
with an ECFP fingerprint as the gold-standard reference. Metric: test
AUROC on the MoleculeNet scaffold split.
"""
import re
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, BRICS
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")
ATOM = re.compile(r"\[[^\]]+\]|Br?|Cl?|[BCNOPSFIbcnops]|[0-9]|[=#()/\\+\-@%.]")


def atom_tokens(s):
    return ATOM.findall(s)


def brics_tokens(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return []
    frags = list(BRICS.BRICSDecompose(m))
    return frags if frags else [Chem.MolToSmiles(m)]


def bpe_encoder(train_strs, k):
    from tokenizers import Tokenizer, models, trainers
    tr = [atom_tokens(s) for s in train_strs]
    alpha = sorted({t for L in tr for t in L})
    cm = {t: chr(0xE000 + i) for i, t in enumerate(alpha)}
    tok = Tokenizer(models.BPE())
    tok.train_from_iterator(
        ["".join(cm[t] for t in L) for L in tr],
        trainers.BpeTrainer(vocab_size=len(alpha) + k, show_progress=False))

    def enc(s):
        return tok.encode(
            "".join(cm[t] for t in atom_tokens(s) if t in cm)).tokens
    return enc


def scaffold(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)


def scaffold_split(smiles, labels, frac_train=0.8, seed=0):
    """Group by Bemis-Murcko scaffold (no scaffold crosses the split),
    shuffle group order with a seed, fill train to frac_train. Re-seed if
    the test fold is single-class so AUROC is defined."""
    import random
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[scaffold(s)].append(i)
    group_list = list(groups.values())
    for attempt in range(20):
        rng = random.Random(seed + attempt)
        order = group_list[:]
        rng.shuffle(order)
        n_train = int(frac_train * len(smiles))
        train, test = [], []
        for g in order:
            (train if len(train) + len(g) <= n_train else test).extend(g)
        if len(set(labels[test])) > 1 and len(set(labels[train])) > 1:
            return train, test
    return train, test


def counts(token_lists, vocab):
    idx = {t: i for i, t in enumerate(vocab)}
    X = np.zeros((len(token_lists), len(vocab)), dtype=np.float32)
    for r, toks in enumerate(token_lists):
        for t in toks:
            if t in idx:
                X[r, idx[t]] += 1.0
    return X


def ecfp(smis, n_bits=2048, radius=2):
    X = np.zeros((len(smis), n_bits), dtype=np.float32)
    for r, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fp = AllChem.GetHashedMorganFingerprint(m, radius, nBits=n_bits)
        for bit, c in fp.GetNonzeroElements().items():
            X[r, bit] = c
    return X


def auroc(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def auroc_for(name, fn, tr_s, te_s, ytr, yte, topk):
    if name == "ECFP":
        Xtr, Xte = ecfp(tr_s), ecfp(te_s)
    else:
        tr_tok = [fn(s) for s in tr_s]
        te_tok = [fn(s) for s in te_s]
        vocab = [t for t, _ in Counter(
            t for L in tr_tok for t in L).most_common(topk)]
        Xtr, Xte = counts(tr_tok, vocab), counts(te_tok, vocab)
    return auroc(Xtr, ytr, Xte, yte)


def run(path, smi_col, label_col, n_seeds=5, topk=2000):
    df = pd.read_csv(path)
    df = df[[smi_col, label_col]].dropna()
    df[smi_col] = df[smi_col].astype(str).str.strip()
    df = df[df[smi_col].map(lambda s: Chem.MolFromSmiles(s) is not None)]
    smis = df[smi_col].tolist()
    y = df[label_col].to_numpy().astype(int)
    feat_names = ["atom", "bpe512", "bpe2048", "brics", "ECFP"]
    scores = {k: [] for k in feat_names}
    for seed in range(n_seeds):
        tr_idx, te_idx = scaffold_split(smis, y, seed=seed * 100)
        ytr, yte = y[tr_idx], y[te_idx]
        tr_s = [smis[i] for i in tr_idx]
        te_s = [smis[i] for i in te_idx]
        fns = {"atom": atom_tokens, "bpe512": bpe_encoder(tr_s, 512),
               "bpe2048": bpe_encoder(tr_s, 2048), "brics": brics_tokens,
               "ECFP": None}
        for name in feat_names:
            scores[name].append(
                auroc_for(name, fns[name], tr_s, te_s, ytr, yte, topk))
    print(f"== {path.split('/')[-1]}  n={len(smis)}  pos={y.mean():.2f}  "
          f"({n_seeds} scaffold splits, mean+/-std AUROC)", flush=True)
    for name in feat_names:
        a = np.array(scores[name])
        print(f"   {name:10s} {a.mean():.3f} +/- {a.std():.3f}", flush=True)


if __name__ == "__main__":
    D = sys.argv[1]
    run(f"{D}/BBBP.csv", "smiles", "p_np")
    run(f"{D}/bace.csv", "mol", "Class")
