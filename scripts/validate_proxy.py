"""Validate the bigram entropy proxy against a trained LM.

The grammar search ranks grammars by a cheap bigram bits/molecule proxy.
This only works if the proxy *ranks* grammars the way a real language
model would. Here we train a small GPT on several grammars and compare
its held-out bits/molecule to the bigram proxy, reporting the Spearman
rank correlation. High correlation => the proxy is a valid in-loop
fitness; the loop can trust it.
"""

import argparse
import json
import math
import os
import random

import torch

from mojito.gpt import GPT, GPTConfig
from mojito.grammar_search import Grammar, encode_ids, _bigram_bits


def pad(seqs, pad_id, device):
    n = max(len(s) for s in seqs)
    x = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x.to(device)


def batches(data, bs, pad_id, device, shuffle=True):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        seqs = [data[j] for j in idx[i:i + bs]]
        yield (pad(seqs, pad_id, device),
               [len(s) for s in seqs])


def add_specials(seqs, bos, eos):
    return [[bos] + s + [eos] for s in seqs]


@torch.no_grad()
def lm_bits(model, data, pad_id, device):
    """Mean bits/molecule under the trained model."""
    model.eval()
    tot, n = 0.0, 0
    for x, _ in batches(data, 64, pad_id, device, shuffle=False):
        logits = model(x[:, :-1])
        ll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
            ignore_index=pad_id, reduction="sum")
        tot += ll.item()
        n += len(x)
    model.train()
    return tot / n / math.log(2)


def train_lm(tr_ids, ev_ids, vocab, steps, device, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    bos, eos, pad_id = vocab, vocab + 1, vocab + 2
    tr = add_specials([s for s in tr_ids if len(s) <= 126], bos, eos)
    ev = add_specials([s for s in ev_ids if len(s) <= 126], bos, eos)
    cfg = GPTConfig(vocab_size=vocab + 3, block_size=128, n_layer=2,
                    n_head=4, n_embd=128)
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    step = 0
    while step < steps:
        for x, _ in batches(tr, 64, pad_id, device):
            loss = model.loss(x[:, :-1], x[:, 1:], pad_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step >= steps:
                break
    return lm_bits(model, ev, pad_id, device)


def spearman(a, b):
    def ranks(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    d2 = sum((ra[i] - rb[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


def run(args):
    import pandas as pd
    os.makedirs(args.out, exist_ok=True)
    smis = pd.read_csv(args.data)["smiles"].str.strip().tolist()
    random.Random(args.seed).shuffle(smis)
    train = smis[:args.n_train]
    held = smis[args.n_train:args.n_train + args.n_eval]
    grammars = [Grammar(False, 0), Grammar(False, 512),
                Grammar(True, 0), Grammar(True, 512), Grammar(True, 2048)]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for g in grammars:
        enc = encode_ids(g, train, held)
        tr_ids, ev_ids, vocab, tok_per, _ = enc
        proxy = _bigram_bits(tr_ids, ev_ids, vocab)
        lm = train_lm(tr_ids, ev_ids, vocab, args.steps, device, args.seed)
        row = {"grammar": g.label(), "bigram_proxy_bits": proxy,
               "lm_bits": lm, "tokens_per_mol": tok_per, "vocab": vocab}
        rows.append(row)
        print(json.dumps(row), flush=True)
    rho = spearman([r["bigram_proxy_bits"] for r in rows],
                   [r["lm_bits"] for r in rows])
    out = {"rows": rows, "spearman_rho": rho}
    with open(os.path.join(args.out, "proxy_validation.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(f"SPEARMAN rho = {rho:.3f}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data")
    p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=15000)
    p.add_argument("--n-eval", type=int, default=2000)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())
