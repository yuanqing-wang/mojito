"""Generative benchmark: train a small GPT on one tokenization and
report validity, uniqueness, novelty, and held-out bits/molecule.

Compares the bijective block-cut tokenizer against canonical SMILES
under an identical model and budget. Held-out bits/molecule (model NLL /
ln2) is the density-estimation metric and is tokenizer-fair. Streams
JSONL metrics so a live dashboard can plot during training.

Usage:
    python scripts/generate_exp.py DATA.csv --tokenizer mojito --out RUN
"""

import argparse
import json
import math
import os
import random
import time

import torch

from mojito.gpt import GPT, GPTConfig
from mojito.lm_data import Vocab, build
from mojito.blocktree import canonical_smiles


def pad_batch(seqs, pad, device):
    n = max(len(s) for s in seqs)
    x = torch.full((len(seqs), n), pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x.to(device)


def batches(data, bs, pad, device, shuffle=True):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        seqs = [data[j] for j in idx[i:i + bs]]
        x = pad_batch([s[:-1] for s in seqs], pad, device)
        y = pad_batch([s[1:] for s in seqs], pad, device)
        yield x, y


@torch.no_grad()
def heldout_bits(model, data, bs, vocab, device):
    """Exact mean bits/molecule on held-out sequences."""
    model.eval()
    tot_nats, n = 0.0, 0
    for x, y in batches(data, bs, vocab.pad, device, shuffle=False):
        logits = model(x)
        ll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1),
            ignore_index=vocab.pad, reduction="sum",
        )
        tot_nats += ll.item()
        n += len(x)
    model.train()
    return tot_nats / n / math.log(2)


@torch.no_grad()
def sample_molecules(model, vocab, detok, n, device, block_size):
    model.eval()
    valid, smis = 0, []
    for _ in range(0, n, 100):
        k = min(100, n - len(smis) if n - len(smis) < 100 else 100)
        start = torch.full((k, 1), vocab.bos, dtype=torch.long, device=device)
        out = model.generate(start, block_size - 1, vocab.eos)
        for row in out.tolist():
            mol = detok(vocab.decode(row[1:]))
            if mol is not None:
                valid += 1
                smis.append(canonical_smiles(mol))
    model.train()
    uniq = len(set(smis))
    return {
        "validity": valid / n,
        "uniqueness": uniq / max(len(smis), 1),
        "canonical": smis,
    }


def run(args):
    os.makedirs(args.out, exist_ok=True)
    device = args.device
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    import pandas as pd
    smis = pd.read_csv(args.data)["smiles"].str.strip().tolist()
    random.Random(args.seed).shuffle(smis)
    train_smi = smis[:args.n_train]
    valid_smi = smis[args.n_train:args.n_train + args.n_valid]
    train_lists, detok = build(train_smi, args.tokenizer)
    valid_lists, _ = build(valid_smi, args.tokenizer)
    vocab = Vocab(train_lists)
    train_set = train_seen = {canonical_smiles(s) for s in train_smi}
    enc = [vocab.encode(t) for t in train_lists
           if len(t) + 2 <= args.block_size]
    val = [vocab.encode(t) for t in valid_lists
           if len(t) + 2 <= args.block_size]

    cfg = GPTConfig(
        vocab_size=len(vocab), block_size=args.block_size,
        n_layer=args.layers, n_head=args.heads, n_embd=args.embd,
    )
    model = GPT(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump({**vars(args), "vocab_size": len(vocab),
                   "params": sum(p.numel() for p in model.parameters()),
                   "train_seqs": len(enc)}, f, indent=1)

    log = open(os.path.join(args.out, "metrics.jsonl"), "a")
    step, start = 0, time.time()
    while step < args.steps:
        for x, y in batches(enc, args.batch_size, vocab.pad, device):
            loss = model.loss(x, y, vocab.pad)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                bits = heldout_bits(model, val, args.batch_size, vocab,
                                    device)
                gen = sample_molecules(model, vocab, detok, args.n_sample,
                                       device, args.block_size)
                novel = sum(c not in train_seen for c in gen["canonical"])
                rec = {
                    "step": step, "time": time.time() - start,
                    "tokenizer": args.tokenizer,
                    "train_loss_bits": loss.item() / math.log(2),
                    "heldout_bits_per_mol": bits,
                    "validity": gen["validity"],
                    "uniqueness": gen["uniqueness"],
                    "novelty": novel / max(len(gen["canonical"]), 1),
                }
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print(json.dumps(rec), flush=True)
            if step >= args.steps:
                break


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data")
    p.add_argument("--tokenizer", choices=["mojito", "smiles"],
                   required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--n-valid", type=int, default=2000)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--embd", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--n-sample", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda"
                   if torch.cuda.is_available() else "cpu")
    run(p.parse_args())
