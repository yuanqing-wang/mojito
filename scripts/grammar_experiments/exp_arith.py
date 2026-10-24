"""Experiment #3: arithmetic coding vs tokenizer.

Arithmetic coding against a learned prior emits exactly the model's NLL
as bits. So held-out bits/molecule under a trained LM = the achievable
compressed size with that tokenizer + arithmetic coding. Train an
equal-budget LM on three tokenizations (char, atom, BPE-512) and compare
bits/molecule. If they converge, compression is tokenizer-independent
(all approach entropy) -- the compression game is over and tokenizer
choice must be justified by compute/learnability/downstream, not bits.
"""
import math
import random
import re
import sys

import pandas as pd
import torch
from tokenizers import Tokenizer, models, trainers

sys.path.insert(0, ".")
from minigpt import GPT, GPTConfig  # noqa: E402
import smiles_program as sp  # noqa: E402
from rdkit import Chem  # noqa: E402

ATOM = re.compile(r"\[[^\]]+\]|Br?|Cl?|[BCNOPSFIbcnops]|[0-9]|[=#()/\\+\-@%.]")


def canon(s):
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def char_tokens(s):
    return list(s)


def atom_tokens(s):
    return ATOM.findall(s)


def bpe_encode(train_strs, held_strs, k):
    """Return id-lists for train/held and vocab size for BPE over atoms."""
    def atoms(s):
        return ATOM.findall(s)
    tr = [atoms(s) for s in train_strs]
    ev = [atoms(s) for s in held_strs]
    alpha = sorted({t for L in tr + ev for t in L})
    cm = {t: chr(0xE000 + i) for i, t in enumerate(alpha)}
    tr_c = ["".join(cm[t] for t in L) for L in tr]
    ev_c = ["".join(cm[t] for t in L) for L in ev]
    tok = Tokenizer(models.BPE())
    tok.train_from_iterator(
        tr_c, trainers.BpeTrainer(vocab_size=len(alpha) + k,
                                  show_progress=False))
    tr_ids = [e.ids for e in tok.encode_batch(tr_c)]
    ev_ids = [e.ids for e in tok.encode_batch(ev_c)]
    return tr_ids, ev_ids, tok.get_vocab_size()


def to_ids(seqs):
    vocab = {}
    out = []
    for s in seqs:
        out.append([vocab.setdefault(t, len(vocab)) for t in s])
    return out, vocab


def pad(batch, pad_id, dev):
    n = max(len(x) for x in batch)
    t = torch.full((len(batch), n), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        t[i, :len(x)] = torch.tensor(x)
    return t.to(dev)


def batches(data, bs, pad_id, dev, shuffle=True):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield pad([data[j] for j in idx[i:i + bs]], pad_id, dev)


@torch.no_grad()
def bits_per_mol(model, data, pad_id, dev):
    model.eval()
    tot, n = 0.0, 0
    for x in batches(data, 64, pad_id, dev, shuffle=False):
        logits = model(x[:, :-1])
        ll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1),
            ignore_index=pad_id, reduction="sum")
        tot += ll.item()
        n += len(x)
    model.train()
    return tot / n / math.log(2)


def train_eval(tr_ids, ev_ids, vocab, steps, dev, seed=0):
    torch.manual_seed(seed)
    random.seed(seed)
    bos, eos, pad_id = vocab, vocab + 1, vocab + 2
    tr = [[bos] + s + [eos] for s in tr_ids if len(s) <= 254]
    ev = [[bos] + s + [eos] for s in ev_ids if len(s) <= 254]
    cfg = GPTConfig(vocab_size=vocab + 3, block_size=256, n_layer=3,
                    n_head=4, n_embd=192)
    model = GPT(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    step = 0
    while step < steps:
        for x in batches(tr, 64, pad_id, dev):
            model.loss(x[:, :-1], x[:, 1:], pad_id).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 1000 == 0:
                print(f"    step {step} val {bits_per_mol(model, ev, pad_id, dev):.1f}",
                      flush=True)
            if step >= steps:
                break
    return bits_per_mol(model, ev, pad_id, dev)


def run(path, n_train, n_eval, steps):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    smis = pd.read_csv(path)["smiles"].str.strip().tolist()
    random.Random(0).shuffle(smis)
    cs = [c for c in (canon(s) for s in smis) if c]
    train, held = cs[:n_train], cs[n_train:n_train + n_eval]
    # char
    tr, vocab = to_ids([char_tokens(s) for s in train])
    ev = [[vocab.get(t, len(vocab)) for t in char_tokens(s)] for s in held]
    print("char-level:", flush=True)
    bc = train_eval(tr, ev, len(vocab) + 1, steps, dev)
    # atom
    tr, vocab = to_ids([atom_tokens(s) for s in train])
    ev = [[vocab.get(t, len(vocab)) for t in atom_tokens(s)] for s in held]
    print("atom-level:", flush=True)
    ba = train_eval(tr, ev, len(vocab) + 1, steps, dev)
    # bpe-512
    tr_ids, ev_ids, v = bpe_encode(train, held, 512)
    print("bpe-512:", flush=True)
    bb = train_eval(tr_ids, ev_ids, v, steps, dev)
    print(f"RESULT bits/mol  char={bc:.1f}  atom={ba:.1f}  bpe512={bb:.1f}",
          flush=True)


if __name__ == "__main__":
    run(sys.argv[1], 15000, 2000, 4000)
