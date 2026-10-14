"""Train the FSQ molecular tokenizer and report held-out reconstruction.

Writes ``config.json``, ``data.json`` (split sizes, OOV drops),
``metrics.jsonl`` (one record per evaluation), ``best.pt`` and
``test.json`` (test metrics of the best-validation checkpoint) to
``--out``. Set ``WANDB_API_KEY`` and pass ``--wandb`` to mirror logs.
"""

import argparse
import json
import math
import os
import time

import torch

from mojito import FSQ, Decoder, Encoder, Tokenizer, reconstruction_loss
from mojito.data import (
    build_vocab,
    collate,
    featurize,
    read_smiles,
    split,
)
from mojito.metrics import Accumulator, batch_counts


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="CSV with `smiles`")
    p.add_argument("--out", required=True)
    p.add_argument("--subset", type=int, default=None,
                   help="use only the first N SMILES (smoke tests)")
    p.add_argument("--levels", type=int, nargs="+", default=[8, 5, 5, 5])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda"
                   if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true")
    return p.parse_args(argv)


def build_model(args, num_atom_types: int) -> Tokenizer:
    fsq = FSQ(args.levels)
    encoder = Encoder(
        num_atom_types, args.hidden, fsq.dim,
        depth=args.depth, num_heads=args.heads, dropout=args.dropout,
    )
    decoder = Decoder(
        fsq.dim, args.hidden, num_atom_types,
        depth=args.depth, num_heads=args.heads, dropout=args.dropout,
    )
    return Tokenizer(encoder, fsq, decoder)


def loader(graphs, batch_size, shuffle, seed=0):
    return torch.utils.data.DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )


def cosine(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    t = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))


@torch.no_grad()
def evaluate(model, data, device) -> dict:
    model.eval()
    acc, losses, n = Accumulator(), [0.0, 0.0], 0
    for batch in data:
        batch = batch.to(device)
        atom_logits, bond_logits, indices = model(batch)
        for i, loss in enumerate(
            reconstruction_loss(atom_logits, bond_logits, batch)
        ):
            losses[i] += loss.item()
        n += 1
        acc.update(batch_counts(atom_logits, bond_logits, indices, batch))
    model.train()
    out = acc.summarize(model.quantizer.codebook_size)
    out["loss_atom"], out["loss_bond"] = losses[0] / n, losses[1] / n
    return out


def prepare(args, out):
    smiles = read_smiles(args.data)[:args.subset]
    train, valid, test = split(smiles, seed=args.seed)
    keys = build_vocab(train)
    sets, info = {}, {"num_atom_types": len(keys), "atom_keys": keys}
    for name, part in (("train", train), ("valid", valid), ("test", test)):
        sets[name], dropped = featurize(part, keys)
        info[name] = {"molecules": len(sets[name]), "dropped_oov": dropped}
    with open(os.path.join(out, "data.json"), "w") as f:
        json.dump(info, f, indent=1)
    return sets, keys


def run(args):
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=1)
    torch.manual_seed(args.seed)
    sets, keys = prepare(args, args.out)
    model = build_model(args, len(keys)).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine(s, args.warmup, args.steps)
    )
    train = loader(sets["train"], args.batch_size, True, args.seed)
    valid = loader(sets["valid"], args.batch_size, False)
    if args.wandb:
        import wandb
        wandb.init(project="mojito", config=vars(args))

    log = open(os.path.join(args.out, "metrics.jsonl"), "a")
    best, step, start = -1.0, 0, time.time()
    while step < args.steps:
        for batch in train:
            batch = batch.to(args.device)
            atom_logits, bond_logits, _ = model(batch)
            loss_atom, loss_bond = reconstruction_loss(
                atom_logits, bond_logits, batch
            )
            optimizer.zero_grad(set_to_none=True)
            (loss_atom + loss_bond).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            if step % args.eval_every == 0 or step == args.steps:
                record = {
                    "step": step,
                    "time": time.time() - start,
                    "train_loss_atom": loss_atom.item(),
                    "train_loss_bond": loss_bond.item(),
                    **{f"valid_{k}": v for k, v in
                       evaluate(model, valid, args.device).items()},
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps(record), flush=True)
                if args.wandb:
                    wandb.log(record, step=step)
                if record["valid_exact"] > best:
                    best = record["valid_exact"]
                    torch.save(
                        {"model": model.state_dict(), "args": vars(args),
                         "atom_keys": keys, "step": step},
                        os.path.join(args.out, "best.pt"),
                    )
            if step >= args.steps:
                break

    # test set touched once, with the best-validation checkpoint
    state = torch.load(os.path.join(args.out, "best.pt"))
    model.load_state_dict(state["model"])
    test = evaluate(
        model, loader(sets["test"], args.batch_size, False), args.device
    )
    test["step"] = state["step"]
    with open(os.path.join(args.out, "test.json"), "w") as f:
        json.dump(test, f, indent=1)
    print("test", json.dumps(test))


if __name__ == "__main__":
    run(parse_args())
