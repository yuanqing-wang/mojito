"""FunSearch-style search over reversible molecular grammars.

A *grammar* is a reversible codec: a base string transform (optionally
DeepSMILES, which drops ring-closure matching and parentheses) followed
by a BPE re-segmentation with ``k`` merges. Every grammar is lossless by
construction -- the search never trusts a proposal, it gates on a
round-trip check -- and is scored by two-part minimum description
length:

    fitness = L(corpus | grammar) + L(grammar) / n_train          (bits)

* L(corpus | grammar): held-out bits/molecule under a bigram model fit
  on the encoded corpus (a cheap, faithful proxy for LM NLL -- far
  tighter than the uniform tokens x log2(vocab) bound).
* L(grammar): bits to transmit the BPE merge table, amortized over the
  training set. This is the vocab regularizer; without it ``k`` -> inf.

The proposer (which mutates grammars) is pluggable: a programmatic
operator here, an LLM API call in a full run. FunSearch
(Romera-Paredes et al., Nature 2024) is exactly this loop with an LLM
proposer and a correctness evaluator.
"""

import math
import random
import re
from dataclasses import dataclass
from typing import Optional

import deepsmiles
from rdkit import Chem, RDLogger
from tokenizers import Tokenizer, models, trainers

RDLogger.DisableLog("rdApp.*")
_ATOM = re.compile(
    r"\[[^\]]+\]|Br?|Cl?|[BCNOPSFIbcnops]|[0-9]|[=#()/\\+\-@%.]"
)
_DS = deepsmiles.Converter(rings=True, branches=True)


@dataclass(frozen=True)
class Grammar:
    """A candidate reversible codec."""

    deep: bool          # apply DeepSMILES transform
    k: int              # number of BPE merges (0 = atom-level)

    def label(self):
        return f"{'deep' if self.deep else 'smiles'}+bpe{self.k}"


def canonical(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def base_string(smiles, deep):
    """Reversible base encoding (canonical SMILES, optionally DeepSMILES)."""
    c = canonical(smiles)
    if c is None:
        return None
    return _DS.encode(c) if deep else c


def base_decode(text, deep):
    try:
        smi = _DS.decode(text) if deep else text
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def round_trips(smiles, deep):
    m = base_decode(base_string(smiles, deep), deep)
    return m is not None and canonical(
        Chem.MolToSmiles(m)
    ) == canonical(smiles)


def _charmap(token_lists):
    alpha = sorted({t for L in token_lists for t in L})
    return {t: chr(0xE000 + i) for i, t in enumerate(alpha)}, len(alpha)


def _train_bpe(lines, k, base_alpha):
    tok = Tokenizer(models.BPE())
    tok.train_from_iterator(
        lines,
        trainers.BpeTrainer(vocab_size=base_alpha + k, show_progress=False),
    )
    return tok


def _bigram_bits(train_ids, held_ids, vocab, alpha=0.1):
    """Mean bits/sequence on held-out under an add-alpha bigram model."""
    uni = [alpha] * vocab
    bi = {}
    for seq in train_ids:
        prev = None
        for t in seq:
            uni[t] += 1
            if prev is not None:
                bi.setdefault(prev, [alpha] * vocab)[t] += 1
            prev = t
    uni_tot = sum(uni)
    uni_logp = [math.log2(c / uni_tot) for c in uni]
    bi_tot = {p: sum(row) for p, row in bi.items()}
    total, n = 0.0, 0
    for seq in held_ids:
        prev = None
        bits = 0.0
        for t in seq:
            if prev is not None and prev in bi:
                bits -= math.log2(bi[prev][t] / bi_tot[prev])
            else:
                bits -= uni_logp[t]
            prev = t
        total += bits
        n += 1
    return total / max(n, 1)


def _grammar_bits(k, base_alpha):
    """Bits to transmit the BPE merge table (Zipf/MDL vocab cost)."""
    return sum(2 * math.log2(base_alpha + i) for i in range(k))


def evaluate(grammar, train_smi, held_smi, gate_sample=200):
    """Round-trip gate, then two-part MDL fitness (lower is better)."""
    sample = random.sample(train_smi, min(gate_sample, len(train_smi)))
    if not all(round_trips(s, grammar.deep) for s in sample):
        return None
    tr = [_ATOM.findall(base_string(s, grammar.deep)) for s in train_smi]
    ev = [_ATOM.findall(base_string(s, grammar.deep)) for s in held_smi]
    tr = [t for t in tr if t]
    ev = [t for t in ev if t]
    cm, base_alpha = _charmap(tr + ev)
    tr_c = ["".join(cm[t] for t in L) for L in tr]
    ev_c = ["".join(cm[t] for t in L) for L in ev]
    if grammar.k > 0:
        tok = _train_bpe(tr_c, grammar.k, base_alpha)
        tr_ids = [e.ids for e in tok.encode_batch(tr_c)]
        ev_enc = tok.encode_batch(ev_c)
        ev_ids = [e.ids for e in ev_enc]
        vocab = tok.get_vocab_size()
        tok_per = sum(len(e.tokens) for e in ev_enc) / len(ev_enc)
    else:
        stoi = {c: i for i, c in enumerate(sorted({c for s in tr_c for c in s}
                                                  | {c for s in ev_c
                                                     for c in s}))}
        tr_ids = [[stoi[c] for c in s] for s in tr_c]
        ev_ids = [[stoi[c] for c in s] for s in ev_c]
        vocab = len(stoi)
        tok_per = sum(len(s) for s in ev_c) / len(ev_c)
    corpus_bits = _bigram_bits(tr_ids, ev_ids, vocab)
    gram_bits = _grammar_bits(grammar.k, base_alpha)
    fitness = corpus_bits + gram_bits / len(train_smi)
    return {
        "grammar": grammar.label(),
        "deep": grammar.deep, "k": grammar.k,
        "corpus_bits_per_mol": corpus_bits,
        "grammar_bits_amortized": gram_bits / len(train_smi),
        "fitness": fitness,
        "tokens_per_mol": tok_per,
        "vocab": vocab,
    }


def mutate(g, ks, rng):
    """Programmatic proposer: flip DeepSMILES or step k. (An LLM proposer
    would instead write a new reversible rule.)"""
    if rng.random() < 0.5:
        return Grammar(not g.deep, g.k)
    i = ks.index(g.k)
    j = min(max(i + rng.choice([-1, 1]), 0), len(ks) - 1)
    return Grammar(g.deep, ks[j])


def search(train_smi, held_smi, ks, generations=30, seed=0, log=print):
    """Evolutionary search minimizing two-part MDL. Returns all evaluated
    results and the best grammar."""
    rng = random.Random(seed)
    seeds = [Grammar(False, 0), Grammar(False, max(ks)), Grammar(True, 0)]
    pop, seen = [], {}
    for g in seeds:
        r = evaluate(g, train_smi, held_smi)
        if r:
            seen[(g.deep, g.k)] = r
            pop.append((r["fitness"], g))
            log(r)
    for _ in range(generations):
        _, parent = min(pop, key=lambda x: x[0])
        child = mutate(parent, ks, rng)
        if (child.deep, child.k) in seen:
            child = mutate(
                rng.choice([g for _, g in pop]), ks, rng)
        if (child.deep, child.k) in seen:
            continue
        r = evaluate(child, train_smi, held_smi)
        if r:
            seen[(child.deep, child.k)] = r
            pop.append((r["fitness"], child))
            log(r)
    best = min(seen.values(), key=lambda r: r["fitness"])
    return list(seen.values()), best
