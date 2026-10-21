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
    """A candidate reversible codec.

    Composable reversible rules: kekulize (aromatic -> Kekule bonds),
    then DeepSMILES (drop ring-closure matching + parentheses), then BPE
    with ``k`` merges. Each rule is individually invertible, so the
    pipeline is lossless; the search gates on a round-trip check."""

    deep: bool          # apply DeepSMILES transform
    k: int              # number of BPE merges (0 = atom-level)
    kekule: bool = False  # write Kekule instead of aromatic SMILES
    extra: tuple = ()     # names of reversible token rules (rules.REGISTRY)

    def label(self):
        base = "deep" if self.deep else "smiles"
        kek = "+kek" if self.kekule else ""
        ex = "".join(f"+{r}" for r in self.extra)
        return f"{base}{kek}{ex}+bpe{self.k}"


def canonical(smiles):
    m = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def base_string(smiles, deep, kekule=False):
    """Reversible base encoding (canonical SMILES, optionally Kekule and
    DeepSMILES)."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    c = Chem.MolToSmiles(m, isomericSmiles=False, kekuleSmiles=kekule)
    return _DS.encode(c) if deep else c


def base_decode(text, deep, kekule=False):
    try:
        smi = _DS.decode(text) if deep else text
        return Chem.MolFromSmiles(smi)
    except Exception:
        return None


def apply_rules(tokens, extra, encode=True):
    """Apply reversible token rules in order (encode) or reverse (decode)."""
    from .rules import REGISTRY
    rules = [REGISTRY[n] for n in extra]
    for rule in (rules if encode else reversed(rules)):
        tokens = rule.encode(tokens) if encode else rule.decode(tokens)
    return tokens


def grammar_tokens(smiles, deep, kekule=False, extra=()):
    """Base string -> atom tokens -> reversible rules (encode)."""
    s = base_string(smiles, deep, kekule)
    if s is None:
        return None
    return apply_rules(_ATOM.findall(s), extra, encode=True)


def round_trips(smiles, deep, kekule=False, extra=()):
    toks = grammar_tokens(smiles, deep, kekule, extra)
    if toks is None:
        return False
    s = "".join(apply_rules(list(toks), extra, encode=False))
    m = base_decode(s, deep, kekule)
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


def encode_ids(grammar, train_smi, held_smi):
    """Encode both splits to integer-id sequences under ``grammar``.

    Returns (train_ids, held_ids, vocab, tokens_per_mol, base_alpha), or
    None if the grammar fails the round-trip gate. Shared by the MDL
    evaluator and by GPT training (proxy validation)."""
    sample = random.sample(train_smi, min(200, len(train_smi)))
    if not all(round_trips(s, grammar.deep, grammar.kekule, grammar.extra)
               for s in sample):
        return None
    tr = [grammar_tokens(s, grammar.deep, grammar.kekule, grammar.extra)
          for s in train_smi]
    ev = [grammar_tokens(s, grammar.deep, grammar.kekule, grammar.extra)
          for s in held_smi]
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
        chars = sorted({c for s in tr_c + ev_c for c in s})
        stoi = {c: i for i, c in enumerate(chars)}
        tr_ids = [[stoi[c] for c in s] for s in tr_c]
        ev_ids = [[stoi[c] for c in s] for s in ev_c]
        vocab = len(stoi)
        tok_per = sum(len(s) for s in ev_c) / len(ev_c)
    return tr_ids, ev_ids, vocab, tok_per, base_alpha


def evaluate(grammar, train_smi, held_smi, gate_sample=200):
    """Round-trip gate, then two-part MDL fitness (lower is better)."""
    enc = encode_ids(grammar, train_smi, held_smi)
    if enc is None:
        return None
    tr_ids, ev_ids, vocab, tok_per, base_alpha = enc
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


PROPOSER_SYSTEM = (
    "You design reversible string grammars for molecules. Given scored "
    "candidates (lower MDL bits/molecule is better), propose the next "
    "grammar to evaluate as JSON {\"deep\": bool, \"kekule\": bool, "
    "\"k\": int} where deep=DeepSMILES, kekule=Kekule bonds, k=number of "
    "BPE merges. Explore toward lower MDL; respect the vocab/length "
    "trade-off (k too large inflates grammar cost). Reply with JSON only."
)


def llm_propose(results, ks, timeout=30):
    """LLM proposer (FunSearch's mutation step). Calls the Anthropic
    Messages API if ANTHROPIC_API_KEY is set; returns a Grammar or None
    (caller falls back to the programmatic proposer). A fuller version
    would have the LLM emit a *new reversible rule* as code, gated by the
    round-trip check; here it selects within the existing rule library,
    which needs no sandboxed execution."""
    import json
    import os
    import urllib.request
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    best = sorted(results, key=lambda r: r["fitness"])[:8]
    rows = "\n".join(
        f'{{"deep": {r["deep"]}, "kekule": {r.get("kekule", False)}, '
        f'"k": {r["k"]}}} -> {r["fitness"]:.1f} bits' for r in best)
    body = json.dumps({
        "model": os.environ.get("MOJITO_PROPOSER_MODEL", "claude-haiku-5-5"),
        "max_tokens": 128,
        "system": PROPOSER_SYSTEM,
        "messages": [{"role": "user",
                      "content": f"Scored candidates:\n{rows}\n"
                      f"Allowed k: {ks}. Propose the next grammar."}],
    }).encode()
    req = urllib.request.Request(
        f"{base}/v1/messages", data=body, headers={
            "x-api-key": key, "anthropic-version": "2023-06-01",
            "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            out = json.load(resp)
        text = "".join(b.get("text", "") for b in out["content"])
        spec = json.loads(text[text.index("{"):text.rindex("}") + 1])
        k = min(ks, key=lambda x: abs(x - int(spec["k"])))
        return Grammar(bool(spec["deep"]), k, bool(spec.get("kekule", False)))
    except Exception:
        return None


def mutate(g, ks, rng):
    """Programmatic proposer: toggle a reversible rule or step k. (An LLM
    proposer would instead write a new reversible rule.)"""
    r = rng.random()
    if r < 0.4:
        return Grammar(not g.deep, g.k, g.kekule)
    if r < 0.6:
        return Grammar(g.deep, g.k, not g.kekule)
    i = ks.index(g.k)
    j = min(max(i + rng.choice([-1, 1]), 0), len(ks) - 1)
    return Grammar(g.deep, ks[j], g.kekule)


def search(train_smi, held_smi, ks, generations=30, seed=0, log=print,
           use_llm=False):
    """Evolutionary search minimizing two-part MDL. Returns all evaluated
    results and the best grammar. With ``use_llm`` the Anthropic API
    proposes mutations (FunSearch's LLM step), falling back to the
    programmatic proposer per step when unavailable."""
    rng = random.Random(seed)
    seeds = [Grammar(False, 0), Grammar(False, max(ks)), Grammar(True, 0),
             Grammar(True, 512, True)]
    pop, seen = [], {}
    for g in seeds:
        r = evaluate(g, train_smi, held_smi)
        if r:
            seen[(g.deep, g.k, g.kekule)] = r
            pop.append((r["fitness"], g))
            log(r)
    for _ in range(generations):
        _, parent = min(pop, key=lambda x: x[0])
        child = None
        if use_llm:
            child = llm_propose(list(seen.values()), ks)
        if child is None or (child.deep, child.k, child.kekule) in seen:
            child = mutate(parent, ks, rng)
        if (child.deep, child.k, child.kekule) in seen:
            child = mutate(rng.choice([g for _, g in pop]), ks, rng)
        if (child.deep, child.k, child.kekule) in seen:
            continue
        r = evaluate(child, train_smi, held_smi)
        if r:
            seen[(child.deep, child.k, child.kekule)] = r
            pop.append((r["fitness"], child))
            log(r)
    best = min(seen.values(), key=lambda r: r["fitness"])
    return list(seen.values()), best
