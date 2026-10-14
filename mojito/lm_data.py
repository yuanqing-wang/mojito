"""Tokenizer-agnostic vocabulary and batching for molecular LMs.

Two tokenizations are compared, both reversible:

* ``mojito``  -- canonical block-cut string (``serialize.to_string``),
  bijective and permutation-invariant;
* ``smiles``  -- canonical SMILES.

Each becomes a list of atom tokens; a shared ``Vocab`` maps tokens to
ids with ``<bos> <eos> <pad>`` specials. Detokenizing concatenates the
atom tokens (the block string uses " . " between fragments) and parses
back to a molecule, so generation validity is well defined.
"""

import re
from typing import Callable, Optional

from rdkit import Chem, RDLogger

from .blocktree import canonical_smiles
from .serialize import _SEP, to_string, from_string, tokens as block_tokens

RDLogger.DisableLog("rdApp.*")

_ATOM = re.compile(
    r"\[[^\]]+\]|Br?|Cl?|[BCNOPSFIbcnops]|[0-9]|[=#()/\\+\-@%.]"
)
BOS, EOS, PAD, UNK = "<bos>", "<eos>", "<pad>", "<unk>"
_TAG = re.compile(r"\[(.+):(\d+)\]")


def _smiles_tokens(s: str) -> Optional[list]:
    c = canonical_smiles(s)
    return _ATOM.findall(c) if c else None


def _smiles_detok(toks: list) -> Optional[Chem.Mol]:
    return Chem.MolFromSmiles("".join(toks))


def _mojito_tokens(s: str) -> Optional[list]:
    """Block-string tokens, with attachment ids split out of brackets so
    the vocabulary stays small and bounded: ``[N+:3]`` -> ``[N+]``, ``@3``."""
    text = to_string(s)
    if text is None:
        return None
    out = []
    for t in block_tokens(text):
        m = _TAG.fullmatch(t)
        if m:
            out.append(f"[{m.group(1)}]")
            out.append(f"@{m.group(2)}")
        else:
            out.append(t)
    return out


def _mojito_detok(toks: list) -> Optional[Chem.Mol]:
    merged, i = [], 0
    while i < len(toks):
        t = toks[i]
        if (t.startswith("[") and i + 1 < len(toks)
                and toks[i + 1].startswith("@")):
            merged.append(f"[{t[1:-1]}:{toks[i + 1][1:]}]")
            i += 2
        else:
            merged.append(t)
            i += 1
    parts, cur = [], []
    for t in merged:
        if t == ".":
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(t)
    parts.append("".join(cur))
    return from_string(_SEP.join(parts))


TOKENIZERS = {
    "mojito": (_mojito_tokens, _mojito_detok),
    "smiles": (_smiles_tokens, _smiles_detok),
}


class Vocab:
    """Bidirectional token<->id map with specials first."""

    def __init__(self, token_lists):
        vocab = {BOS: 0, EOS: 1, PAD: 2, UNK: 3}
        for toks in token_lists:
            for t in toks:
                vocab.setdefault(t, len(vocab))
        self.stoi = vocab
        self.itos = {i: t for t, i in vocab.items()}
        self.bos, self.eos, self.pad, self.unk = 0, 1, 2, 3

    def __len__(self):
        return len(self.stoi)

    def encode(self, toks):
        mid = [self.stoi.get(t, self.unk) for t in toks]
        return [self.bos] + mid + [self.eos]

    def decode(self, ids):
        out = []
        for i in ids:
            if i == self.eos:
                break
            if i not in (self.bos, self.pad):
                out.append(self.itos[int(i)])
        return out


def build(smis, which: str):
    """Tokenize ``smis`` with tokenizer ``which``; return (lists, detok)."""
    tok_fn, detok = TOKENIZERS[which]
    lists = []
    for s in smis:
        try:
            t = tok_fn(s)
        except Exception:
            t = None
        if t:
            lists.append(t)
    return lists, detok
