"""Reversible rewrite rules and a sandboxed proposer for them.

A rule is a lossless token-list transform: ``encode(tokens) -> tokens``
and ``decode(tokens) -> tokens`` with ``decode(encode(x)) == x``. Rules
operate on token lists (not raw strings) so multi-character atoms like
``Cl``/``Br`` can never be mis-split.

FunSearch's mutation step: an LLM writes a new rule as code; it is
executed in a restricted namespace and admitted only if it passes a
round-trip gate on sample molecules. Correctness is enforced by the
gate, never trusted from the model -- so a wrong proposal is rejected,
not silently corrupting data.
"""

import re
from dataclasses import dataclass
from typing import Callable, Optional

TokenRule = Callable[[list], list]


@dataclass(frozen=True)
class ReversibleRule:
    name: str
    encode: TokenRule
    decode: TokenRule


REGISTRY: dict = {}


def register(rule: ReversibleRule) -> ReversibleRule:
    REGISTRY[rule.name] = rule
    return rule


def round_trip_ok(rule: ReversibleRule, token_lists, limit=500) -> bool:
    """Gate: decode(encode(x)) == x on sample token lists."""
    for toks in token_lists[:limit]:
        try:
            if rule.decode(rule.encode(list(toks))) != list(toks):
                return False
        except Exception:
            return False
    return True


# -- a hand-written (stand-in for LLM-proposed) rule -----------------

def _carbon_run_encode(tokens):
    """Collapse runs of >=2 aliphatic-carbon tokens 'C' into one 'C*n'."""
    out, i = [], 0
    while i < len(tokens):
        if tokens[i] == "C":
            j = i
            while j < len(tokens) and tokens[j] == "C":
                j += 1
            n = j - i
            out.append(f"C*{n}" if n > 1 else "C")
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out


def _carbon_run_decode(tokens):
    out = []
    for t in tokens:
        m = re.fullmatch(r"C\*(\d+)", t)
        out.extend(["C"] * int(m.group(1)) if m else [t])
    return out


carbon_run = register(ReversibleRule(
    "carbon_run", _carbon_run_encode, _carbon_run_decode))


# -- sandboxed code proposer -----------------------------------------

_SAFE_BUILTINS = {
    "len": len, "range": range, "enumerate": enumerate, "list": list,
    "dict": dict, "set": set, "tuple": tuple, "str": str, "int": int,
    "min": min, "max": max, "sorted": sorted, "reversed": reversed,
    "zip": zip, "map": map, "filter": filter, "abs": abs, "any": any,
    "all": all, "sum": sum,
}


def rule_from_code(name, code, gate_tokens):
    """Build and gate a rule from source defining encode(t)/decode(t).

    Executed with a restricted builtins set and only ``re`` importable.
    Returns a ReversibleRule if it round-trips, else None. This is the
    admission test an LLM proposal passes before entering the library.
    """
    ns = {"re": re, "__builtins__": _SAFE_BUILTINS}
    try:
        exec(code, ns)
        enc, dec = ns["encode"], ns["decode"]
    except Exception:
        return None
    rule = ReversibleRule(name, enc, dec)
    return rule if round_trip_ok(rule, gate_tokens) else None


_PROPOSER_SYSTEM = (
    "You invent reversible token-list rewrite rules that shorten "
    "molecular token streams without losing information. Given example "
    "token lists (SMILES atoms/bonds), write Python defining encode("
    "tokens) and decode(tokens), list->list, with decode(encode(x))==x "
    "for ALL inputs. The module `re` is preloaded -- use it without "
    "importing; no other imports are available. Target a redundancy BPE "
    "cannot capture (non-local structure), since local repeats are "
    "already handled. Reply with JSON {\"name\": str, \"code\": str}."
)


def llm_propose_rule(gate_tokens, examples, timeout=40):
    """Ask the Anthropic API for a new rule as code; gate it; return a
    ReversibleRule or None. Needs ANTHROPIC_API_KEY; the round-trip gate
    makes trusting the model unnecessary."""
    import json
    import os
    import urllib.request
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    base = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
    sample = "\n".join(str(t) for t in examples[:5])
    body = json.dumps({
        "model": os.environ.get("MOJITO_PROPOSER_MODEL", "claude-opus-4-8"),
        "max_tokens": 1024,
        "system": _PROPOSER_SYSTEM,
        "messages": [{"role": "user",
                      "content": f"Example token lists:\n{sample}\n"
                      "Propose one reversible rule as JSON."}],
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
    except Exception:
        return None
    return rule_from_code(spec["name"], spec["code"], gate_tokens)
