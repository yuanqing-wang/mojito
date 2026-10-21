"""Demonstrate the code-generating proposer end to end.

Shows the FunSearch admission flow on reversible rewrite rules:
- a correct LLM-written rule passes the round-trip gate and is used,
- a lossy rule is rejected by the gate (never trusted),
- grammars that compose the new rule are scored by the search.

With ANTHROPIC_API_KEY set and ``--llm``, the rule is written by the
Anthropic API; otherwise two illustrative rules stand in for the model.
"""

import argparse
import random

import pandas as pd

from mojito import rules
from mojito.grammar_search import Grammar, evaluate, grammar_tokens

# re is preloaded in the sandbox; the rule must not import it.
GOOD = r'''
def encode(tokens):
    out = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "c":
            j = i
            while j < len(tokens) and tokens[j] == "c":
                j += 1
            n = j - i
            out.append("c*%d" % n if n > 1 else "c")
            i = j
        else:
            out.append(tokens[i])
            i += 1
    return out
def decode(tokens):
    out = []
    for t in tokens:
        m = re.fullmatch(r"c\*(\d+)", t)
        if m:
            out.extend(["c"] * int(m.group(1)))
        else:
            out.append(t)
    return out
'''
BAD = '''
def encode(tokens):
    return [t for t in tokens if t != "C"]
def decode(tokens):
    return tokens
'''


def run(args):
    smis = pd.read_csv(args.data)["smiles"].str.strip().tolist()
    random.Random(args.seed).shuffle(smis)
    tr, ev = smis[:args.n_train], smis[args.n_train:args.n_train + 500]
    gate = [g for g in (grammar_tokens(s, False) for s in tr[:500]) if g]

    if args.llm:
        rule = rules.llm_propose_rule(gate, gate)
        print("LLM proposed rule admitted:", rule is not None)
    else:
        rule = rules.rule_from_code("arom_run", GOOD, gate)
        print("hand-written 'arom_run' admitted:", rule is not None)
        print("lossy 'drop_C' admitted:",
              rules.rule_from_code("drop_C", BAD, gate) is not None,
              "(expect False -- gate rejects it)")
    if rule:
        rules.register(rule)

    base = Grammar(True, args.k)
    tried = [base]
    if rule:
        tried.append(Grammar(True, args.k, False, (rule.name,)))
    for g in tried:
        r = evaluate(g, tr, ev)
        print(f"{g.label():30s} fit {r['fitness']:.1f}  "
              f"corpus {r['corpus_bits_per_mol']:.1f}  "
              f"tok {r['tokens_per_mol']:.1f}  vocab {r['vocab']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data")
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--k", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--llm", action="store_true",
                   help="use the Anthropic API proposer (needs API key)")
    run(p.parse_args())
