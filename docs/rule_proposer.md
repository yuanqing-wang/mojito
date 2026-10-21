# Code-generating rule proposer (FunSearch's mutation step)

The strongest lever left is new *reversible rewrite rules* that remove
redundancy BPE and DeepSMILES miss. This builds the FunSearch mechanism
for discovering them: an LLM writes a rule as code, a round-trip gate
admits it only if lossless, and the search composes it.

## Pieces (`mojito/rules.py`)
- `ReversibleRule`: a lossless token-list transform (encode/decode),
  operating on *tokens* not raw strings so `Cl`/`Br` cannot be mis-split.
- `rule_from_code(name, code, gate_tokens)`: executes LLM-written source
  in a **restricted namespace** (whitelisted builtins; `re` preloaded, no
  other imports) and admits the rule only if `decode(encode(x)) == x` on
  sample molecules. Correctness is enforced by the gate, never trusted.
- `llm_propose_rule(...)`: the Anthropic API call that writes the rule
  (needs `ANTHROPIC_API_KEY`); returns a gated rule or None.
- Grammars carry an `extra` tuple of rule names, composed after
  DeepSMILES/Kekule and before BPE; the search's round-trip gate covers
  the whole pipeline.

## Demonstrated (`scripts/demo_rule_proposer.py`, ZINC)
- A correct rule (collapse aromatic-`c` runs) **is admitted** and used.
- A lossy rule (drops `C` tokens) **is rejected** by the gate.
- The admitted rule composes into the grammar and is scored.

```
arom_run admitted: True
drop_C   admitted: False   (gate rejects it)
deep+bpe512            fit 64.0  tok 8.5  vocab 552
deep+arom_run+bpe512   fit 64.2  tok 8.6  vocab 561
```

## Honest result
The illustrative rules are **neutral-to-slightly-worse** -- they target
*local* repeats that BPE already captures. This is the expected and
important finding: value lives in *non-local* reversible rules (the kind
DeepSMILES' ring-closure removal is), which are hard to write correctly
by hand -- exactly why a strong LLM proposer exploring under the gate is
the right tool, and why gains past DeepSMILES are uncertain, not
guaranteed. The machinery is proven; the open question is whether an LLM
can invent a *winning* non-local rule. That run needs an API key (for the
proposer) and GPU (converged LM fitness instead of the bigram proxy).

## Safety note
`rule_from_code` restricts builtins and blocks imports, and the inputs
are molecule token lists. This is not a hardened sandbox; a production
run should execute proposals in an isolated subprocess/container.
