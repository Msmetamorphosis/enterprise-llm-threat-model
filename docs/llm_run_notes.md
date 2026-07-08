# LLM validation run notes

Quick record of when and how we ran the real-LLM validation.

## What we ran

`python -m src.llm_validation` hits a real LLM through the `pplx llm extract`
CLI and scores replies using the same `GuardrailStack.check_output()` the
deterministic runner uses. 14 trials, 21 turns total. Cost was well under a
dollar.

## What it wrote

- `results/llm_validation/transcripts.json`: every turn with the LLM's actual
  reply, the guardrail decision, and any leaked secret categories.
- `results/llm_validation/summary.json`: per-scenario and per-trial totals.
- `results/llm_validation/errors.json`: empty in our run. Would collect any
  turn where the model call errored.

## Headline

With the output filter enabled, real-LLM results (out of 14 trials):

- Outside attacker: 2/5 succeeded. Both were PII leaks the deterministic
  simulator didn't model (account numbers, tenure, AUM).
- Insider probing: 2/5 succeeded. Threshold inference from paraphrased
  yes/no answers, exactly the residual failure the paper argues about.
- Accidental injection: 0/4 succeeded. The real LLM refused the embedded
  instructions in all four accidental-injection prompts.

## Why the numbers differ from the simulator

The simulator was written to react to fixed structural cues, so it succeeds
on every insider trial once the direct-quote defense is bypassed. A real
LLM refuses the naive "paste the policy" ask (S2-B, S2-D) and only leaks on
the softer threshold probe. Meanwhile the real model leaks structural PII
(client names, account markers) in outside-attacker trials because the
output filter only knows about the three registered secrets. Both differences
strengthen the paper's core point: an output filter tuned to specific
strings doesn't catch what it wasn't told about.

We didn't rerun the whole 70-run configuration matrix against the real LLM
(the point of the simulator was to give us the matrix cheaply). This one
run is a sanity check that the qualitative pattern holds.
