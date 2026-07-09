# Methodology and process notes

Working notes, not a polished writeup. This is roughly what we told each other in Slack and then cleaned up enough to be readable.

## Framing (why we narrowed the question)

We started with "build a threat model for a copilot-style assistant" as the pitch. Too broad, honestly, and we knew it within the first week. What we actually wanted to know: given the kind of assistant regulated orgs are deploying right now, and given the kind of guardrails those orgs have actually bothered to turn on, which attacks get through and which don't. That's a question you can put a number on. So that's what we built toward.

## What we built

The assistant is small on purpose. A real production copilot has a dozen tool calls, retrieval over a huge corpus, memory across sessions, and a model picked for capability rather than for being predictable. Every one of those things makes it harder to point at a failure and say "that's why." We stripped it down to the minimum that still shows the behaviors we cared about: one system prompt naming three confidential facts, TF-IDF retrieval over a small local corpus, one model backend, and a guardrail layer we can flip on and off per configuration.

The three confidential facts are the exact approval threshold for exceptions, an internal escalation phone number, and the exact wording of a KYC override policy (POL-014). These stand in for the kind of operational detail a regulated org has real reasons to keep out of routine assistant answers. Both the model backend and the output filter read from the same registry of these facts, so leak categories line up across scenarios and configs (this matters for the figures later).

## Why the simulator

The default backend is a deterministic simulator. It reacts to structural cues, secrets present in retrieved context, imperative instructions sitting inside retrieved documents, numeric probes in the user's message, that kind of thing. Not a language model. We knew this was going to need defending, so here's the actual case for it.

Our claim is about which classes of attack succeed against a typical guardrail setup, not about what one specific frontier model happens to say on one specific day. A live LLM adds two nuisance variables, model version drift and sampling noise, that make comparing across defense configs noisier without adding any signal to the question we actually care about. Whether the guardrail stack catches the attack before it lands is answered better by a backend whose response to a given prompt shape is known ahead of time. In practice this also meant we could iterate on the guardrail regexes without burning API credits every time we wanted to rerun the suite.

The obvious objection: the simulator isn't a language model, and a real one might behave differently. Fair, and we don't really dispute it. What we did to blunt that: built the simulator around the same cues a real model plausibly reacts to (a system prompt marking data confidential, that data showing up in retrieved context, and the tendency for models to follow instructions embedded in what looks like ordinary content). We also went back and ran a real-LLM validation pass as a complement, not a replacement, using `src/llm_validation.py` through `pplx llm extract`. Results in `results/llm_validation/`. The qualitative pattern holds up: same classes of attack succeed and fail in the same rough places. The differences are interesting on their own (see the appendix), especially around PII leakage the simulator has no way to produce because it just doesn't model that category.

## Guardrail configs (why these 5)

We picked five configurations to bracket what's actually deployed today, not what a research paper would recommend. `input_only` is the modal enterprise posture right now: regex over adversarial phrasing, run against the user message and retrieved context. `input_output` and `full_stack` are the natural next steps a governance team adds after their first indirect-injection incident scares them. `provenance_block` is a content-aware step some orgs are starting to try but that isn't standard yet.

We left out research-grade defenses (spotlighting, structured queries, provenance-signed retrieval) because governance teams mostly haven't deployed them yet, and testing defenses nobody uses felt like it was answering a different question than the one we set out to answer. The framework could be extended to cover them. We just didn't, this time around.

## Attack corpus (why these 3 scenarios, and the negative control)

Three scenarios, chosen to hit different parts of the threat model rather than variations on the same idea.

Scenario one is the loud, obvious attack pattern, the one most of the defensive literature is already oriented around. We made the payload deliberately unsubtle. The question we wanted answered was whether an output filter handles the strong version of this attack, not whether it handles a sneaky weak one. That's a separate (also interesting) question we didn't chase.

Scenario two is the one that intent-based defenses are structurally blind to, because nothing in the message stream looks adversarial. The binary-search sequence is the main version. We also threw in a policy-number lookup and a softer two-turn hypothetical, just to check the result wasn't an artifact of numeric probing specifically.

Scenario three has no adversary present at all when the advisor acts. This one matters most for governance conversations because it breaks the assumption that "the attacker" is a person doing something on purpose. We included a negative control (a plain email with nothing hidden in it) to make sure the assistant doesn't have a runaway false-positive problem when nothing bad is actually happening.

Each trial's success criterion is something we can check mechanically. Pass or fail, no eyeballing required.

## Corpus size

Small on purpose: eleven internal documents, two untrusted ones. We talked about scaling it up and decided against it. A bigger corpus makes retrieval quality noisier and makes it harder to trace an outcome back to a specific document or payload. We took the trade toward smaller and more legible, and we're accepting the hit to how much this generalizes statistically. That's a real limitation, not something we're glossing over.

## Reproducibility

`python -m src.run_experiments` writes the raw and summarized JSON to `results/`. `python -m src.make_figures` builds the figures from that JSON. Both run in well under a minute on a laptop, no GPU, no API key required for the default path. The report and slide deck live outside this repo now and get rebuilt from the figures separately.

## Real LLM validation

We later ran the same trials against a real LLM via `src/llm_validation.py`, using `pplx llm extract` as the transport. Results live in `results/llm_validation/`. Qualitatively it lines up with the simulator's picture, same attacks succeed, same ones mostly don't, with the differences (the LLM refusing some insider trials, and leaking PII in some outside-attacker trials the simulator never modeled) noted in the appendix of the report.

## Things we skipped and why

- Multi-agent orchestration. Real copilots route through several agents; ours has one. Adding orchestration opens up genuinely interesting attack paths (agent-to-agent injection, tool-use manipulation) but it would also make the guardrail comparison a lot harder to read. We didn't do it because it wasn't worth the headache for this project's scope.
- Session memory. Real copilots keep state across sessions. Ours doesn't. Memory-based attacks feel like their own research project, honestly.
- Adversarial suffix attacks, GCG-style. Model-specific, research-grade, not what a governance team is actually defending against today. The deterministic simulator also has no real way to model them faithfully, so it would've been a fake number if we'd tried.

Any of these would be a reasonable follow-up if someone wanted to extend this.
