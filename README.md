# Threat Modeling Enterprise Conversational AI

CS7545 AI Security. Semester project. Summer 2026.

---

## The Problem

More than 10,000 employees at a large regulated financial services company regularly use conversational AI: Microsoft Copilot, ChatGPT Enterprise, and a handful of internally built RAG assistants. Three critical-severity vulnerabilities got disclosed against Microsoft 365 Copilot in 2026 alone. A separate piece of research on custom GPTs found a 97% success rate for extracting hidden system instructions and a 100% success rate for pulling uploaded knowledge files out, using nothing but plain-language prompts.

Most of the security writing we read this semester was about autonomous agents. Fine, agents are scary, but the thing actually sitting in front of ten thousand people every day is a chat assistant that reads your files and answers questions. That gets treated like it's basically solved. It isn't.

So we picked a narrower question: if an org's advisors and staff are already using Copilot, ChatGPT Enterprise, and custom GPTs, who's actually attacking that surface, and what does the attack look like in practice?

---

## Approach

We built one sandboxed RAG assistant that stands in for an internal advisor support tool. It has a small synthetic corpus (client notes, policy docs, calendar entries, plus a couple of untrusted incoming documents) and we run three adversary scenarios against it. Each scenario is a different threat actor with a different vector and a different goal. Then we run five defense configurations across all of them to see which defense actually stops which attack.

Everything in `data/` is made up. No real institution, client, or employee shows up anywhere in this repo.

---

## The Three Adversary Scenarios

### Scenario A: The Outside Attacker
An external actor with zero access to internal systems hides an indirect prompt injection payload inside a PDF. That document later gets pulled into the assistant's retrieval context when an advisor asks something totally unrelated to the attack. The advisor's own query is benign the whole time.

**Attack class:** Indirect prompt injection via retrieved document context
**Goal:** Exfiltrate whatever is sitting in the assistant's connected knowledge base
**Why it matters now:** Same architectural pattern behind EchoLeak (CVE-2025-32711) and SearchLeak (CVE-2026-42824), both disclosed against production Microsoft Copilot deployments in 2025 and 2026.

### Scenario B: The Insider Probing for Boundaries
An authenticated employee who already has legitimate access asks a short run of hypothetical policy questions. None of them directly ask for the confidential number they want. Over a handful of turns they bracket a threshold value the system prompt explicitly marks as restricted.

**Attack class:** Inferential threshold extraction via multi-turn probing
**Goal:** Reconstruct a confidential operational constant (an approval limit, a review threshold) without ever tripping a direct-disclosure refusal
**Why it matters now:** Almost nobody treats threshold inference as its own attack category, but in financial services, knowing the exact threshold is basically a blueprint for structuring around it. Guardrails built for stopping direct disclosure don't touch this.

### Scenario C: Accidental Self-Inflicted Injection
Nobody targets the firm on purpose here. An advisor pastes a routine client email into the assistant and asks for a summary. The email has a hidden injection payload planted by a scammer who guessed (correctly) that the recipient uses an AI assistant. The advisor had no reason to think twice about the email.

**Attack class:** Supply chain injection via everyday advisor workflow
**Goal:** Get the assistant to follow attacker instructions without any direct system access and without any malicious intent on the insider's part
**Why it matters now:** No special access needed, and it rides on the most ordinary workflow you can imagine. It's also invisible to any defense that tries to classify threat actors by intent, because from the system's point of view a trusted user typed every word.

---

## The Five Defense Configurations

| ID | Configuration | What it does |
|---|---|---|
| `off` | No guardrails | Baseline; all protections disabled |
| `input_only` | Input filter | Regex detector scans user message and retrieved context for adversarial phrasing |
| `input_output` | Input and output filter | Adds last-line output detector for known secret values leaving the assistant |
| `full_stack` | Full stack with logging | Adds provenance filter that flags untrusted retrieved documents; logs but does not remove |
| `provenance_block` | Full stack with blocking | Same as full stack but flagged documents are removed from context entirely before the model sees them |

---

## Results

| Configuration | Scenario A: Outside Attacker | Scenario B: Insider Probing | Scenario C: Accidental Injection |
|---|---|---|---|
| `off` | 100% | 100% | 75% |
| `input_only` | 100% | 100% | 75% |
| `input_output` | 0% | 60% | 0% |
| `full_stack` | 0% | 60% | 0% |
| `provenance_block` | 0% | 60% | 0% |

Turning the output filter on takes the outside attacker to 0% and the accidental self-injection to 0%. Insider probing drops from 100% to 60% and then just sits there, no matter what else we turn on. That stuck 60% is really the finding here. It's three trials where the attacker reconstructs a confidential threshold from a run of yes/no answers to hypothetical dollar amounts. The assistant never says the secret out loud, so there's nothing for a content filter to match against.

That's architectural, not a bug we didn't get around to fixing. The guardrails in this stack are built to catch a secret leaking in the response text, or an adversarial payload sitting in retrieved context. The three residual insider trials do neither of those things. The user is authenticated, no untrusted document ever enters the pipeline, and the assistant only ever says "yes" or "no" to a hypothetical amount. The attacker infers the threshold from the pattern of answers. Nothing gets extracted from the model in the sense these filters check for.

---

## Real LLM validation

We didn't want to just trust the simulator's story, so we also ran the same 14 trials against a real LLM through `pplx llm extract`, via `src/llm_validation.py`. Results are in `results/llm_validation/`. The LLM actually refused a couple of insider trials that the simulator happily succeeded on, which was a little reassuring. On the flip side, it leaked client PII (account numbers, identity details) in some outside-attacker trials that the deterministic simulator never modeled at all, since the simulator doesn't have any notion of PII beyond the categories we hardcoded. Overall the qualitative pattern holds: real LLM, same shape of results.

---

## Repository Structure

```
src/                    assistant core, model backends, guardrail stack, scenario harness
data/
  synthetic/            client notes, policy docs, calendar entries (all fabricated)
  untrusted/            two documents carrying indirect injection payloads (Scenario A only)
tests/                  unit tests for retrieval, guardrails, and scenario harness
results/                raw JSON and summarized CSV from a full experiment run
results/llm_validation/ transcripts and summary from the real-LLM validation run
figures/                four figures used in the final report
docs/
  methodology.md         testbed design decisions, modeling choices, and rationale
  llm_run_notes.md       notes from the real-LLM validation run
```

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q
python -m src.run_experiments
python -m src.make_figures
```

`run_experiments` reruns all 14 trials across all 5 defense configurations, writes JSON to `results/`, and finishes in under a second on a laptop with no GPU. No API key needed for the default backend.

---

## On the Model Backend

The default backend is a deterministic simulator. Its response rules react to the structural stuff that actually matters for these attacks: secrets sitting in retrieved context, imperative instructions buried in retrieved documents, numeric probes in the user message. We went with this on purpose, mainly so the results are fully reproducible without an API key, and so every outcome traces back to a specific rule you can actually go read, instead of "the model felt like it that day."

There's an OpenAI-compatible backend in `src/model_backends.py` that turns on when `OPENAI_API_KEY` is set. Running the same scenarios against a real frontier model was an obvious next step, and we did a version of that (see Real LLM validation above).

---

## References

- Greshake et al. 2023. Not What You've Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injections. arXiv:2302.12173
- Perez and Ribeiro 2022. Ignore Previous Prompt: Attack Techniques for Language Models. arXiv:2211.09527
- Reddy and Gujral 2025. EchoLeak: Zero-Click AI Data Exfiltration from Microsoft 365 Copilot. arXiv:2509.10540 (CVE-2025-32711)
- Yu et al. 2023. Assessing Prompt Injection Risks in 200+ Custom GPTs. arXiv:2311.11538
- OWASP Top 10 for LLM Applications, 2025 revision
- NIST AI Risk Management Framework 2.0, January 2025

---

## Data and Ethics Statement

Everything in `data/` is made up. Names, account numbers, dollar amounts, policy IDs, phone numbers, calendar entries, all fabricated for this project. We never targeted or tested any real institution, client, employee, or system. Scenarios A, B, and C ran entirely inside the sandbox in this repo, start to finish.
