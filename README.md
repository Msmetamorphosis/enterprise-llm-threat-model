# CS7545 semester project

Threat model of an enterprise conversational AI assistant.

Real writeup coming later, just tracking scope for now.

## Scope

- one sandboxed RAG assistant, small synthetic corpus
- three adversary scenarios (outside injection, insider probing, accidental self-injection)
- five guardrail configs (off, input_only, input_output, full_stack, provenance_block)
- results dumped as JSON, figures generated from that

## TODO

- corpus (client notes, policies, calendar, untrusted docs)
- assistant + retriever + guardrails
- scenarios + trial harness
- tests
- experiment runner
- figures
- writeup + slides
