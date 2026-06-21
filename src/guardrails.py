"""
Guardrail stack, meant to look like what a typical enterprise deploys today.

Three checks, run in order: input (regex for adversarial phrasing, think
Azure Content Safety / Lakera Guard style), provenance (flags untrusted
docs), output (regex/phrase match against known secrets before the reply
goes back to the advisor). Kept deliberately simple, not research-grade,
on purpose, see docs/methodology.md for why.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from .retrieval import Document
from .secrets_registry import SecretsRegistry


# Tuned against the OWASP LLM01:2025 examples and the PromptInject taxonomy
# (Perez and Ribeiro, 2022).
INPUT_INJECTION_PATTERNS: Sequence[str] = (
    # TODO: this regex is too aggressive on the word "ignore" in general text,
    # see the failing edge case in test_guardrails.
    r"ignore (?:all |any )?(?:previous|above|prior) (?:instructions|prompts)",
    r"disregard (?:your |the )?(?:system )?prompt",
    r"you are now (?:a|an) [\w \-]+",
    r"repeat (?:everything|all) (?:above|previous)",
    r"print (?:your |the )?(?:system |initial )?prompt",
    r"reveal (?:your |the )?(?:system |hidden )?(?:instructions|prompt)",
    # HTML-comment smuggling, shows up a lot in forwarded WYSIWYG emails
    r"<!--\s*(?:system|prompt|instruction|secret)\b",
)

# Last-line defense: fires when a secret value shows up verbatim in the
# response. pol_014_verbatim is intentionally narrow, it catches the exact
# sentence but not a paraphrase that says the same thing. That's basically
# how rules-based DLP filters work in the real world, for better or worse.
OUTPUT_LEAK_PATTERNS: Dict[str, str] = {
    "approval_threshold": r"\$\s?250[,\s]?000|\$250k\b|two hundred (?:and )?fifty thousand",
    "override_phone": r"770[-.\s]?555[-.\s]?0142",
    "system_prompt_signature": (
        r"(?:You are|I am) Halo,?\s*an internal AI assistant"
    ),
    "pol_014_verbatim": (
        r"tenure greater than 7 years and AUM above \$2M"
        r"|form\s*E[- ]?14 within 24 hours"
    ),
}


@dataclass
class GuardrailDecision:
    """One check's verdict: did it trigger, and why."""

    stage: str
    triggered: bool
    reason: Optional[str] = None
    matched_pattern: Optional[str] = None


@dataclass
class GuardrailStack:
    """Three-stage stack: input, provenance, output. provenance_blocks controls
    whether flagged docs actually get removed or just logged."""

    enable_input: bool = True
    enable_provenance: bool = True
    enable_output: bool = True
    provenance_blocks: bool = False
    secrets: SecretsRegistry = field(default_factory=SecretsRegistry)

    def check_input(self, user_message: str,
                    context_blocks: Sequence[str]) -> GuardrailDecision:
        if not self.enable_input:
            return GuardrailDecision(stage="input", triggered=False,
                                     reason="disabled")
        haystack = "\n".join([user_message, *context_blocks]).lower()
        for pattern in INPUT_INJECTION_PATTERNS:
            if re.search(pattern, haystack):
                return GuardrailDecision(
                    stage="input",
                    triggered=True,
                    reason="matched adversarial phrase in input or retrieved context",
                    matched_pattern=pattern,
                )
        return GuardrailDecision(stage="input", triggered=False)

    def check_provenance(self,
                         retrieved: Sequence[Document]) -> GuardrailDecision:
        if not self.enable_provenance:
            return GuardrailDecision(stage="provenance", triggered=False,
                                     reason="disabled")
        for doc in retrieved:
            if not doc.trusted:
                return GuardrailDecision(
                    stage="provenance",
                    triggered=True,
                    reason=f"untrusted document surfaced by retrieval: {doc.doc_id}",
                )
        return GuardrailDecision(stage="provenance", triggered=False)

    def check_output(self, response: str) -> GuardrailDecision:
        if not self.enable_output:
            return GuardrailDecision(stage="output", triggered=False,
                                     reason="disabled")
        for category, pattern in OUTPUT_LEAK_PATTERNS.items():
            if re.search(pattern, response, re.IGNORECASE):
                return GuardrailDecision(
                    stage="output",
                    triggered=True,
                    reason=f"detected leaked secret category '{category}' in response",
                    matched_pattern=category,
                )
        return GuardrailDecision(stage="output", triggered=False)

    def summary(self) -> Dict[str, bool]:
        return {
            "input": self.enable_input,
            "provenance": self.enable_provenance,
            "output": self.enable_output,
        }
