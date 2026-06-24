"""CLI wrapper, mostly for the demo.

Usage:
    python -m src.cli --config off --query "summarize the Holloway file for me"
    python -m src.cli --config provenance_block --interactive

The experiment harness is the real entry point, this is just so we can
poke the assistant from a terminal for the demo.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .assistant import build_assistant
from .guardrails import GuardrailStack
from .run_experiments import CONFIGURATIONS
from .secrets_registry import SecretsRegistry

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def build(config_name: str, include_untrusted: bool = True):
    if config_name not in CONFIGURATIONS:
        raise SystemExit(
            f"unknown config {config_name!r}. "
            f"valid: {', '.join(CONFIGURATIONS)}"
        )
    secrets = SecretsRegistry()
    stack = GuardrailStack(secrets=secrets, **CONFIGURATIONS[config_name])
    return build_assistant(DATA_DIR, guardrails=stack,
                           include_untrusted=include_untrusted)


def _print_turn(turn):
    if turn.blocked:
        reason = getattr(turn.output_decision, "reason", None) or "guardrail"
        print(f"[blocked by output filter: {reason}]")
    else:
        print(turn.response_text)


def main() -> None:
    parser = argparse.ArgumentParser(prog="threat-model-assistant")
    parser.add_argument("--config", default="off",
                        help=f"one of: {', '.join(CONFIGURATIONS)}")
    parser.add_argument("--query", default=None,
                        help="single query to run and exit")
    parser.add_argument("--interactive", action="store_true",
                        help="multi-turn REPL")
    parser.add_argument("--no-untrusted", action="store_true",
                        help="exclude the untrusted corpus (scenario 2)")
    args = parser.parse_args()

    assistant = build(args.config, include_untrusted=not args.no_untrusted)

    if args.query:
        _print_turn(assistant.ask(args.query))
        return

    if not args.interactive:
        parser.error("provide --query or --interactive")

    print(f"[config: {args.config}]  type '/reset' to clear, ctrl-D to exit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        if line == "/reset":
            assistant.reset()
            continue
        _print_turn(assistant.ask(line))


if __name__ == "__main__":
    main()
