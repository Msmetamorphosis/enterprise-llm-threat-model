"""
Main experiment runner.

Runs every trial under all 5 guardrail configs (see CONFIGURATIONS below)
and writes raw outcomes plus a summary to results/. Point of testing all
five instead of just the best one: we care about where each individual
defense earns its keep, not just the final number.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

from .guardrails import GuardrailStack
from .scenarios import ALL_SCENARIOS, TrialOutcome, run_all, summarize
from .secrets_registry import SecretsRegistry


RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


CONFIGURATIONS: Dict[str, Dict[str, bool]] = {
    "off": {"enable_input": False, "enable_provenance": False,
            "enable_output": False, "provenance_blocks": False},
    "input_only": {"enable_input": True, "enable_provenance": False,
                   "enable_output": False, "provenance_blocks": False},
    "input_output": {"enable_input": True, "enable_provenance": False,
                     "enable_output": True, "provenance_blocks": False},
    "full_stack": {"enable_input": True, "enable_provenance": True,
                   "enable_output": True, "provenance_blocks": False},
    "provenance_block": {"enable_input": True, "enable_provenance": True,
                          "enable_output": True, "provenance_blocks": True},
}


def run_all_configurations(results_dir: Path = RESULTS_DIR) -> Dict:
    # runs the full 14-trial suite once per config, dumps raw + summary JSON
    results_dir.mkdir(parents=True, exist_ok=True)
    combined: Dict = {}
    secrets = SecretsRegistry()

    for config_name, flags in CONFIGURATIONS.items():
        guardrails = GuardrailStack(secrets=secrets, **flags)
        outcomes = run_all(guardrails)
        summary = summarize(outcomes)

        raw_path = results_dir / f"raw_{config_name}.json"
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump([o.to_dict() for o in outcomes], f, indent=2)

        combined[config_name] = {"summary": summary,
                                 "outcomes": [o.to_dict() for o in outcomes]}

    combined_path = results_dir / "all_configurations.json"
    with combined_path.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    summary_path = results_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({k: v["summary"] for k, v in combined.items()}, f, indent=2)

    return combined


def print_summary(combined: Dict) -> None:
    print(f"{'config':<15} {'scenario':<12} {'n':>4} {'success':>8} "
          f"{'rate':>8} {'avg_turns':>10}")
    print("-" * 62)
    for config_name, block in combined.items():
        for scenario, stats in block["summary"]["per_scenario"].items():
            rate_pct = f"{stats['success_rate'] * 100:.0f}%"
            avg_turns = (f"{stats['avg_turns_to_success']:.1f}"
                         if stats['avg_turns_to_success'] is not None else "-")
            print(f"{config_name:<15} {scenario:<12} {stats['n_trials']:>4} "
                  f"{stats['n_success']:>8} {rate_pct:>8} {avg_turns:>10}")


if __name__ == "__main__":
    combined = run_all_configurations()
    print_summary(combined)
