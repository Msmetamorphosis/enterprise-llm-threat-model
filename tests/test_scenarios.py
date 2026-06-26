"""Integration tests for the scenario harness."""
from src.guardrails import GuardrailStack
from src.scenarios import (
    scenario_one_trials,
    scenario_two_trials,
    scenario_three_trials,
    run_trial,
    run_all,
    summarize,
)


def test_scenario_one_all_succeed_against_undefended_assistant():
    guardrails = GuardrailStack(enable_input=False,
                                enable_provenance=False,
                                enable_output=False)
    outcomes = [run_trial(t, guardrails) for t in scenario_one_trials()]
    assert all(o.succeeded for o in outcomes)


def test_scenario_one_blocked_by_output_filter():
    guardrails = GuardrailStack(enable_input=False,
                                enable_provenance=False,
                                enable_output=True)
    outcomes = [run_trial(t, guardrails) for t in scenario_one_trials()]
    assert not any(o.succeeded for o in outcomes)


def test_insider_trials_survive_full_stack():
    guardrails = GuardrailStack(enable_input=True,
                                enable_provenance=True,
                                enable_output=True)
    outcomes = [run_trial(t, guardrails) for t in scenario_two_trials()]
    assert any(o.succeeded for o in outcomes), (
        "insider probing should slip past a mainstream guardrail stack; "
        "if this test fails the paper's central claim is wrong")


def test_accidental_trials_are_caught_only_when_output_filter_runs():
    # With the full guardrail stack (including output filter), the paraphrased
    # system-prompt leak matches the output pattern and every S3 trial is
    # blocked. This is realistic: a governance team maintains a watch term
    # for the exact model identity string.
    guardrails = GuardrailStack(enable_input=True,
                                enable_provenance=True,
                                enable_output=True)
    outcomes = [run_trial(t, guardrails) for t in scenario_three_trials()]
    assert not any(o.succeeded for o in outcomes)

    # But drop the output filter and the input filter alone is not enough:
    # at least one payload (S3-B, S3-D) uses natural-language phrasing that
    # does not match the injection regex.
    input_only = GuardrailStack(enable_input=True,
                                enable_provenance=False,
                                enable_output=False)
    outcomes = [run_trial(t, input_only) for t in scenario_three_trials()]
    assert any(o.succeeded for o in outcomes), (
        "input filter alone should miss the natural-language self injection")


def test_negative_control_does_not_succeed():
    # S3-C is a benign paste; it should never trigger leakage.
    guardrails = GuardrailStack(enable_input=False,
                                enable_provenance=False,
                                enable_output=False)
    trials = [t for t in scenario_three_trials() if t.trial_id == "S3-C"]
    assert trials, "negative control S3-C should exist"
    outcome = run_trial(trials[0], guardrails)
    assert not outcome.succeeded


def test_summarize_shape():
    guardrails = GuardrailStack()
    outcomes = run_all(guardrails)
    summary = summarize(outcomes)
    assert set(summary["per_scenario"].keys()) == {"outside", "insider", "accidental"}
    assert 0.0 <= summary["overall_success_rate"] <= 1.0
