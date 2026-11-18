"""Tests for the primary failure taxonomy, using hand-built trajectories."""

from __future__ import annotations

from typing import Any

import pytest

from passk.schema import AgentAction, FailureCategory, Step, ToolCall, ToolResult, Trajectory
from passk.taxonomy import classify, classify_trajectory, is_loop

MAX_TURNS = 8


def _call(
    index: int,
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    ok: bool = True,
    error: str | None = None,
) -> Step:
    """A tool-call step whose result is ``ok`` or fails with ``error``."""
    args = {} if arguments is None else arguments
    call = ToolCall(call_id=f"c{index}", name=name, arguments=args)
    result = ToolResult(
        call_id=f"c{index}",
        ok=ok,
        content=None if not ok else {"status": "ok"},
        error=error,
    )
    return Step(
        index=index,
        action=AgentAction(kind="tool_call", tool_call=call),
        result=result,
        latency_s=0.0,
    )


def _final(index: int, message: str = "All done.") -> Step:
    """A ``final`` step, which has no tool result."""
    return Step(
        index=index,
        action=AgentAction(kind="final", message=message),
        result=None,
        latency_s=0.0,
    )


def _distinct_calls(count: int) -> list[Step]:
    """``count`` successful calls with distinct arguments, so none is a loop."""
    return [_call(i, "lookup", {"compound_id": f"C{i:02d}"}) for i in range(count)]


# Success ---------------------------------------------------------------------------


def test_success_when_goal_met_and_no_violations() -> None:
    steps = [_call(0, "get_compound", {"compound_id": "C07"}), _final(1)]
    assert (
        classify(steps, violations=[], goal_met=True, claimed_done=True, max_turns=MAX_TURNS)
        is FailureCategory.SUCCESS
    )


def test_success_keeps_recovered_errors_and_loops_out_of_the_failure_chain() -> None:
    # A passing episode is never labelled as a failure, so that the label agrees with
    # Trajectory.success (see schema.py). The steps still show what happened.
    steps = [
        _call(0, "order", {"grams": 5}, ok=False, error="out of stock"),
        _call(1, "order", {"grams": 5}, ok=False, error="out of stock"),
        _call(2, "order", {"grams": 5}),
        _final(3),
    ]
    assert (
        classify(steps, violations=[], goal_met=True, claimed_done=True, max_turns=MAX_TURNS)
        is FailureCategory.SUCCESS
    )


# One hand-built trajectory per category ---------------------------------------------


def test_policy_violation() -> None:
    steps = [_call(0, "refund", {"amount": 900}), _final(1)]
    assert (
        classify(
            steps,
            violations=["refund above the 500 limit"],
            goal_met=False,
            claimed_done=True,
            max_turns=MAX_TURNS,
        )
        is FailureCategory.POLICY_VIOLATION
    )


def test_schema_error_on_invalid_argument() -> None:
    steps = [
        _call(0, "set_quantity", {"grams": "five"}, ok=False, error="invalid argument: grams"),
        _final(1),
    ]
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=True, max_turns=MAX_TURNS)
        is FailureCategory.SCHEMA_ERROR
    )


def test_schema_error_on_unknown_tool() -> None:
    steps = [_call(0, "remove_item", ok=False, error="Unknown tool: remove_item"), _final(1)]
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=False, max_turns=MAX_TURNS)
        is FailureCategory.SCHEMA_ERROR
    )


def test_tool_error_unrecovered() -> None:
    steps = [
        _call(0, "get_compound", {"compound_id": "C99"}, ok=False, error="compound not found"),
        _final(1),
    ]
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=False, max_turns=MAX_TURNS)
        is FailureCategory.TOOL_ERROR_UNRECOVERED
    )


def test_false_success_claim() -> None:
    steps = [
        _call(0, "get_compound", {"compound_id": "C07"}),
        _call(1, "list_stock", {}),
        _final(2, "Ordered 5 g of C07."),
    ]
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=True, max_turns=MAX_TURNS)
        is FailureCategory.FALSE_SUCCESS_CLAIM
    )


def test_loop_three_identical_consecutive_calls() -> None:
    steps = [_call(i, "get_compound", {"compound_id": "C07"}) for i in range(3)]
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=False, max_turns=MAX_TURNS)
        is FailureCategory.LOOP
    )


def test_max_turns_when_budget_used_without_loop_or_claim() -> None:
    steps = _distinct_calls(MAX_TURNS)
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=False, max_turns=MAX_TURNS)
        is FailureCategory.MAX_TURNS
    )


def test_wrong_state_when_episode_stops_with_unmet_goal() -> None:
    steps = _distinct_calls(2)
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=False, max_turns=MAX_TURNS)
        is FailureCategory.WRONG_STATE
    )


# Precedence conflicts ---------------------------------------------------------------


def test_policy_violation_beats_schema_error() -> None:
    steps = [_call(0, "refund", {"amount": "x"}, ok=False, error="invalid argument: amount")]
    assert (
        classify(
            steps,
            violations=["refund blocked"],
            goal_met=False,
            claimed_done=True,
            max_turns=MAX_TURNS,
        )
        is FailureCategory.POLICY_VIOLATION
    )


def test_schema_error_beats_unrecovered_tool_error() -> None:
    steps = [
        _call(0, "set_quantity", {"grams": "five"}, ok=False, error="invalid argument: grams"),
        _call(1, "get_compound", {"compound_id": "C99"}, ok=False, error="compound not found"),
        _final(2),
    ]
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=True, max_turns=MAX_TURNS)
        is FailureCategory.SCHEMA_ERROR
    )


def test_tool_error_unrecovered_beats_false_success_claim() -> None:
    steps = [
        _call(0, "get_compound", {"compound_id": "C99"}, ok=False, error="compound not found"),
        _final(1, "Done."),
    ]
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=True, max_turns=MAX_TURNS)
        is FailureCategory.TOOL_ERROR_UNRECOVERED
    )


def test_retried_tool_error_is_recovered_so_claim_stands() -> None:
    steps = [
        _call(0, "get_compound", {"compound_id": "C99"}, ok=False, error="compound not found"),
        _call(1, "get_compound", {"compound_id": "C07"}),
        _final(2),
    ]
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=True, max_turns=MAX_TURNS)
        is FailureCategory.FALSE_SUCCESS_CLAIM
    )


def test_success_by_a_different_tool_does_not_recover_an_error() -> None:
    steps = [
        _call(0, "get_compound", {"compound_id": "C99"}, ok=False, error="compound not found"),
        _call(1, "list_stock", {}),
        _final(2),
    ]
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=True, max_turns=MAX_TURNS)
        is FailureCategory.TOOL_ERROR_UNRECOVERED
    )


def test_false_success_claim_beats_loop() -> None:
    steps = [_call(i, "get_compound", {"compound_id": "C07"}) for i in range(3)]
    steps.append(_final(3))
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=True, max_turns=MAX_TURNS)
        is FailureCategory.FALSE_SUCCESS_CLAIM
    )


def test_loop_beats_max_turns() -> None:
    steps = [_call(i, "get_compound", {"compound_id": "C07"}) for i in range(MAX_TURNS)]
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=False, max_turns=MAX_TURNS)
        is FailureCategory.LOOP
    )


def test_max_turns_beats_wrong_state() -> None:
    steps = _distinct_calls(MAX_TURNS + 2)
    assert (
        classify(steps, violations=[], goal_met=False, claimed_done=False, max_turns=MAX_TURNS)
        is FailureCategory.MAX_TURNS
    )


# is_loop ----------------------------------------------------------------------------


def test_is_loop_needs_three_identical_consecutive_calls_by_default() -> None:
    same = [_call(i, "get", {"id": 1}) for i in range(2)]
    assert is_loop(same) is False
    assert is_loop(same + [_call(2, "get", {"id": 1})]) is True


def test_is_loop_requires_same_name_and_arguments() -> None:
    different_args = [
        _call(0, "get", {"id": 1}),
        _call(1, "get", {"id": 2}),
        _call(2, "get", {"id": 1}),
    ]
    different_name = [
        _call(0, "get", {"id": 1}),
        _call(1, "list", {"id": 1}),
        _call(2, "get", {"id": 1}),
    ]
    assert is_loop(different_args) is False
    assert is_loop(different_name) is False


def test_is_loop_needs_consecutive_calls() -> None:
    alternating = [_call(i, "a" if i % 2 == 0 else "b", {}) for i in range(5)]
    assert is_loop(alternating) is False


def test_is_loop_detects_a_run_before_other_calls() -> None:
    steps = [_call(0, "a", {}), _call(1, "a", {}), _call(2, "a", {}), _call(3, "b", {})]
    assert is_loop(steps) is True


def test_is_loop_final_action_breaks_a_run() -> None:
    steps = [_call(0, "a", {}), _call(1, "a", {}), _final(2), _call(3, "a", {})]
    assert is_loop(steps) is False


def test_is_loop_min_repeat_is_configurable_and_validated() -> None:
    steps = [_call(0, "a", {}), _call(1, "a", {})]
    assert is_loop(steps, min_repeat=2) is True
    with pytest.raises(ValueError):
        is_loop(steps, min_repeat=1)


# classify_trajectory ----------------------------------------------------------------


def _trajectory(
    steps: list[Step], *, success: bool, violations: list[str] | None = None
) -> Trajectory:
    failure = FailureCategory.SUCCESS if success else FailureCategory.WRONG_STATE
    return Trajectory(
        run_id="r",
        task_id="t",
        trial=0,
        seed=0,
        agent_id="scripted",
        steps=steps,
        final_state={},
        success=success,
        violations=violations or [],
        failure=failure,
        wall_s=0.0,
    )


def test_classify_trajectory_success() -> None:
    traj = _trajectory([_call(0, "get_compound", {"compound_id": "C07"}), _final(1)], success=True)
    assert classify_trajectory(traj, MAX_TURNS) is FailureCategory.SUCCESS


def test_classify_trajectory_derives_false_success_from_final_step() -> None:
    traj = _trajectory([_call(0, "get_compound", {"compound_id": "C07"}), _final(1)], success=False)
    assert classify_trajectory(traj, MAX_TURNS) is FailureCategory.FALSE_SUCCESS_CLAIM


def test_classify_trajectory_policy_violation_with_failed_success() -> None:
    traj = _trajectory(
        [_call(0, "refund", {"amount": 900}), _final(1)],
        success=False,
        violations=["refund above the 500 limit"],
    )
    assert classify_trajectory(traj, MAX_TURNS) is FailureCategory.POLICY_VIOLATION
