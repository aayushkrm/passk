"""Goal DSL tests: path parsing, operators, edge cases and goal/forbidden semantics."""

from __future__ import annotations

from typing import Any

import pytest

from passk.goals import (
    check_assertion,
    check_assertions,
    check_goal,
    evaluate,
    explain_goal,
    parse_path,
    resolve_path,
    validate_assertion,
)
from passk.schema import Assertion, Task, ToolCall

STATE: dict[str, Any] = {
    "compounds": [
        {"id": "C01", "name": "Acetone", "stock_g": 500, "hazard": "flammable"},
        {"id": "C07", "name": "Glycerol", "stock_g": 465.0, "hazard": "none"},
    ],
    "orders": [
        {"order_id": "O1", "compound_id": "C07", "grams": 25, "status": "placed"},
        {"order_id": "O2", "compound_id": "C07", "grams": 10, "status": "cancelled"},
        {"order_id": "O3", "compound_id": "C01", "grams": 5, "status": "placed"},
    ],
    "approvals": [{"compound_id": "C09", "reason": "qc", "status": "approved"}],
    "budget": {"total": 100.0, "spent": 1.25, "remaining": 98.75},
    "flags": {"ready": True, "count": 0},
    "tags": ["a", "b"],
    "note": "hello world",
}


def _a(path: str, op: str, value: Any = None) -> Assertion:
    return Assertion(path=path, op=op, value=value)  # type: ignore[arg-type]


def _task(goal: list[Assertion], forbidden: list[Assertion] | None = None) -> Task:
    return Task(
        id="unit_goal",
        tier=1,
        instruction="unit",
        initial_state={},
        goal=goal,
        forbidden=forbidden or [],
        reference_solution=[ToolCall(call_id="c", name="get_budget", arguments={})],
    )


@pytest.mark.parametrize(
    "path, expected_names",
    [
        ("budget.remaining", ["budget", "remaining"]),
        ("orders[compound_id=C07].grams", ["orders", "grams"]),
    ],
)
def test_parse_path_splits_segments(path: str, expected_names: list[str]) -> None:
    segments = parse_path(path)
    assert [s.name for s in segments] == expected_names


def test_parse_path_reads_filter_key_and_value() -> None:
    segments = parse_path("orders[compound_id=C07].grams")
    assert segments[0].key == "compound_id"
    assert segments[0].value == "C07"
    assert segments[1].key is None


@pytest.mark.parametrize(
    "path",
    ["", "orders[compound_id]", "orders[=C07]", "orders[compound_id=C07", "a..b", "1abc", "a b"],
)
def test_parse_path_rejects_malformed(path: str) -> None:
    with pytest.raises(ValueError):
        parse_path(path)


@pytest.mark.parametrize(
    "path, expected",
    [
        ("budget.remaining", [98.75]),
        ("orders[compound_id=C07].grams", [25, 10]),
        ("orders[status=placed].order_id", ["O1", "O3"]),
        ("orders.grams", [25, 10, 5]),
        ("orders[order_id=O9].grams", []),
        ("compounds[id=C99].stock_g", []),
        ("budget.nope", []),
        ("note.x", []),
        ("flags.count", [0]),
    ],
)
def test_resolve_path_values(path: str, expected: list[Any]) -> None:
    assert resolve_path(STATE, path) == expected


def test_resolve_path_trailing_list_is_returned_whole() -> None:
    matches = resolve_path(STATE, "orders")
    assert len(matches) == 1
    assert isinstance(matches[0], list)
    assert len(matches[0]) == 3


def test_resolve_path_filter_on_non_list_matches_nothing() -> None:
    assert resolve_path(STATE, "budget[remaining=98]") == []


def test_filter_value_cannot_contain_a_dot() -> None:
    with pytest.raises(ValueError):
        parse_path("budget[remaining=98.75]")


def test_equals_passes_on_exact_value() -> None:
    assert evaluate(_a("budget.remaining", "equals", 98.75), STATE)


def test_equals_uses_numeric_tolerance_only_within_1e6() -> None:
    assert evaluate(_a("budget.remaining", "equals", 98.75 + 1e-9), STATE)
    assert not evaluate(_a("budget.remaining", "equals", 98.7501), STATE)


def test_equals_fails_when_matches_disagree() -> None:
    assert not evaluate(_a("orders[compound_id=C07].grams", "equals", 25), STATE)


def test_equals_passes_when_duplicate_matches_agree() -> None:
    state = {"approvals": [{"status": "approved"}, {"status": "approved"}]}
    assert evaluate(_a("approvals.status", "equals", "approved"), state)


def test_equals_fails_on_missing_path() -> None:
    assert not evaluate(_a("orders[order_id=O9].grams", "equals", 1), STATE)


def test_equals_keeps_bool_distinct_from_number() -> None:
    assert evaluate(_a("flags.ready", "equals", True), STATE)
    assert not evaluate(_a("flags.ready", "equals", 1), STATE)
    assert not evaluate(_a("flags.count", "equals", False), STATE)


def test_equals_compares_strings_exactly() -> None:
    assert evaluate(_a("orders[order_id=O2].status", "equals", "cancelled"), STATE)
    assert not evaluate(_a("orders[order_id=O2].status", "equals", "cancel"), STATE)


def test_contains_on_list_string_and_dict() -> None:
    assert evaluate(_a("tags", "contains", "a"), STATE)
    assert not evaluate(_a("tags", "contains", "z"), STATE)
    assert evaluate(_a("note", "contains", "world"), STATE)
    assert evaluate(_a("budget", "contains", "remaining"), STATE)
    assert not evaluate(_a("budget", "contains", "missing"), STATE)


def test_contains_on_filtered_values_checks_any_match() -> None:
    assert evaluate(_a("orders[status=placed].order_id", "contains", "O3"), STATE)


def test_contains_on_scalar_is_false() -> None:
    assert not evaluate(_a("flags.count", "contains", 0), STATE)


def test_not_contains_is_the_negation_and_true_when_absent() -> None:
    assert evaluate(_a("tags", "not_contains", "z"), STATE)
    assert not evaluate(_a("tags", "not_contains", "a"), STATE)
    assert evaluate(_a("orders[order_id=O9].grams", "not_contains", 1), STATE)


def test_count_eq_counts_matches() -> None:
    assert evaluate(_a("orders[compound_id=C07]", "count_eq", 2), STATE)
    assert not evaluate(_a("orders[compound_id=C07]", "count_eq", 1), STATE)
    assert evaluate(_a("orders[compound_id=C99]", "count_eq", 0), STATE)
    assert evaluate(_a("orders[status=cancelled]", "count_eq", 1), STATE)


def test_absent_holds_only_when_nothing_matches() -> None:
    assert evaluate(_a("orders[compound_id=C99]", "absent"), STATE)
    assert not evaluate(_a("orders[compound_id=C07]", "absent"), STATE)


def test_absent_ignores_value() -> None:
    assert evaluate(_a("orders[compound_id=C99]", "absent", "anything"), STATE)


@pytest.mark.parametrize(
    "assertion",
    [
        _a("orders[compound_id=C07]", "count_eq", "2"),
        _a("orders[compound_id=C07]", "count_eq", -1),
        _a("orders[compound_id=C07]", "count_eq", True),
        _a("tags", "contains", None),
        _a("tags", "not_contains", None),
        _a("orders[compound_id", "equals", 1),
        Assertion.model_construct(path="tags", op="bogus", value=None),
    ],
)
def test_validate_assertion_rejects_malformed(assertion: Assertion) -> None:
    with pytest.raises(ValueError):
        validate_assertion(assertion)


def test_evaluate_raises_on_malformed_assertion() -> None:
    with pytest.raises(ValueError):
        evaluate(_a("tags", "contains", None), STATE)


def test_check_assertion_is_an_alias_of_evaluate() -> None:
    assertion = _a("budget.remaining", "equals", 98.75)
    assert check_assertion(assertion, STATE) is True


def test_check_assertions_reports_only_failures() -> None:
    messages = check_assertions(
        STATE,
        [_a("budget.remaining", "equals", 98.75), _a("orders[compound_id=C99]", "equals", 1)],
    )
    assert len(messages) == 1
    assert messages[0].startswith("unmet: equals orders[compound_id=C99]")


def test_check_assertions_empty_when_all_hold() -> None:
    assert check_assertions(STATE, [_a("tags", "contains", "b")]) == []


def test_check_goal_true_when_goals_hold_and_no_violations() -> None:
    task = _task(
        goal=[_a("orders[order_id=O1].grams", "equals", 25)],
        forbidden=[_a("orders[status=placed]", "count_eq", 3)],
    )
    assert check_goal(task, STATE, []) is True


def test_check_goal_false_when_any_violation_recorded() -> None:
    task = _task(goal=[_a("tags", "contains", "a")])
    assert check_goal(task, STATE, ["order_without_approval: C09 1 g"]) is False


def test_check_goal_false_when_goal_unmet() -> None:
    task = _task(goal=[_a("tags", "contains", "z")])
    assert check_goal(task, STATE, []) is False


def test_check_goal_false_when_forbidden_holds() -> None:
    task = _task(
        goal=[_a("tags", "contains", "a")],
        forbidden=[_a("orders[status=placed]", "count_eq", 2)],
    )
    assert check_goal(task, STATE, []) is False


def test_check_goal_needs_every_goal_assertion() -> None:
    task = _task(goal=[_a("tags", "contains", "a"), _a("tags", "contains", "z")])
    assert check_goal(task, STATE, []) is False


def test_explain_goal_lists_every_reason() -> None:
    task = _task(
        goal=[_a("tags", "contains", "z")],
        forbidden=[_a("orders[status=placed]", "count_eq", 2)],
    )
    reasons = explain_goal(task, STATE, ["policy_x"])
    assert reasons[0] == "violation: policy_x"
    assert any(r.startswith("unmet: contains tags") for r in reasons)
    assert any(r.startswith("forbidden: count_eq orders[status=placed]") for r in reasons)


def test_explain_goal_empty_when_solved() -> None:
    task = _task(goal=[_a("tags", "contains", "a")])
    assert explain_goal(task, STATE, []) == []
