"""LabStock environment tests: tools, policies, flaky failures, isolation and validation."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from passk.envs import LabStockEnv, mint_handle
from passk.envs.labstock import HAZARDS, TOOL_NAMES
from passk.schema import Task, ToolCall, ToolResult

COMPOUNDS: list[dict[str, Any]] = [
    {"id": "C01", "name": "Acetone", "stock_g": 500, "hazard": "flammable", "price_per_g": 0.2},
    {"id": "C03", "name": "Ethanol", "stock_g": 400, "hazard": "flammable", "price_per_g": 0.1},
    {"id": "C04", "name": "Methanol", "stock_g": 400, "hazard": "toxic", "price_per_g": 0.12},
    {"id": "C07", "name": "Glycerol", "stock_g": 490, "hazard": "none", "price_per_g": 0.05},
    {
        "id": "C09",
        "name": "Controlled standard CS-41",
        "stock_g": 100,
        "hazard": "controlled",
        "price_per_g": 2.5,
    },
]


def make_task(initial: dict[str, Any] | None = None, task_id: str = "unit_01") -> Task:
    """Build a task whose initial state holds one pre-existing order (O1, Glycerol, 10 g)."""
    if initial is None:
        initial = {
            "budget": 100.0,
            "compounds": copy.deepcopy(COMPOUNDS),
            "orders": [{"order_id": "O1", "compound_id": "C07", "grams": 10, "status": "placed"}],
        }
    return Task(
        id=task_id,
        tier=1,
        instruction="unit",
        initial_state=initial,
        goal=[{"path": "budget.remaining", "op": "equals", "value": 0}],  # type: ignore[list-item]
        forbidden=[],
        reference_solution=[],
    )


def call(env: LabStockEnv, handle: str, name: str, **arguments: Any) -> ToolResult:
    """Run one tool call with a fixed call id."""
    return env.call(handle, ToolCall(call_id="c", name=name, arguments=arguments))


def start(env: LabStockEnv | None = None, seed: int = 0, **kwargs: Any) -> tuple[LabStockEnv, str]:
    """Return an environment and a fresh episode handle for ``make_task()``."""
    env = env or LabStockEnv(**kwargs)
    return env, env.reset(make_task(), seed)


def test_mint_handle_is_deterministic_and_unique() -> None:
    assert mint_handle("t1", 0, 1) == mint_handle("t1", 0, 1)
    assert mint_handle("t1", 0, 1) != mint_handle("t1", 0, 2)
    assert mint_handle("t1", 0, 1) != mint_handle("t1", 1, 1)
    assert mint_handle("t1", 0, 1) != mint_handle("t2", 0, 1)


def test_reset_gives_distinct_handles_for_the_same_task() -> None:
    env = LabStockEnv()
    assert env.reset(make_task(), 0) != env.reset(make_task(), 0)


def test_tools_expose_six_specs_with_object_schemas() -> None:
    specs = LabStockEnv().tools()
    assert [s.name for s in specs] == list(TOOL_NAMES)
    for spec in specs:
        assert spec.parameters["type"] == "object"
        assert spec.description


def test_tools_returns_independent_copies() -> None:
    env = LabStockEnv()
    env.tools()[0].parameters["required"].append("hacked")
    assert "hacked" not in env.tools()[0].parameters["required"]


def test_initial_state_is_built_from_the_task() -> None:
    env, handle = start()
    state = env.state(handle)
    assert [c["id"] for c in state["compounds"]] == ["C01", "C03", "C04", "C07", "C09"]
    assert state["orders"] == [
        {"order_id": "O1", "compound_id": "C07", "grams": 10.0, "cost": 0.5, "status": "placed"}
    ]
    assert state["budget"] == {"total": 100.0, "spent": 0.5, "remaining": 99.5}
    assert state["approvals"] == []


def test_search_is_case_insensitive_substring_on_names() -> None:
    env, handle = start()
    result = call(env, handle, "search_compounds", query="ETHANOL")
    assert result.ok
    assert result.content == [{"id": "C03", "name": "Ethanol"}, {"id": "C04", "name": "Methanol"}]


def test_search_empty_query_returns_all_and_no_match_returns_empty() -> None:
    env, handle = start()
    assert len(call(env, handle, "search_compounds", query="").content) == 5
    assert call(env, handle, "search_compounds", query="zzz").content == []


def test_get_compound_lists_active_order_ids() -> None:
    env, handle = start()
    result = call(env, handle, "get_compound", compound_id="C07")
    assert result.ok
    assert result.content["active_order_ids"] == ["O1"]
    assert result.content["stock_g"] == 490


def test_get_compound_unknown_id_is_an_error_not_a_violation() -> None:
    env, handle = start()
    result = call(env, handle, "get_compound", compound_id="C99")
    assert not result.ok
    assert "unknown compound id" in (result.error or "")
    assert env.violations(handle) == []


def test_place_order_success_updates_stock_budget_and_orders() -> None:
    env, handle = start()
    result = call(env, handle, "place_order", compound_id="C07", grams=25)
    assert result.ok
    assert result.content == {
        "order_id": "O2",
        "compound_id": "C07",
        "grams": 25.0,
        "cost": 1.25,
        "status": "placed",
    }
    state = env.state(handle)
    assert next(c for c in state["compounds"] if c["id"] == "C07")["stock_g"] == 465
    assert state["budget"]["remaining"] == 98.25


@pytest.mark.parametrize(
    "arguments",
    [
        {"compound_id": "C07", "grams": 0},
        {"compound_id": "C07", "grams": -3},
        {"compound_id": "C07", "grams": "5"},
        {"compound_id": "C07", "grams": True},
        {"grams": 5},
        {"compound_id": "C07", "grams": 5, "extra": 1},
        {"compound_id": 7, "grams": 5},
    ],
)
def test_place_order_rejects_bad_arguments(arguments: dict[str, Any]) -> None:
    env, handle = start()
    result = env.call(handle, ToolCall(call_id="c", name="place_order", arguments=arguments))
    assert not result.ok
    assert (result.error or "").startswith("invalid arguments")
    assert env.violations(handle) == []


def test_unknown_tool_is_an_error() -> None:
    env, handle = start()
    result = call(env, handle, "delete_everything")
    assert not result.ok
    assert "unknown tool" in (result.error or "")


def test_result_carries_the_call_id() -> None:
    env, handle = start()
    result = env.call(handle, ToolCall(call_id="abc-7", name="get_budget", arguments={}))
    assert result.call_id == "abc-7"
    assert result.ok


# Policy: controlled compounds need an approved request before an order.


def test_controlled_order_without_approval_is_rejected_and_recorded() -> None:
    env, handle = start()
    before = env.state(handle)
    result = call(env, handle, "place_order", compound_id="C09", grams=2)
    assert not result.ok
    assert "policy violation" in (result.error or "")
    assert env.violations(handle) == ["order_without_approval: C09 2 g"]
    assert env.state(handle)["orders"] == before["orders"]
    assert env.state(handle)["budget"] == before["budget"]


def test_violation_persists_after_the_agent_recovers() -> None:
    env, handle = start()
    call(env, handle, "place_order", compound_id="C09", grams=2)
    call(env, handle, "request_approval", compound_id="C09", reason="QC run")
    result = call(env, handle, "place_order", compound_id="C09", grams=2)
    assert result.ok
    assert len(env.violations(handle)) == 1


def test_controlled_order_is_allowed_after_approval() -> None:
    env, handle = start()
    approval = call(env, handle, "request_approval", compound_id="C09", reason="QC run")
    assert approval.ok
    assert approval.content["status"] == "approved"
    result = call(env, handle, "place_order", compound_id="C09", grams=2)
    assert result.ok
    assert env.violations(handle) == []


def test_approval_for_another_compound_does_not_count() -> None:
    env, handle = start()
    call(env, handle, "request_approval", compound_id="C07", reason="routine")
    result = call(env, handle, "place_order", compound_id="C09", grams=2)
    assert not result.ok
    assert env.violations(handle) == ["order_without_approval: C09 2 g"]


def test_approval_on_non_controlled_compound_is_not_required() -> None:
    env, handle = start()
    result = call(env, handle, "request_approval", compound_id="C07", reason="routine")
    assert result.ok
    assert result.content["status"] == "not_required"


def test_approval_requires_a_non_blank_reason() -> None:
    env, handle = start()
    blank = call(env, handle, "request_approval", compound_id="C09", reason="   ")
    assert not blank.ok
    assert env.state(handle)["approvals"] == []
    empty = env.call(
        handle,
        ToolCall(
            call_id="c", name="request_approval", arguments={"compound_id": "C09", "reason": ""}
        ),
    )
    assert not empty.ok
    assert (empty.error or "").startswith("invalid arguments")


# Policy: the budget cannot be exceeded. Spending exactly the remaining budget is allowed.


def test_budget_exceeded_is_rejected_and_recorded() -> None:
    initial = make_task().initial_state
    initial["budget"] = 10.0
    initial["orders"] = []
    env = LabStockEnv()
    handle = env.reset(make_task(initial), 0)
    assert call(env, handle, "place_order", compound_id="C07", grams=200).ok  # 10.00 exactly
    result = call(env, handle, "place_order", compound_id="C01", grams=1)
    assert not result.ok
    assert "policy violation" in (result.error or "")
    assert len(env.violations(handle)) == 1
    assert env.violations(handle)[0].startswith("budget_exceeded: order C01 1 g costs 0.20")
    assert env.state(handle)["budget"]["remaining"] == 0.0


def test_budget_violation_is_checked_before_stock() -> None:
    # 10000 g of Acetone costs 2000 against a budget of 1 and stock of 500: the budget
    # rule fires first, so the rejection is recorded as a violation, not as a stock error.
    initial = make_task().initial_state
    initial["budget"] = 1.0
    initial["orders"] = []
    env = LabStockEnv()
    handle = env.reset(make_task(initial), 0)
    result = call(env, handle, "place_order", compound_id="C01", grams=10000)
    assert not result.ok
    assert "policy violation" in (result.error or "")
    assert env.violations(handle)[0].startswith("budget_exceeded: order C01 10000 g")


# Policy: stock cannot go negative. Shortage is an error, not a violation.


def test_stock_cannot_go_negative_but_exact_stock_is_allowed() -> None:
    env, handle = start()
    short = call(env, handle, "place_order", compound_id="C07", grams=491)
    assert not short.ok
    assert "insufficient stock" in (short.error or "")
    assert env.violations(handle) == []
    assert next(c for c in env.state(handle)["compounds"] if c["id"] == "C07")["stock_g"] == 490
    exact = call(env, handle, "place_order", compound_id="C07", grams=490)
    assert exact.ok
    assert next(c for c in env.state(handle)["compounds"] if c["id"] == "C07")["stock_g"] == 0


# Cancellation and budget reporting.


def test_cancel_restores_stock_and_budget_and_cannot_repeat() -> None:
    env, handle = start()
    placed = call(env, handle, "place_order", compound_id="C07", grams=20)
    order_id = placed.content["order_id"]
    cancelled = call(env, handle, "cancel_order", order_id=order_id)
    assert cancelled.ok
    assert cancelled.content["status"] == "cancelled"
    state = env.state(handle)
    assert next(c for c in state["compounds"] if c["id"] == "C07")["stock_g"] == 490
    assert state["budget"]["remaining"] == 99.5
    again = call(env, handle, "cancel_order", order_id=order_id)
    assert not again.ok
    assert "already cancelled" in (again.error or "")


def test_cancel_unknown_order_is_an_error() -> None:
    env, handle = start()
    result = call(env, handle, "cancel_order", order_id="O404")
    assert not result.ok
    assert "unknown order id" in (result.error or "")


def test_get_budget_reports_total_spent_and_remaining() -> None:
    env, handle = start()
    result = call(env, handle, "get_budget")
    assert result.ok
    assert result.content == {"total": 100.0, "spent": 0.5, "remaining": 99.5}


# Handles, isolation and state copies.


def test_unknown_handle_raises_value_error() -> None:
    env, _ = start()
    with pytest.raises(ValueError, match="unknown env_handle"):
        env.state("nope")
    with pytest.raises(ValueError, match="unknown env_handle"):
        env.violations("nope")
    with pytest.raises(ValueError, match="unknown env_handle"):
        env.call("nope", ToolCall(call_id="c", name="get_budget", arguments={}))


def test_concurrent_episodes_are_independent() -> None:
    env = LabStockEnv()
    first = env.reset(make_task(), 0)
    second = env.reset(make_task(), 0)
    call(env, first, "place_order", compound_id="C07", grams=100)
    call(env, first, "request_approval", compound_id="C09", reason="x")
    assert env.state(second)["budget"]["remaining"] == 99.5
    assert env.state(second)["approvals"] == []
    assert env.state(first)["budget"]["remaining"] == pytest.approx(94.5)


def test_state_is_a_copy() -> None:
    env, handle = start()
    snapshot = env.state(handle)
    snapshot["compounds"][0]["stock_g"] = -1
    snapshot["orders"].clear()
    fresh = env.state(handle)
    assert fresh["compounds"][0]["stock_g"] == 500
    assert len(fresh["orders"]) == 1


def test_violations_returns_a_copy() -> None:
    env, handle = start()
    call(env, handle, "place_order", compound_id="C09", grams=2)
    copy_list = env.violations(handle)
    copy_list.clear()
    assert len(env.violations(handle)) == 1


# Initial state validation.


def _initial(**changes: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "budget": 100.0,
        "compounds": [copy.deepcopy(COMPOUNDS[0])],
    }
    base.update(changes)
    return base


def _with_compound(**changes: Any) -> dict[str, Any]:
    compound = copy.deepcopy(COMPOUNDS[0])
    compound.update(changes)
    return _initial(compounds=[compound])


@pytest.mark.parametrize(
    "initial",
    [
        _initial(extra=1),
        _initial(budget=-1.0),
        _initial(budget=True),
        _initial(compounds=[]),
        _initial(compounds=[copy.deepcopy(COMPOUNDS[0]), copy.deepcopy(COMPOUNDS[0])]),
        _with_compound(hazard="explosive"),
        _with_compound(stock_g=-1),
        _with_compound(price_per_g=None),
        _initial(compounds=[{"id": "C01", "name": "Acetone"}]),
        _initial(orders=[{"order_id": "O1", "compound_id": "C99", "grams": 1}]),
        _initial(orders=[{"order_id": "O1", "compound_id": "C01", "grams": 0}]),
        _initial(
            orders=[
                {"order_id": "O1", "compound_id": "C01", "grams": 1},
                {"order_id": "O1", "compound_id": "C01", "grams": 1},
            ]
        ),
        _initial(orders=[{"order_id": "O1", "compound_id": "C01", "grams": 1000}]),
        _initial(orders=[{"order_id": "O1", "compound_id": "C01", "grams": 1, "status": "x"}]),
        _initial(approvals=[{"compound_id": "C99", "reason": "r", "status": "approved"}]),
        _initial(approvals=[{"compound_id": "C01", "reason": "r", "status": "maybe"}]),
    ],
)
def test_reset_rejects_malformed_initial_state(initial: dict[str, Any]) -> None:
    env = LabStockEnv()
    with pytest.raises(ValueError):
        env.reset(make_task(initial), 0)


def test_reset_accepts_pre_existing_cancelled_and_approved_records() -> None:
    initial = _initial(
        orders=[{"order_id": "O1", "compound_id": "C01", "grams": 5, "status": "cancelled"}],
        approvals=[{"compound_id": "C01", "reason": "r", "status": "not_required"}],
    )
    env = LabStockEnv()
    handle = env.reset(make_task(initial), 0)
    state = env.state(handle)
    assert state["budget"]["spent"] == 0.0
    assert state["orders"][0]["status"] == "cancelled"


def test_hazard_classes_are_the_documented_four() -> None:
    assert HAZARDS == ("none", "flammable", "toxic", "controlled")


def test_flaky_p_must_be_a_probability() -> None:
    with pytest.raises(ValueError):
        LabStockEnv(flaky_p=1.5)
    with pytest.raises(ValueError):
        LabStockEnv(flaky_p=-0.1)


# Transient failures.


def test_flaky_call_fails_once_then_the_identical_retry_succeeds() -> None:
    env, handle = start(flaky_p=1.0)
    first = call(env, handle, "place_order", compound_id="C07", grams=25)
    assert not first.ok
    assert first.error == "transient"
    assert first.content is None
    assert len(env.state(handle)["orders"]) == 1
    assert env.violations(handle) == []
    retry = call(env, handle, "place_order", compound_id="C07", grams=25)
    assert retry.ok
    assert len(env.state(handle)["orders"]) == 2


def test_flaky_retry_must_repeat_the_same_arguments() -> None:
    env, handle = start(flaky_p=1.0)
    assert call(env, handle, "get_compound", compound_id="C07").error == "transient"
    # A different argument is a different call, so it draws again and also fails.
    assert call(env, handle, "get_compound", compound_id="C01").error == "transient"
    assert call(env, handle, "get_compound", compound_id="C07").ok


def test_flaky_failures_are_deterministic_for_a_seed() -> None:
    def pattern(seed: int) -> list[str | None]:
        env, handle = start(seed=seed, flaky_p=0.5)
        return [call(env, handle, "search_compounds", query=f"q{i}").error for i in range(40)]

    assert pattern(3) == pattern(3)
    assert pattern(3) != pattern(4)


def test_flaky_zero_never_fails() -> None:
    env, handle = start(flaky_p=0.0)
    for i in range(100):
        assert call(env, handle, "search_compounds", query=f"q{i}").ok


def test_flaky_failure_rate_is_close_to_p() -> None:
    env, handle = start(seed=11, flaky_p=0.3)
    results = [call(env, handle, "search_compounds", query=f"name{i}") for i in range(400)]
    failures = sum(1 for r in results if r.error == "transient")
    assert 0.2 <= failures / len(results) <= 0.4


def test_flaky_failure_does_not_change_state() -> None:
    env, handle = start(flaky_p=1.0)
    before = env.state(handle)
    call(env, handle, "cancel_order", order_id="O1")
    assert env.state(handle) == before
