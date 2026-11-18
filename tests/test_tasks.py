"""Task YAML tests: every shipped task validates, reference solutions solve it, loader errors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from passk.envs import LabStockEnv, load_tasks, tasks_dir
from passk.envs.labstock import TOOL_NAMES
from passk.goals import check_goal, explain_goal
from passk.schema import Task, ToolCall

GLOB = str(tasks_dir("labstock") / "*.yaml")
TASKS: list[Task] = load_tasks(GLOB)
EXPECTED_IDS = [f"t{tier}_{n:02d}" for tier in (1, 2, 3) for n in range(1, 9)]


def test_labstock_ships_24_tasks_in_three_tiers() -> None:
    assert [t.id for t in TASKS] == EXPECTED_IDS
    assert sorted(t.tier for t in TASKS) == [1] * 8 + [2] * 8 + [3] * 8


def test_load_tasks_sorts_by_id(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "many.yaml", [_doc("t9_b"), _doc("t1_a")])
    assert [t.id for t in load_tasks(str(tmp_path / "*.yaml"))] == ["t1_a", "t9_b"]


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_every_task_validates_against_schema(task: Task) -> None:
    assert isinstance(task, Task)
    assert task.instruction.strip()
    assert 8 <= len(task.initial_state["compounds"]) <= 12
    assert task.goal, "goal must not be empty"


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_reference_solution_is_well_formed(task: Task) -> None:
    call_ids = [c.call_id for c in task.reference_solution]
    assert len(set(call_ids)) == len(call_ids)
    assert all(c.name in TOOL_NAMES for c in task.reference_solution)
    # The reference, its final action and the transient-retry budget all fit in max_turns.
    assert len(task.reference_solution) + 1 <= task.max_turns


@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.id)
def test_reference_solution_solves_the_task(task: Task) -> None:
    env = LabStockEnv()
    handle = env.reset(task, seed=0)
    for reference in task.reference_solution:
        result = env.call(
            handle,
            ToolCall(call_id=reference.call_id, name=reference.name, arguments=reference.arguments),
        )
        assert result.ok, f"{reference.name} failed: {result.error}"
    state = env.state(handle)
    violations = env.violations(handle)
    assert violations == []
    assert check_goal(task, state, violations), explain_goal(task, state, violations)


@pytest.mark.parametrize("task", [t for t in TASKS if t.tier == 1], ids=lambda t: t.id)
def test_tier1_is_a_single_call(task: Task) -> None:
    assert len(task.reference_solution) == 1


@pytest.mark.parametrize("task", [t for t in TASKS if t.tier == 2], ids=lambda t: t.id)
def test_tier2_is_a_lookup_then_action(task: Task) -> None:
    assert 2 <= len(task.reference_solution) <= 3
    assert task.reference_solution[0].name in {"search_compounds", "get_compound"}


@pytest.mark.parametrize("task", [t for t in TASKS if t.tier == 3], ids=lambda t: t.id)
def test_tier3_is_policy_constrained(task: Task) -> None:
    assert {"policy", "distractor", "stock"} & set(task.tags)
    assert len(task.reference_solution) >= 2


def test_controlled_orders_request_approval_first_in_reference_solutions() -> None:
    for task in TASKS:
        seen_approval: set[str] = set()
        for reference in task.reference_solution:
            if reference.name == "request_approval":
                seen_approval.add(str(reference.arguments["compound_id"]))
            if reference.name == "place_order":
                compound_id = str(reference.arguments["compound_id"])
                controlled = any(
                    c["id"] == compound_id and c["hazard"] == "controlled"
                    for c in task.initial_state["compounds"]
                )
                if controlled:
                    assert compound_id in seen_approval, task.id


def test_instructions_are_unique() -> None:
    instructions = [t.instruction for t in TASKS]
    assert len(set(instructions)) == len(instructions)


def test_distractor_tasks_contain_near_duplicate_names() -> None:
    by_id = {t.id: t for t in TASKS}
    names = {c["name"] for c in by_id["t3_05"].initial_state["compounds"]}
    assert {"Ethanol", "Methanol"} <= names
    names = {c["name"] for c in by_id["t3_06"].initial_state["compounds"]}
    assert {"Acetone", "Acetonitrile"} <= names


# Loader error messages.


def _doc(task_id: str, **changes: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "id": task_id,
        "tier": 1,
        "instruction": "Place an order for 5 g of compound C07.",
        "initial_state": {
            "budget": 100.0,
            "compounds": [
                {
                    "id": "C07",
                    "name": "Glycerol",
                    "stock_g": 500,
                    "hazard": "none",
                    "price_per_g": 0.05,
                }
            ],
        },
        "goal": [{"path": "orders[compound_id=C07].grams", "op": "equals", "value": 5}],
        "forbidden": [],
        "reference_solution": [
            {
                "call_id": "r1",
                "name": "place_order",
                "arguments": {"compound_id": "C07", "grams": 5},
            }
        ],
    }
    doc.update(changes)
    return doc


def _write_yaml(path: Path, data: Any) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_load_tasks_no_match_is_a_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no task files match"):
        load_tasks(str(tmp_path / "*.yaml"))


def test_load_tasks_invalid_yaml_is_a_value_error(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text("id: [unclosed", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_tasks(str(tmp_path / "*.yaml"))


def test_load_tasks_scalar_document_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "scalar.yaml").write_text("just a string\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top level must be"):
        load_tasks(str(tmp_path / "*.yaml"))


def test_load_tasks_list_entry_must_be_a_mapping(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "list.yaml", [3])
    with pytest.raises(ValueError, match="each task must be a mapping"):
        load_tasks(str(tmp_path / "*.yaml"))


def test_load_tasks_schema_failure_names_the_task(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "t.yaml", _doc("t1_x", tier=4))
    with pytest.raises(ValueError, match="t1_x.*does not validate"):
        load_tasks(str(tmp_path / "*.yaml"))


def test_load_tasks_rejects_malformed_assertion_path(tmp_path: Path) -> None:
    doc = _doc("t1_x", goal=[{"path": "orders[compound_id", "op": "equals", "value": 5}])
    _write_yaml(tmp_path / "t.yaml", doc)
    with pytest.raises(ValueError, match="invalid path segment"):
        load_tasks(str(tmp_path / "*.yaml"))


def test_load_tasks_rejects_count_eq_without_int(tmp_path: Path) -> None:
    doc = _doc("t1_x", goal=[{"path": "orders", "op": "count_eq", "value": "two"}])
    _write_yaml(tmp_path / "t.yaml", doc)
    with pytest.raises(ValueError, match="count_eq"):
        load_tasks(str(tmp_path / "*.yaml"))


def test_load_tasks_rejects_empty_goal(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "t.yaml", _doc("t1_x", goal=[]))
    with pytest.raises(ValueError, match="empty goal"):
        load_tasks(str(tmp_path / "*.yaml"))


def test_load_tasks_rejects_duplicate_ids_across_files(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "a.yaml", _doc("t1_x"))
    _write_yaml(tmp_path / "b.yaml", _doc("t1_x"))
    with pytest.raises(ValueError, match="duplicate task id 't1_x'"):
        load_tasks(str(tmp_path / "*.yaml"))


def test_load_tasks_rejects_duplicate_reference_call_ids(tmp_path: Path) -> None:
    reference = [
        {"call_id": "r1", "name": "get_budget", "arguments": {}},
        {"call_id": "r1", "name": "get_budget", "arguments": {}},
    ]
    _write_yaml(tmp_path / "t.yaml", _doc("t1_x", reference_solution=reference))
    with pytest.raises(ValueError, match="duplicate reference call_ids"):
        load_tasks(str(tmp_path / "*.yaml"))


def test_load_tasks_accepts_a_list_document(tmp_path: Path) -> None:
    _write_yaml(tmp_path / "many.yaml", [_doc("t1_a"), _doc("t1_b")])
    assert [t.id for t in load_tasks(str(tmp_path / "*.yaml"))] == ["t1_a", "t1_b"]


def test_load_tasks_ignores_directories_matched_by_the_glob(tmp_path: Path) -> None:
    (tmp_path / "sub.yaml").mkdir()
    _write_yaml(tmp_path / "t.yaml", _doc("t1_x"))
    assert [t.id for t in load_tasks(str(tmp_path / "*.yaml"))] == ["t1_x"]
