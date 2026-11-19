"""Schema tests: JSON round trips, enum order and protocol shape."""

from __future__ import annotations

import typing
from enum import StrEnum

import pytest
from pydantic import BaseModel, ValidationError

from passk import schema
from passk.schema import (
    Agent,
    AgentAction,
    AgentConfig,
    Assertion,
    Env,
    FailureCategory,
    GateDecision,
    GatePolicy,
    RunConfig,
    RunReport,
    Step,
    Task,
    TaskStats,
    ToolCall,
    ToolResult,
    ToolSpec,
    ToolTransport,
    Trajectory,
)


def _sample_models() -> list[BaseModel]:
    call = ToolCall(call_id="c1", name="get_compound", arguments={"compound_id": "C07"})
    result = ToolResult(call_id="c1", ok=True, content={"name": "Ethanol"})
    action = AgentAction(kind="tool_call", tool_call=call)
    step = Step(
        index=0,
        action=action,
        result=result,
        latency_s=0.25,
        input_tokens=120,
        output_tokens=18,
    )
    assertion = Assertion(path="orders[compound_id=C07].grams", op="equals", value=10)
    task = Task(
        id="t1-001",
        tier=1,
        instruction="Look up compound C07.",
        initial_state={"compounds": {"C07": {"name": "Ethanol"}}},
        goal=[assertion],
        forbidden=[Assertion(path="orders", op="absent")],
        reference_solution=[call],
        tags=["lookup"],
    )
    run_config = RunConfig(
        run_id="demo",
        tasks_glob="src/passk/tasks/labstock/*.yaml",
        agent=AgentConfig(kind="oracle", params={}),
        n_trials=5,
        base_seed=7,
        transport="inprocess",
    )
    traj = Trajectory(
        run_id="demo",
        task_id=task.id,
        trial=0,
        seed=123,
        agent_id="oracle",
        steps=[step],
        final_state={"orders": []},
        success=True,
        violations=[],
        failure=FailureCategory.SUCCESS,
        wall_s=0.5,
    )
    stats = TaskStats(
        task_id=task.id,
        tier=1,
        n=5,
        c=3,
        pass_at_k={1: 0.6, 2: 0.9},
        pass_hat_k={1: 0.6, 2: 0.3},
        failures={"wrong_state": 2},
    )
    report = RunReport(
        run_id="demo",
        config=run_config,
        env_info={"git_sha": None},
        tasks=[stats],
        summary={"pass_hat_k": {1: {"mean": 0.6, "ci_low": 0.4, "ci_high": 0.8}}},
        failure_totals={"wrong_state": 2, "success": 3},
        tokens={"input_median": 120},
        wall_s=12.5,
    )
    return [
        ToolSpec(
            name="get_compound", description="Get one compound.", parameters={"type": "object"}
        ),
        call,
        result,
        action,
        AgentAction(kind="final", message="Done."),
        step,
        assertion,
        task,
        traj,
        AgentConfig(kind="noisy_oracle", params={"p_error": 0.1, "seed": 1}),
        run_config,
        stats,
        report,
        GatePolicy(k=5, max_drop=0.1, min_pass_hat_k=0.5),
        GateDecision(passed=True, reasons=[], delta=-0.02, ci_low=-0.1, ci_high=0.05),
    ]


@pytest.mark.parametrize("model", _sample_models(), ids=lambda m: type(m).__name__)
def test_json_round_trip(model: BaseModel) -> None:
    restored = type(model).model_validate_json(model.model_dump_json())
    assert restored == model


def test_run_report_int_keys_survive_round_trip() -> None:
    report = next(m for m in _sample_models() if isinstance(m, RunReport))
    restored = RunReport.model_validate_json(report.model_dump_json())
    assert restored.tasks[0].pass_hat_k == {1: 0.6, 2: 0.3}
    assert all(isinstance(k, int) for k in restored.summary["pass_hat_k"])


def test_run_report_schema_version_is_fixed() -> None:
    report = next(m for m in _sample_models() if isinstance(m, RunReport))
    assert report.schema_version == "1"
    data = report.model_dump(mode="json")
    data["schema_version"] = "2"
    with pytest.raises(ValidationError):
        RunReport.model_validate(data)


def test_failure_category_is_strenum_in_precedence_order() -> None:
    assert issubclass(FailureCategory, StrEnum)
    assert [c.value for c in FailureCategory] == [
        "policy_violation",
        "schema_error",
        "tool_error_unrecovered",
        "false_success_claim",
        "loop",
        "max_turns",
        "wrong_state",
        "success",
    ]


def test_failure_category_serialises_as_plain_string() -> None:
    traj = next(m for m in _sample_models() if isinstance(m, Trajectory))
    assert '"failure":"success"' in traj.model_dump_json()


def test_task_tier_is_restricted() -> None:
    with pytest.raises(ValidationError):
        Task(
            id="bad",
            tier=4,  # type: ignore[arg-type]
            instruction="x",
            initial_state={},
            goal=[],
            forbidden=[],
            reference_solution=[],
        )


@pytest.mark.parametrize("protocol", [Env, Agent, ToolTransport])
def test_protocols_are_importable_and_structural(protocol: type) -> None:
    assert getattr(protocol, "_is_protocol", False) is True


def test_tool_transport_mirrors_env_methods() -> None:
    env_methods = {"reset", "tools", "call", "state", "violations"}
    assert env_methods <= set(vars(Env))
    assert env_methods <= set(vars(ToolTransport))
    assert env_methods <= set(vars(ToolTransport))


def test_schema_module_exports_expected_names() -> None:
    expected = {
        "ToolSpec",
        "ToolCall",
        "ToolResult",
        "AgentAction",
        "Step",
        "Assertion",
        "Task",
        "FailureCategory",
        "Trajectory",
        "AgentConfig",
        "RunConfig",
        "TaskStats",
        "RunReport",
        "GatePolicy",
        "GateDecision",
        "Env",
        "Agent",
        "ToolTransport",
    }
    assert expected <= set(dir(schema))
    assert typing.get_type_hints(RunReport)["schema_version"] == typing.Literal["1"]
