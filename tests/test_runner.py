"""Runner tests with a deterministic FakeAgent and FakeTransport. No model, no network."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from passk.agents import ModelCallInfo
from passk.runner import RunDeps, episode_seed, run
from passk.schema import (
    AgentAction,
    AgentConfig,
    FailureCategory,
    RunConfig,
    RunReport,
    Step,
    Task,
    ToolCall,
    ToolResult,
    ToolSpec,
    Trajectory,
)

REPO = Path(__file__).resolve().parents[1]

TOOLS = [
    ToolSpec(
        name="search_compounds",
        description="Search compounds.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    ),
    ToolSpec(
        name="place_order",
        description="Order a compound.",
        parameters={"type": "object", "properties": {"compound_id": {"type": "string"}}},
    ),
]


def make_task(task_id: str, tier: int = 1, max_turns: int = 4) -> Task:
    """Return a task with the given id."""
    return Task(
        id=task_id,
        tier=tier,  # type: ignore[arg-type]
        instruction=f"Instruction for {task_id}.",
        initial_state={},
        goal=[],
        forbidden=[],
        reference_solution=[],
        max_turns=max_turns,
    )


class FakeTransport:
    """Deterministic ToolTransport. Every call succeeds and echoes its arguments."""

    def __init__(self) -> None:
        self.resets: list[tuple[str, int]] = []
        self.calls: list[ToolCall] = []
        self.closed = False
        self._handle = ""

    def reset(self, task: Task, seed: int) -> str:
        self.resets.append((task.id, seed))
        self._handle = f"h-{task.id}-{seed}"
        return self._handle

    def tools(self) -> list[ToolSpec]:
        return list(TOOLS)

    def call(self, handle: str, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return ToolResult(call_id=call.call_id, ok=True, content={"echo": call.arguments})

    def state(self, handle: str) -> dict[str, Any]:
        return {"handle": handle, "calls": len(self.calls)}

    def violations(self, handle: str) -> list[str]:
        return []

    def close(self) -> None:
        self.closed = True


class FakeAgent:
    """Deterministic agent driven by ``policy(task, history) -> AgentAction``."""

    model_id = "fake-model"
    agent_id = "fake:v1"
    provider_name = "fake"

    def __init__(self, policy: Callable[[Task, list[Step]], AgentAction]) -> None:
        self.policy = policy
        self.begun: list[int] = []
        self.last_call: ModelCallInfo | None = None

    def begin_episode(self, seed: int) -> None:
        self.begun.append(seed)

    def act(self, task: Task, history: list[Step], tools: list[ToolSpec]) -> AgentAction:
        self.last_call = ModelCallInfo(
            input_tokens=100 + len(history), output_tokens=10, latency_s=0.0
        )
        return self.policy(task, history)


def search_then_final(task: Task, history: list[Step]) -> AgentAction:
    """Make one tool call, then finish."""
    if not history:
        return AgentAction(
            kind="tool_call",
            tool_call=ToolCall(call_id="c0", name="search_compounds", arguments={"q": task.id}),
        )
    return AgentAction(kind="final", message="done")


def never_final(task: Task, history: list[Step]) -> AgentAction:
    """Keep calling the same tool forever."""
    return AgentAction(
        kind="tool_call",
        tool_call=ToolCall(call_id=f"c{len(history)}", name="search_compounds", arguments={}),
    )


def goal_on_calls(task: Task, state: dict[str, Any]) -> bool:
    """Goal holds when the episode made at least one call."""
    return state["calls"] >= 1


def classify(
    steps: list[Step],
    *,
    violations: list[str],
    goal_met: bool,
    claimed_done: bool,
    max_turns: int,
) -> FailureCategory:
    """Minimal classifier for runner tests (the real taxonomy has its own tests).

    No final action means the budget ran out. A final action with the goal met is success.
    A final action with the goal unmet is a false success claim.
    """
    if not claimed_done:
        return FailureCategory.MAX_TURNS
    if goal_met and not violations:
        return FailureCategory.SUCCESS
    return FailureCategory.FALSE_SUCCESS_CLAIM


def make_config(run_id: str, n_trials: int = 3, base_seed: int = 42) -> RunConfig:
    """Return a run config for the fake agent (the agent itself is injected)."""
    return RunConfig(
        run_id=run_id,
        tasks_glob="unused",
        agent=AgentConfig(kind="oracle"),
        n_trials=n_trials,
        base_seed=base_seed,
        transport="inprocess",
        k_values=[1, 2, 3],
    )


def make_deps(agent: FakeAgent, transport: FakeTransport, tasks: list[Task]) -> RunDeps:
    """Wire the fakes into RunDeps."""
    return RunDeps(
        tasks=tasks,
        make_agent=lambda _cfg: agent,
        make_transport=lambda _cfg: transport,  # type: ignore[arg-type,return-value]
        check_goal=lambda task, state, violations: goal_on_calls(task, state) and not violations,
        classify=classify,
    )


def test_episode_seed_is_stable_and_in_range() -> None:
    assert episode_seed(42, "t01", 0) == episode_seed(42, "t01", 0)
    assert episode_seed(42, "t01", 0) != episode_seed(42, "t01", 1)
    assert episode_seed(42, "t01", 0) != episode_seed(42, "t02", 0)
    assert episode_seed(42, "t01", 0) != episode_seed(43, "t01", 0)
    for trial in range(50):
        assert 0 <= episode_seed(7, "t09", trial) < 2**31 - 1


def test_episode_seed_does_not_depend_on_python_hash_randomisation() -> None:
    code = "from passk.runner import episode_seed; print(episode_seed(42, 't01', 3))"
    outputs = set()
    for hash_seed in ("1", "2"):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO,
            env={
                "PYTHONHASHSEED": hash_seed,
                "PATH": "/usr/bin:/bin",
                "PYTHONPATH": str(REPO / "src"),
            },
        )
        outputs.add(proc.stdout.strip())
    assert outputs == {str(episode_seed(42, "t01", 3))}


def test_run_writes_all_four_files_and_report_round_trips(tmp_path: Path) -> None:
    agent = FakeAgent(search_then_final)
    transport = FakeTransport()
    tasks = [make_task("t02"), make_task("t01")]
    report = run(make_config("r1", n_trials=3), tmp_path, deps=make_deps(agent, transport, tasks))

    run_dir = tmp_path / "r1"
    for name in ("trajectories.jsonl", "spans.jsonl", "report.json", "env.json"):
        assert (run_dir / name).is_file(), name
    assert not list(run_dir.glob("*.tmp"))

    reloaded = RunReport.model_validate_json((run_dir / "report.json").read_text())
    assert reloaded == report
    assert [t.task_id for t in report.tasks] == ["t01", "t02"]  # sorted by id
    assert all(t.n == 3 for t in report.tasks)

    trajectories = [
        Trajectory.model_validate_json(line)
        for line in (run_dir / "trajectories.jsonl").read_text().splitlines()
    ]
    assert len(trajectories) == 6
    assert [(t.task_id, t.trial) for t in trajectories] == [
        ("t01", 0),
        ("t01", 1),
        ("t01", 2),
        ("t02", 0),
        ("t02", 1),
        ("t02", 2),
    ]


def test_seeds_are_stable_across_runs_and_match_episode_seed(tmp_path: Path) -> None:
    tasks = [make_task("t01")]
    run(
        make_config("a", n_trials=3),
        tmp_path,
        deps=make_deps(FakeAgent(search_then_final), FakeTransport(), tasks),
    )
    run(
        make_config("b", n_trials=3),
        tmp_path,
        deps=make_deps(FakeAgent(search_then_final), FakeTransport(), tasks),
    )
    seeds_a = [
        json.loads(x)["seed"]
        for x in (tmp_path / "a" / "trajectories.jsonl").read_text().splitlines()
    ]
    seeds_b = [
        json.loads(x)["seed"]
        for x in (tmp_path / "b" / "trajectories.jsonl").read_text().splitlines()
    ]
    expected = [episode_seed(42, "t01", trial) for trial in range(3)]
    assert seeds_a == seeds_b == expected
    assert len(set(seeds_a)) == 3


def test_agent_receives_episode_seed_before_each_episode(tmp_path: Path) -> None:
    agent = FakeAgent(search_then_final)
    transport = FakeTransport()
    run(
        make_config("seeded", n_trials=2),
        tmp_path,
        deps=make_deps(agent, transport, [make_task("t01")]),
    )
    assert agent.begun == [episode_seed(42, "t01", 0), episode_seed(42, "t01", 1)]
    assert [seed for _, seed in transport.resets] == agent.begun


def test_max_turns_is_enforced(tmp_path: Path) -> None:
    agent = FakeAgent(never_final)
    transport = FakeTransport()
    run(
        make_config("mt", n_trials=1),
        tmp_path,
        deps=make_deps(agent, transport, [make_task("t01", max_turns=5)]),
    )
    traj = Trajectory.model_validate_json((tmp_path / "mt" / "trajectories.jsonl").read_text())
    assert len(traj.steps) == 5
    assert traj.failure == FailureCategory.MAX_TURNS
    assert traj.success is False
    assert len(transport.calls) == 5


def test_final_action_ends_episode_and_step_records_tokens(tmp_path: Path) -> None:
    agent = FakeAgent(search_then_final)
    run(
        make_config("tok", n_trials=1),
        tmp_path,
        deps=make_deps(agent, FakeTransport(), [make_task("t01")]),
    )
    traj = Trajectory.model_validate_json((tmp_path / "tok" / "trajectories.jsonl").read_text())
    assert [s.action.kind for s in traj.steps] == ["tool_call", "final"]
    assert traj.steps[0].input_tokens == 100
    assert traj.steps[1].input_tokens == 101
    assert traj.steps[0].output_tokens == 10
    assert traj.success is True
    assert traj.agent_id == "fake:v1"
    assert traj.steps[0].action.tool_call is not None
    assert traj.steps[0].action.tool_call.call_id == "c0"


def test_schema_error_is_recorded_and_not_sent_to_environment(tmp_path: Path) -> None:
    def bad_then_final(task: Task, history: list[Step]) -> AgentAction:
        if not history:
            return AgentAction(
                kind="tool_call",
                tool_call=ToolCall(
                    call_id="c0",
                    name="__schema_error__",
                    arguments={"raw": "nope", "reason": "output is not valid JSON"},
                ),
            )
        return AgentAction(kind="final", message="recovered")

    transport = FakeTransport()
    run(
        make_config("se", n_trials=1),
        tmp_path,
        deps=make_deps(FakeAgent(bad_then_final), transport, [make_task("t01")]),
    )
    assert transport.calls == []
    traj = Trajectory.model_validate_json((tmp_path / "se" / "trajectories.jsonl").read_text())
    first = traj.steps[0].result
    assert first is not None
    assert first.ok is False
    assert first.error == "schema_error: output is not valid JSON"
    assert traj.steps[1].action.kind == "final"


def test_failure_categories_and_totals(tmp_path: Path) -> None:
    def every_other(task: Task, history: list[Step]) -> AgentAction:
        if task.id == "t01":
            return search_then_final(task, history)
        return never_final(task, history)

    report = run(
        make_config("tot", n_trials=2),
        tmp_path,
        deps=make_deps(
            FakeAgent(every_other),
            FakeTransport(),
            [make_task("t01"), make_task("t02", max_turns=2)],
        ),
    )
    assert report.failure_totals == {"max_turns": 2, "success": 2}
    by_task = {t.task_id: t for t in report.tasks}
    assert by_task["t01"].c == 2
    assert by_task["t02"].c == 0
    assert by_task["t01"].failures == {"success": 2}
    assert by_task["t02"].failures == {"max_turns": 2}


def test_pass_at_and_hat_k_are_reported_for_valid_k_only(tmp_path: Path) -> None:
    report = run(
        make_config("pk", n_trials=2),
        tmp_path,
        deps=make_deps(FakeAgent(search_then_final), FakeTransport(), [make_task("t01")]),
    )
    stat = report.tasks[0]
    assert set(stat.pass_at_k) == {1, 2}  # k=3 skipped because n=2
    assert stat.pass_hat_k[2] == pytest.approx(1.0)
    assert set(report.summary["pass_hat_k"]) == {1, 2}
    assert report.summary["pass_hat_k"][1]["mean"] == pytest.approx(1.0)


def test_token_totals_in_report(tmp_path: Path) -> None:
    report = run(
        make_config("tk", n_trials=1),
        tmp_path,
        deps=make_deps(FakeAgent(search_then_final), FakeTransport(), [make_task("t01")]),
    )
    assert report.tokens["model_calls"] == 2
    assert report.tokens["input_total"] == 100 + 101
    assert report.tokens["output_total"] == 20
    assert report.tokens["input_median_per_episode"] == pytest.approx(201.0)


def test_spans_file_is_valid_and_uses_semconv_names(tmp_path: Path) -> None:
    run(
        make_config("sp", n_trials=1),
        tmp_path,
        deps=make_deps(FakeAgent(search_then_final), FakeTransport(), [make_task("t01")]),
    )
    spans = [
        json.loads(line) for line in (tmp_path / "sp" / "spans.jsonl").read_text().splitlines()
    ]
    names = [s["name"] for s in spans]
    assert names == [
        "invoke_agent passk",
        "chat fake-model",
        "execute_tool search_compounds",
        "chat fake-model",
    ]
    root = spans[0]
    assert root["parent_span_id"] is None
    assert all(s["parent_span_id"] == root["span_id"] for s in spans[1:])
    assert all(s["trace_id"] == root["trace_id"] for s in spans)
    assert root["attributes"]["gen_ai.operation.name"] == "invoke_agent"
    assert spans[1]["attributes"]["gen_ai.request.model"] == "fake-model"
    assert spans[2]["attributes"]["gen_ai.tool.name"] == "search_compounds"
    assert spans[2]["attributes"]["gen_ai.tool.call.id"] == "c0"


def test_env_json_has_no_host_or_user_identifiers(tmp_path: Path) -> None:
    import getpass
    import socket

    run(
        make_config("env", n_trials=1),
        tmp_path,
        deps=make_deps(FakeAgent(search_then_final), FakeTransport(), [make_task("t01")]),
    )
    text = (tmp_path / "env" / "env.json").read_text()
    data = json.loads(text)
    assert "python" in data or "packages" in data
    assert socket.gethostname() not in text
    assert getpass.getuser() not in text
    assert str(Path.home()) not in text
    assert str(REPO) not in text


def test_transport_is_closed_after_run(tmp_path: Path) -> None:
    transport = FakeTransport()
    run(
        make_config("cl", n_trials=1),
        tmp_path,
        deps=make_deps(FakeAgent(search_then_final), transport, [make_task("t01")]),
    )
    assert transport.closed is True


def test_existing_run_dir_with_files_is_refused(tmp_path: Path) -> None:
    (tmp_path / "dup").mkdir()
    (tmp_path / "dup" / "report.json").write_text("{}")
    with pytest.raises(FileExistsError):
        run(
            make_config("dup", n_trials=1),
            tmp_path,
            deps=make_deps(FakeAgent(search_then_final), FakeTransport(), [make_task("t01")]),
        )


@pytest.mark.parametrize("bad_id", ["../escape", "a/b", "", ".hidden"])
def test_unsafe_run_ids_are_rejected(tmp_path: Path, bad_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        run(
            make_config(bad_id, n_trials=1),
            tmp_path,
            deps=make_deps(FakeAgent(search_then_final), FakeTransport(), [make_task("t01")]),
        )


def test_zero_trials_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_trials"):
        run(
            make_config("zero", n_trials=0),
            tmp_path,
            deps=make_deps(FakeAgent(search_then_final), FakeTransport(), [make_task("t01")]),
        )


def test_provider_style_exception_aborts_without_writing(tmp_path: Path) -> None:
    class Boom(RuntimeError):
        pass

    def explode(task: Task, history: list[Step]) -> AgentAction:
        raise Boom("provider down")

    with pytest.raises(Boom):
        run(
            make_config("boom", n_trials=1),
            tmp_path,
            deps=make_deps(FakeAgent(explode), FakeTransport(), [make_task("t01")]),
        )
    assert not (tmp_path / "boom" / "report.json").exists()


def test_build_agent_constructs_scripted_and_model_kinds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from passk.runner import build_agent

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-real")
    oracle = build_agent(AgentConfig(kind="oracle"))
    assert hasattr(oracle, "act")

    class Model:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        def create_chat_completion(self, **_kwargs: Any) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "{}"}}]}

    llama = build_agent(
        AgentConfig(
            kind="llamacpp",
            params={"model_path": str(tmp_path / "m.gguf"), "llama_factory": Model},
        )
    )
    assert llama.model_id == "m"  # type: ignore[attr-defined]

    hosted = build_agent(
        AgentConfig(kind="openai_compat", params={"model": "vendor/model:free", "post": None})
    )
    assert hosted.agent_id == "openai_compat:vendor/model:free"  # type: ignore[attr-defined]


def test_build_agent_rejects_unknown_kind() -> None:
    from passk.runner import build_agent

    with pytest.raises(ValueError):
        build_agent(AgentConfig.model_construct(kind="mystery", params={}))


def test_immediate_final_without_goal_is_a_false_success_claim(tmp_path: Path) -> None:
    def claim_now(task: Task, history: list[Step]) -> AgentAction:
        return AgentAction(kind="final", message="all done")

    report = run(
        make_config("fsc", n_trials=1),
        tmp_path,
        deps=make_deps(FakeAgent(claim_now), FakeTransport(), [make_task("t01")]),
    )
    assert report.failure_totals == {"false_success_claim": 1}
    traj = Trajectory.model_validate_json((tmp_path / "fsc" / "trajectories.jsonl").read_text())
    assert traj.success is False
