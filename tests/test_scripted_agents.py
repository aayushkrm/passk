"""Scripted agent tests, run through a local driver loop (no runner dependency).

SYNTHETIC: these agents are scripted, so their results are never model results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from passk.agents.scripted import NoisyOracleAgent, OracleAgent, ReplayAgent
from passk.envs import LabStockEnv, load_tasks, tasks_dir
from passk.goals import check_goal
from passk.schema import (
    Agent,
    AgentAction,
    FailureCategory,
    Step,
    Task,
    Trajectory,
)
from passk.schema import Env as EnvProtocol

TASKS: list[Task] = load_tasks(str(tasks_dir("labstock") / "*.yaml"))


def run_episode(
    env: LabStockEnv, agent: Agent, task: Task, seed: int = 0
) -> tuple[bool, list[Step], dict[str, Any], list[str]]:
    """Drive one episode: act, call, repeat until a final action or the turn budget.

    Returns:
        (success, steps, final_state, violations). A final action counts as a step.
    """
    handle = env.reset(task, seed)
    tools = env.tools()
    history: list[Step] = []
    for index in range(task.max_turns):
        action = agent.act(task, history, tools)
        if action.kind == "final":
            history.append(Step(index=index, action=action, result=None, latency_s=0.0))
            break
        assert action.tool_call is not None
        result = env.call(handle, action.tool_call)
        history.append(Step(index=index, action=action, result=result, latency_s=0.0))
    state = env.state(handle)
    violations = env.violations(handle)
    return check_goal(task, state, violations), history, state, violations


def _solved(results: list[bool]) -> int:
    return sum(results)


def test_labstock_env_satisfies_the_env_protocol_shape() -> None:
    env: EnvProtocol = LabStockEnv()
    assert callable(env.reset)
    assert callable(env.tools)
    assert callable(env.call)
    assert callable(env.state)
    assert callable(env.violations)


def test_oracle_solves_all_24_tasks_in_process() -> None:
    results: list[bool] = []
    failures: list[str] = []
    for task in TASKS:
        success, steps, _, violations = run_episode(LabStockEnv(), OracleAgent(), task)
        results.append(success)
        if not success:
            failures.append(f"{task.id}: {violations}")
        assert steps[-1].action.kind == "final"
        assert len(steps) <= task.max_turns
    assert _solved(results) == 24, failures


def test_oracle_replays_the_reference_calls_in_order() -> None:
    task = TASKS[0]
    _, steps, _, _ = run_episode(LabStockEnv(), OracleAgent(), task)
    issued = [s.action.tool_call for s in steps if s.action.tool_call is not None]
    assert [c.name for c in issued] == [r.name for r in task.reference_solution]
    assert [c.call_id for c in issued] == [r.call_id for r in task.reference_solution]


def test_oracle_recovers_from_transient_failures() -> None:
    # With flaky_p=1 every new call fails once. The oracle re-issues it with a fresh call id.
    results: list[bool] = []
    saw_retry = False
    for task in TASKS:
        success, steps, _, _ = run_episode(LabStockEnv(flaky_p=1.0), OracleAgent(), task)
        results.append(success)
        ids = [s.action.tool_call.call_id for s in steps if s.action.tool_call is not None]
        assert len(ids) == len(set(ids))
        saw_retry = saw_retry or any("-retry" in call_id for call_id in ids)
    assert _solved(results) == 24
    assert saw_retry


def test_oracle_stops_at_final_action_after_solution() -> None:
    task = next(t for t in TASKS if t.id == "t1_01")
    env = LabStockEnv()
    handle = env.reset(task, 0)
    agent = OracleAgent()
    history: list[Step] = []
    for index in range(len(task.reference_solution)):
        action = agent.act(task, history, env.tools())
        assert action.kind == "tool_call"
        result = env.call(handle, action.tool_call)  # type: ignore[arg-type]
        history.append(Step(index=index, action=action, result=result, latency_s=0.0))
    final = agent.act(task, history, env.tools())
    assert final.kind == "final"
    assert final.tool_call is None


def test_noisy_oracle_with_p_one_fails_all_24_tasks() -> None:
    results: list[bool] = []
    for task in TASKS:
        success, _, _, violations = run_episode(
            LabStockEnv(), NoisyOracleAgent(p_error=1.0, seed=3), task
        )
        results.append(success)
        assert violations, f"{task.id} should record a violation"
    assert _solved(results) == 0


def test_noisy_oracle_with_p_zero_matches_the_oracle() -> None:
    for task in TASKS:
        _, oracle_steps, _, _ = run_episode(LabStockEnv(), OracleAgent(), task)
        success, noisy_steps, _, _ = run_episode(
            LabStockEnv(), NoisyOracleAgent(p_error=0.0, seed=5), task
        )
        assert success
        assert [s.action for s in noisy_steps] == [s.action for s in oracle_steps]


def test_noisy_oracle_corrupts_only_the_chosen_steps() -> None:
    task = next(t for t in TASKS if t.id == "t2_05")
    agent = NoisyOracleAgent(p_error=0.5, seed=7)
    _, steps, _, _ = run_episode(LabStockEnv(), agent, task)
    corrupted = [
        s.index
        for s in steps
        if s.action.tool_call is not None and s.action.tool_call.call_id.startswith("noise-")
    ]
    assert corrupted, "seed 7 at p=0.5 should corrupt at least one step of t2_05"
    for step in steps:
        if step.action.tool_call is not None and step.index in corrupted:
            assert step.action.tool_call.call_id == f"noise-{step.index}"
            assert step.action.tool_call.name == "place_order"


def test_noisy_oracle_is_deterministic_per_seed_task_and_step() -> None:
    def trace(seed: int) -> list[list[AgentAction]]:
        out = []
        for task in TASKS:
            _, steps, _, _ = run_episode(
                LabStockEnv(), NoisyOracleAgent(p_error=0.3, seed=seed), task
            )
            out.append([s.action for s in steps])
        return out

    assert trace(7) == trace(7)
    assert trace(7) != trace(8)


def test_noisy_oracle_rejects_probability_outside_unit_interval() -> None:
    with pytest.raises(ValueError):
        NoisyOracleAgent(p_error=-0.1, seed=0)
    with pytest.raises(ValueError):
        NoisyOracleAgent(p_error=1.1, seed=0)


def test_noisy_oracle_never_corrupts_the_final_action() -> None:
    task = next(t for t in TASKS if t.id == "t1_01")
    agent = NoisyOracleAgent(p_error=1.0, seed=0)
    history = [
        Step(
            index=0,
            action=AgentAction(kind="tool_call", tool_call=task.reference_solution[0]),
            result=None,
            latency_s=0.0,
        )
    ]
    final = agent.act(task, history, [])
    assert final.kind == "final"


def _write_trajectories(path: Path) -> dict[str, list[Step]]:
    """Run the oracle over every task and write its trajectories as JSONL."""
    histories: dict[str, list[Step]] = {}
    lines: list[str] = []
    for task in TASKS:
        env = LabStockEnv()
        success, steps, state, violations = run_episode(env, OracleAgent(), task)
        histories[task.id] = steps
        trajectory = Trajectory(
            run_id="synthetic-oracle",
            task_id=task.id,
            trial=0,
            seed=0,
            agent_id="oracle",
            steps=steps,
            final_state=state,
            success=success,
            violations=violations,
            failure=FailureCategory.SUCCESS if success else FailureCategory.WRONG_STATE,
            wall_s=0.0,
        )
        lines.append(trajectory.model_dump_json())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return histories


def test_replay_agent_reproduces_recorded_actions(tmp_path: Path) -> None:
    path = tmp_path / "trajectories.jsonl"
    recorded = _write_trajectories(path)
    replay = ReplayAgent(path)
    for task in TASKS:
        success, steps, _, _ = run_episode(LabStockEnv(), replay, task)
        assert success
        assert [s.action for s in steps] == [s.action for s in recorded[task.id]]


def test_replay_agent_returns_final_when_recording_is_exhausted(tmp_path: Path) -> None:
    path = tmp_path / "one.jsonl"
    _write_trajectories(path)
    replay = ReplayAgent(path)
    task = next(t for t in TASKS if t.id == "t1_01")
    action = replay.act(
        task, [Step(index=0, action=AgentAction(kind="final"), result=None, latency_s=0.0)] * 10, []
    )
    assert action.kind == "final"


def test_replay_agent_rejects_a_task_with_no_recording(tmp_path: Path) -> None:
    path = tmp_path / "partial.jsonl"
    _write_trajectories(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:1]) + "\n", encoding="utf-8")
    replay = ReplayAgent(path)
    missing = next(t for t in TASKS if t.id == "t3_08")
    with pytest.raises(ValueError, match="no recorded trajectory for task 't3_08'"):
        replay.act(missing, [], [])


def test_replay_agent_rejects_a_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="bad.jsonl:1"):
        ReplayAgent(path)


def test_replay_agent_skips_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "blank.jsonl"
    _write_trajectories(path)
    path.write_text("\n" + path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    replay = ReplayAgent(path)
    task = next(t for t in TASKS if t.id == "t2_01")
    success, _, _, _ = run_episode(LabStockEnv(), replay, task)
    assert success


def test_noisy_oracle_trials_are_independent_given_episode_seeds() -> None:
    """Different episode seeds must give different outcomes for one task (regression test)."""
    task = next(t for t in load_tasks("src/passk/tasks/labstock/t2_*.yaml") if t.id == "t2_01")
    outcomes = []
    for episode_seed in range(60):
        agent = NoisyOracleAgent(p_error=0.5, seed=0)
        agent.begin_episode(episode_seed)
        outcomes.append(run_episode(LabStockEnv(), agent, task, seed=episode_seed)[0])
    assert 0 < sum(outcomes) < len(outcomes)
