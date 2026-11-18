"""End-to-end check: the in-process and MCP stdio transports give identical trajectories.

Runs the real runner on all packaged LabStock tasks over both transports. The MCP side starts
``python -m passk.mcp_server labstock`` as a subprocess. Skipped when the ``mcp`` package is
missing. Takes a few seconds on a laptop.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp")

import passk  # noqa: E402
from passk import runner  # noqa: E402
from passk.envs.labstock import LabStockEnv  # noqa: E402
from passk.schema import AgentConfig, RunConfig, Trajectory  # noqa: E402
from passk.transport import MCPStdioTransport  # noqa: E402

TASKS_GLOB = str(Path(passk.__file__).parent / "tasks" / "labstock" / "*.yaml")
TIME_BUDGET_S = 60.0


def _comparable(trajectory: Trajectory) -> dict[str, Any]:
    """Drop the fields that legitimately differ between runs (timings and run id)."""
    data = trajectory.model_dump(mode="json")
    data.pop("wall_s")
    data.pop("run_id")
    for step in data["steps"]:
        step.pop("latency_s")
    return data


def _run_trajectories(
    tmp_path: Path, transport: str, agent: AgentConfig, n_trials: int
) -> list[Trajectory]:
    config = RunConfig(
        run_id=f"integ-{transport}-{agent.kind}",
        tasks_glob=TASKS_GLOB,
        agent=agent,
        n_trials=n_trials,
        base_seed=7,
        transport=transport,  # type: ignore[arg-type]
    )
    runner.run(config, out_root=tmp_path)
    lines = (tmp_path / config.run_id / "trajectories.jsonl").read_text().splitlines()
    return [Trajectory.model_validate(json.loads(line)) for line in lines if line.strip()]


def test_mcp_tool_list_matches_the_environment() -> None:
    transport = MCPStdioTransport(env_name="labstock")
    try:
        served = transport.tools()
    finally:
        transport.close()
    expected = LabStockEnv().tools()
    assert [spec.name for spec in served] == [spec.name for spec in expected]
    assert [spec.parameters for spec in served] == [spec.parameters for spec in expected]


def test_oracle_on_all_tasks_gives_identical_trajectories(tmp_path: Path) -> None:
    started = time.perf_counter()
    agent = AgentConfig(kind="oracle")
    in_process = _run_trajectories(tmp_path, "inprocess", agent, n_trials=1)
    over_mcp = _run_trajectories(tmp_path, "mcp_stdio", agent, n_trials=1)

    assert len(in_process) == len(over_mcp) == 24
    assert all(t.success for t in in_process)
    assert all(t.success for t in over_mcp)
    for left, right in zip(in_process, over_mcp, strict=True):
        assert _comparable(left) == _comparable(right), left.task_id
    assert time.perf_counter() - started < TIME_BUDGET_S


def test_noisy_oracle_violations_match_across_transports(tmp_path: Path) -> None:
    # A corrupted step records a policy violation; the MCP path must report it identically.
    agent = AgentConfig(kind="noisy_oracle", params={"p_error": 0.3, "seed": 3})
    in_process = _run_trajectories(tmp_path, "inprocess", agent, n_trials=2)
    over_mcp = _run_trajectories(tmp_path, "mcp_stdio", agent, n_trials=2)

    assert len(in_process) == len(over_mcp) == 48
    assert any(t.violations for t in in_process), "seed should produce at least one violation"
    for left, right in zip(in_process, over_mcp, strict=True):
        assert _comparable(left) == _comparable(right), (left.task_id, left.trial)
