"""RunConfig.env_params reach the environment, in process and over MCP stdio."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import passk
from passk import runner
from passk.envs.base import load_tasks
from passk.envs.labstock import TRANSIENT_ERROR, validate_env_params
from passk.schema import AgentConfig, RunConfig, RunReport, Trajectory

REPO = Path(__file__).resolve().parents[1]
TASKS_GLOB = str(Path(passk.__file__).parent / "tasks" / "labstock" / "*.yaml")


def test_validate_env_params_accepts_flaky_p_and_defaults_to_empty() -> None:
    assert validate_env_params({}) == {}
    assert validate_env_params({"flaky_p": 0.2}) == {"flaky_p": 0.2}


@pytest.mark.parametrize(
    "params, message",
    [
        ({"seed": 1}, "unknown labstock env_params"),
        ({"flaky_p": 1.5}, r"must be in \[0, 1\]"),
        ({"flaky_p": "0.2"}, "must be a number"),
    ],
)
def test_validate_env_params_rejects_bad_input(params: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_env_params(params)


def test_runconfig_without_env_params_still_loads() -> None:
    data = {
        "run_id": "old",
        "tasks_glob": TASKS_GLOB,
        "agent": {"kind": "oracle"},
        "n_trials": 1,
        "base_seed": 1,
        "transport": "inprocess",
    }
    config = RunConfig.model_validate(data)
    assert config.env_params == {}


def _run(transport: str, tmp_path: Path, params: dict[str, object]) -> RunReport:
    tasks = sorted(load_tasks(TASKS_GLOB), key=lambda t: t.id)[:4]
    config = RunConfig(
        run_id=f"flaky-{transport}",
        tasks_glob=TASKS_GLOB,
        agent=AgentConfig(kind="oracle"),
        n_trials=1,
        base_seed=5,
        transport=transport,  # type: ignore[arg-type]
        env_params=params,
    )
    return runner.run(config, out_root=tmp_path, deps=runner.RunDeps(tasks=tasks))


def _trajectories(tmp_path: Path, run_id: str) -> list[Trajectory]:
    lines = (tmp_path / run_id / "trajectories.jsonl").read_text().splitlines()
    return [Trajectory.model_validate(json.loads(line)) for line in lines]


def test_flaky_p_one_makes_every_call_fail_once_and_the_oracle_recovers(tmp_path: Path) -> None:
    report = _run("inprocess", tmp_path, {"flaky_p": 1.0})
    assert report.failure_totals == {}  # the oracle recovers from every transient error
    trajectories = _trajectories(tmp_path, "flaky-inprocess")
    assert all(t.success for t in trajectories)
    transient = [
        step.result
        for t in trajectories
        for step in t.steps
        if step.result is not None and step.result.error == TRANSIENT_ERROR
    ]
    assert transient, "flaky_p=1.0 must produce transient failures"


def test_flaky_p_reaches_the_mcp_server_and_matches_in_process(tmp_path: Path) -> None:
    params = {"flaky_p": 0.5}
    _run("inprocess", tmp_path, params)
    _run("mcp_stdio", tmp_path, params)
    left = _trajectories(tmp_path, "flaky-inprocess")
    right = _trajectories(tmp_path, "flaky-mcp_stdio")
    assert len(left) == len(right) == 4

    def comparable(t: Trajectory) -> dict[str, object]:
        data = t.model_dump(mode="json")
        data.pop("wall_s")
        data.pop("run_id")
        for step in data["steps"]:
            step.pop("latency_s")
        return data

    for a, b in zip(left, right, strict=True):
        assert comparable(a) == comparable(b), a.task_id
    assert any(
        step.result is not None and step.result.error == TRANSIENT_ERROR
        for t in right
        for step in t.steps
    ), "the MCP server must have been built with flaky_p"


def test_e6_config_is_the_flaky_one_point_five_b_run() -> None:
    data = yaml.safe_load((REPO / "configs" / "e6_flaky.yaml").read_text(encoding="utf-8"))
    config = RunConfig.model_validate(data)
    assert config.env_params == {"flaky_p": 0.2}
    assert config.n_trials == 3
    assert config.transport == "mcp_stdio"
    assert config.agent.kind == "llamacpp"
    assert "1.5b" in str(config.agent.params["model_path"]).lower()
