"""CLI tests: help text, gate exit codes, JSON output, and clean errors for stubs and bad input."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from passk import mcp_server, runner
from passk.cli import app

runner_cli = CliRunner()
FIXTURES = Path(__file__).parent / "fixtures"
COMMANDS = ["run", "report", "diff", "gate", "serve-tools", "validate", "demo"]


def _run_dir(tmp_path: Path, name: str, fixture: str) -> str:
    directory = tmp_path / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "report.json").write_text((FIXTURES / fixture).read_text(encoding="utf-8"))
    return str(directory)


def _policy(tmp_path: Path, text: str) -> str:
    path = tmp_path / "gate.yaml"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_top_level_help_works() -> None:
    result = runner_cli.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in COMMANDS:
        assert name in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help_works(command: str) -> None:
    result = runner_cli.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_every_command_accepts_json_flag(command: str) -> None:
    result = runner_cli.invoke(app, [command, "--help"])
    assert "--json" in result.output


# gate exit codes -------------------------------------------------------------------------


def test_gate_passes_identical_runs_with_exit_0(tmp_path: Path) -> None:
    base = _run_dir(tmp_path, "a", "report_baseline.json")
    policy = _policy(tmp_path, "k: 1\nmax_drop: 0.05\n")
    result = runner_cli.invoke(app, ["gate", base, base, "--policy", policy])
    assert result.exit_code == 0, result.output
    assert "PASS" in result.output


def test_gate_significant_regression_exits_1(tmp_path: Path) -> None:
    base = _run_dir(tmp_path, "a", "report_baseline.json")
    cand = _run_dir(tmp_path, "b", "report_regression.json")
    policy = _policy(tmp_path, "k: 1\nmax_drop: 0.10\n")
    result = runner_cli.invoke(app, ["gate", base, cand, "--policy", policy])
    assert result.exit_code == 1, result.output
    assert "FAIL" in result.output and "regression" in result.output


def test_gate_non_significant_drop_exits_0(tmp_path: Path) -> None:
    base = _run_dir(tmp_path, "a", "report_baseline.json")
    cand = _run_dir(tmp_path, "b", "report_minor_drop.json")
    policy = _policy(tmp_path, "k: 1\nmax_drop: 0.05\n")
    result = runner_cli.invoke(app, ["gate", base, cand, "--policy", policy])
    assert result.exit_code == 0, result.output


def test_gate_significance_switch_turns_drop_into_failure(tmp_path: Path) -> None:
    base = _run_dir(tmp_path, "a", "report_baseline.json")
    cand = _run_dir(tmp_path, "b", "report_minor_drop.json")
    policy = _policy(tmp_path, "k: 1\nmax_drop: 0.05\nrequire_significance: false\n")
    result = runner_cli.invoke(app, ["gate", base, cand, "--policy", policy])
    assert result.exit_code == 1, result.output


def test_gate_floor_breach_exits_1(tmp_path: Path) -> None:
    base = _run_dir(tmp_path, "a", "report_baseline.json")
    policy = _policy(tmp_path, "k: 1\nmax_drop: 0.5\nmin_pass_hat_k: 0.9\n")
    result = runner_cli.invoke(app, ["gate", base, base, "--policy", policy])
    assert result.exit_code == 1, result.output
    assert "floor" in result.output


def test_gate_json_output_is_parseable(tmp_path: Path) -> None:
    base = _run_dir(tmp_path, "a", "report_baseline.json")
    cand = _run_dir(tmp_path, "b", "report_regression.json")
    policy = _policy(tmp_path, "k: 1\nmax_drop: 0.10\n")
    result = runner_cli.invoke(app, ["gate", base, cand, "--policy", policy, "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["passed"] is False
    assert payload["synthetic"] is True
    assert payload["policy"]["k"] == 1
    assert payload["baseline"] == "synthetic-baseline"


@pytest.mark.parametrize(
    "policy_text",
    ["k: 1\nmax_drop: 0.1\nmax_dorp: 0.2\n", "k: 0\nmax_drop: 0.1\n", "just text\n"],
)
def test_gate_bad_policy_exits_2(tmp_path: Path, policy_text: str) -> None:
    base = _run_dir(tmp_path, "a", "report_baseline.json")
    policy = _policy(tmp_path, policy_text)
    result = runner_cli.invoke(app, ["gate", base, base, "--policy", policy])
    assert result.exit_code == 2, result.output
    assert "Traceback" not in result.output
    assert "passk: error" in result.output


def test_gate_missing_run_exits_2(tmp_path: Path) -> None:
    base = _run_dir(tmp_path, "a", "report_baseline.json")
    policy = _policy(tmp_path, "k: 1\nmax_drop: 0.1\n")
    result = runner_cli.invoke(app, ["gate", base, str(tmp_path / "nope"), "--policy", policy])
    assert result.exit_code == 2
    assert "no report.json" in result.output


def test_gate_task_mismatch_exits_2_with_json_error(tmp_path: Path) -> None:
    base = _run_dir(tmp_path, "a", "report_baseline.json")
    other = _run_dir(tmp_path, "b", "report_other_tasks.json")
    policy = _policy(tmp_path, "k: 1\nmax_drop: 0.1\n")
    result = runner_cli.invoke(app, ["gate", base, other, "--policy", policy, "--json"])
    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["exit_code"] == 2
    assert "task ids differ" in payload["error"]


def test_gate_requires_policy_option(tmp_path: Path) -> None:
    base = _run_dir(tmp_path, "a", "report_baseline.json")
    result = runner_cli.invoke(app, ["gate", base, base])
    assert result.exit_code == 2


# diff and report ---------------------------------------------------------------------------


def test_diff_json_lists_tasks_worst_first(tmp_path: Path) -> None:
    base = _run_dir(tmp_path, "a", "report_baseline.json")
    cand = _run_dir(tmp_path, "b", "report_regression.json")
    result = runner_cli.invoke(app, ["diff", base, cand, "--json", "--k", "1"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["k"] == 1
    assert payload["tasks"][0]["delta"] <= payload["tasks"][-1]["delta"]


def test_diff_human_output_runs(tmp_path: Path) -> None:
    base = _run_dir(tmp_path, "a", "report_baseline.json")
    cand = _run_dir(tmp_path, "b", "report_other_tasks.json")
    result = runner_cli.invoke(app, ["diff", base, cand])
    assert result.exit_code == 0, result.output
    assert "SYNTHETIC" in result.output
    assert "t3-003" in result.output  # present only in the candidate, listed in the table


@pytest.mark.parametrize("fmt", ["md", "json", "html"])
def test_report_formats_render(tmp_path: Path, fmt: str) -> None:
    run = _run_dir(tmp_path, "a", "report_baseline.json")
    result = runner_cli.invoke(app, ["report", run, "--format", fmt])
    assert result.exit_code == 0, result.output
    if fmt == "json":
        assert json.loads(result.output)["run_id"] == "synthetic-baseline"
    if fmt == "html":
        assert "<svg" in result.output


def test_report_writes_file_with_out_option(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "a", "report_baseline.json")
    target = tmp_path / "out.html"
    result = runner_cli.invoke(app, ["report", run, "--format", "html", "--out", str(target)])
    assert result.exit_code == 0, result.output
    assert target.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_report_bad_format_exits_2(tmp_path: Path) -> None:
    run = _run_dir(tmp_path, "a", "report_baseline.json")
    result = runner_cli.invoke(app, ["report", run, "--format", "pdf"])
    assert result.exit_code == 2
    assert "--format" in result.output


# stubs and runtime errors ------------------------------------------------------------------


def _not_yet(*_args: object, **_kwargs: object) -> None:
    raise NotImplementedError("run: scheduled for work package P-D")


def test_run_without_runner_exits_2_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(runner, "run", _not_yet)
    config = tmp_path / "run.yaml"
    config.write_text(
        "run_id: r\ntasks_glob: x\nagent: {kind: oracle}\nn_trials: 1\n"
        "base_seed: 0\ntransport: inprocess\n",
        encoding="utf-8",
    )
    result = runner_cli.invoke(app, ["run", "-c", str(config)])
    assert result.exit_code == 2
    assert "not implemented yet" in result.output
    assert "Traceback" not in result.output


def test_run_missing_config_exits_2(tmp_path: Path) -> None:
    result = runner_cli.invoke(app, ["run", "-c", str(tmp_path / "absent.yaml"), "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout)["exit_code"] == 2


def test_demo_without_runner_exits_2_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "run", _not_yet)
    result = runner_cli.invoke(app, ["demo"])
    assert result.exit_code == 2
    assert "Traceback" not in result.output


def test_validate_without_task_loader_exits_2_cleanly(monkeypatch: pytest.MonkeyPatch) -> None:
    from passk.envs import base

    monkeypatch.setattr(base, "load_tasks", _not_yet)
    result = runner_cli.invoke(app, ["validate"])
    assert result.exit_code == 2
    assert "not implemented yet" in result.output


def test_serve_tools_rejects_unknown_environment() -> None:
    result = runner_cli.invoke(app, ["serve-tools", "nope"])
    assert result.exit_code == 2
    assert "only labstock" in result.output


def test_serve_tools_calls_the_labstock_server(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        mcp_server, "serve_labstock_stdio", lambda: calls.append("labstock"), raising=False
    )
    result = runner_cli.invoke(app, ["serve-tools"])
    assert result.exit_code == 0, result.output
    assert calls == ["labstock"]
