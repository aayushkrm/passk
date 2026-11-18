"""A partial report is marked partial everywhere, and the CLI stops cleanly with exit code 2."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from typer.testing import CliRunner

import passk
from passk import runner
from passk.agents.openai_compat import HttpResponse, OpenAICompatAgent
from passk.cli import app
from passk.diff import load_report
from passk.schema import RunReport
from passk.site.build import build_site, render_report

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
TASKS_GLOB = str(Path(passk.__file__).parent / "tasks" / "labstock" / "*.yaml")
RESET = "2026-11-20T06:00:00+00:00"
NOW = 1_800_000_000.0


def _partial(path: Path | None = None) -> RunReport:
    base = load_report(FIXTURES / "report_baseline.json")
    return base.model_copy(
        update={
            "complete": False,
            "stop_reason": "provider returned HTTP 429: account quota exhausted",
            "reset_at": RESET,
        }
    )


def _render_readme_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "render_readme", REPO / "scripts" / "render_readme.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_markdown_report_says_partial_and_names_the_reset() -> None:
    text = render_report(_partial(), "md")
    assert "**PARTIAL RUN**" in text
    assert "- Complete: no" in text
    assert RESET in text
    assert "not a full result" in text


def test_complete_report_has_no_partial_banner() -> None:
    text = render_report(load_report(FIXTURES / "report_baseline.json"), "md")
    assert "PARTIAL" not in text
    assert "- Complete: yes" in text


def test_site_marks_partial_runs_on_the_run_page_and_the_index(tmp_path: Path) -> None:
    index = build_site([_partial()], tmp_path / "site")
    run_page = next((tmp_path / "site" / "runs").glob("*/index.html")).read_text()
    assert "PARTIAL RUN." in run_page
    assert RESET in run_page
    assert "PARTIAL" in index.read_text()


def test_readme_block_labels_a_partial_run(tmp_path: Path) -> None:
    module = _render_readme_module()
    run_dir = tmp_path / "partial-run"
    run_dir.mkdir()
    (run_dir / "report.json").write_text(_partial().model_dump_json(), encoding="utf-8")
    record = module._run_record(_partial(), run_dir / "report.json")
    block = "\n".join(module._run_block(record))
    assert "PARTIAL" in block
    assert "not a full result" in block
    assert f"resets at {RESET}" in block
    assert f"of {record['episodes_total']} episodes" in block


def test_gate_refuses_an_incomplete_report(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    candidate.write_text(_partial().model_dump_json(), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "gate",
            str(FIXTURES / "report_baseline.json"),
            str(candidate),
            "--policy",
            str(REPO / "configs" / "gate.yaml"),
        ],
    )
    assert result.exit_code == 2
    assert "incomplete candidate report" in result.output


def _quota_agent(_cfg: Any) -> OpenAICompatAgent:
    quota = HttpResponse(
        429,
        {
            "X-RateLimit-Limit": "50",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int((NOW + 4 * 3600) * 1000)),
        },
        b'{"error": {"message": "daily limit reached"}}',
    )

    def post(url: str, headers: dict[str, str], body: bytes, timeout: float) -> HttpResponse:
        return quota

    return OpenAICompatAgent(
        model="test/model", post=post, sleep=lambda _s: None, wall_clock=lambda: NOW
    )


def test_cli_run_stops_cleanly_on_quota_with_exit_2_and_the_reset_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-SECRET-0123456789")
    monkeypatch.setattr(runner, "build_agent", _quota_agent)
    monkeypatch.setattr(runner.time, "perf_counter", lambda: 0.0)
    config = {
        "run_id": "cli-quota",
        "tasks_glob": TASKS_GLOB,
        "agent": {"kind": "openai_compat", "params": {"model": "test/model"}},
        "n_trials": 1,
        "base_seed": 1,
        "transport": "inprocess",
    }
    config_path = tmp_path / "quota.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    out = tmp_path / "out"

    result = CliRunner().invoke(app, ["run", "-c", str(config_path), "--out", str(out)])

    assert result.exit_code == 2, result.output
    assert "run stopped before it finished" in result.output
    assert "resets at" in result.output
    assert "0 of 24" in result.output
    report = json.loads((out / "cli-quota" / "report.json").read_text())
    assert report["complete"] is False
    assert (out / "cli-quota" / "state.json").is_file()


def test_cli_run_without_a_reset_header_still_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise runner.RunIncomplete(_partial(), "network down", None, None)

    monkeypatch.setattr(runner, "run", boom)
    config_path = tmp_path / "c.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "x",
                "tasks_glob": TASKS_GLOB,
                "agent": {"kind": "oracle"},
                "n_trials": 1,
                "base_seed": 1,
                "transport": "inprocess",
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["run", "-c", str(config_path)])
    assert result.exit_code == 2
    assert "network down" in result.output
    assert "resets at" not in result.output


def test_cli_resume_flags_are_passed_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}

    def fake_run(config: Any, out_root: Any, deps: Any = None, resume: Any = None, **_: Any) -> Any:
        seen["resume"] = resume
        return _partial().model_copy(
            update={"complete": True, "stop_reason": None, "reset_at": None}
        )

    monkeypatch.setattr(runner, "run", fake_run)
    config_path = tmp_path / "c.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "run_id": "y",
                "tasks_glob": TASKS_GLOB,
                "agent": {"kind": "oracle"},
                "n_trials": 1,
                "base_seed": 1,
                "transport": "inprocess",
            }
        ),
        encoding="utf-8",
    )
    CliRunner().invoke(app, ["run", "-c", str(config_path), "--no-resume"])
    assert seen["resume"] is False
    CliRunner().invoke(app, ["run", "-c", str(config_path)])
    assert seen["resume"] is None
