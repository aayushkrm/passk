"""Site and single-run rendering from the SYNTHETIC fixtures, plus the committed JSON Schema."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from passk.diff import load_report, report_schema_text
from passk.schema import FailureCategory, RunReport, Trajectory
from passk.site.build import build_site, render_report

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
SCHEMA_FILE = ROOT / "schemas" / "run_report.schema.json"


def _fixture_reports() -> list[RunReport]:
    return [load_report(p) for p in sorted(FIXTURES.glob("report_*.json"))]


def _baseline() -> RunReport:
    return load_report(FIXTURES / "report_baseline.json")


def test_committed_schema_matches_generated_schema() -> None:
    assert SCHEMA_FILE.read_text(encoding="utf-8") == report_schema_text()


def test_build_site_from_fixtures_writes_all_pages(tmp_path: Path) -> None:
    index = build_site(_fixture_reports(), tmp_path)
    assert index == tmp_path / "index.html"
    assert (tmp_path / "llms.txt").is_file()
    for run_id in (
        "synthetic-baseline",
        "synthetic-regression",
        "synthetic-minor-drop",
        "synthetic-other-tasks",
    ):
        run_dir = tmp_path / "runs" / run_id
        assert (run_dir / "index.html").is_file()
        copied = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
        assert copied["run_id"] == run_id


def test_run_page_has_expected_sections(tmp_path: Path) -> None:
    build_site([_baseline()], tmp_path)
    page = (tmp_path / "runs" / "synthetic-baseline" / "index.html").read_text(encoding="utf-8")
    for heading in (
        "Summary",
        "pass^k against k",
        "Success rate per task",
        "Failure categories",
        "Tasks",
        "Record",
    ):
        assert heading in page, heading
    # three inline charts: curves, heatmap, failure bars
    assert page.count("<svg") == 3
    assert 'class="banner"' in page and "SYNTHETIC" in page
    assert 'href="report.json"' in page


def test_pages_load_nothing_from_the_network(tmp_path: Path) -> None:
    build_site(_fixture_reports(), tmp_path)
    files = [tmp_path / "index.html", *tmp_path.glob("runs/*/index.html")]
    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "http://" not in text and "https://" not in text, path
        assert not re.search(r"<script|<link|@import|src=", text), path


def test_index_lists_runs_with_synthetic_badges(tmp_path: Path) -> None:
    build_site(_fixture_reports(), tmp_path)
    index = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "synthetic-regression" in index
    assert index.count('class="badge">SYNTHETIC') == 4
    assert index.count("<svg") == 1


def test_banner_absent_for_a_run_not_marked_synthetic(tmp_path: Path) -> None:
    real = _baseline().model_copy(update={"env_info": {"git_sha": "abc"}})
    build_site([real], tmp_path)
    page = (tmp_path / "runs" / "synthetic-baseline" / "index.html").read_text(encoding="utf-8")
    assert 'class="banner"' not in page
    assert "SYNTHETIC" not in page
    assert "SYNTHETIC" not in (tmp_path / "index.html").read_text(encoding="utf-8")


def _trajectory(report: RunReport, task_id: str, trial: int, ok: bool) -> Trajectory:
    return Trajectory.model_validate(
        {
            "run_id": report.run_id,
            "task_id": task_id,
            "trial": trial,
            "seed": trial,
            "agent_id": "test",
            "steps": [
                {
                    "index": 0,
                    "action": {
                        "kind": "tool_call",
                        "tool_call": {
                            "call_id": "c0",
                            "name": "place_order",
                            "arguments": {"compound_id": "<script>alert(1)</script>"},
                        },
                        "message": None,
                    },
                    "result": {"call_id": "c0", "ok": ok, "content": None, "error": None},
                    "latency_s": 0.01,
                },
                {
                    "index": 1,
                    "action": {"kind": "final", "tool_call": None, "message": "Done."},
                    "result": None,
                    "latency_s": 0.01,
                },
            ],
            "final_state": {},
            "success": ok,
            "violations": [],
            "failure": "success" if ok else "wrong_state",
            "wall_s": 0.02,
        }
    )


def test_trajectory_viewer_and_trial_heatmap_from_run_directory(tmp_path: Path) -> None:
    base = _baseline()
    run_dir = tmp_path / "source" / "run"
    run_dir.mkdir(parents=True)
    (run_dir / "report.json").write_text(base.model_dump_json(), encoding="utf-8")
    trajectories = [
        _trajectory(base, task.task_id, trial, trial < 3)
        for task in base.tasks
        for trial in range(base.config.n_trials)
    ]
    (run_dir / "trajectories.jsonl").write_text(
        "\n".join(t.model_dump_json() for t in trajectories) + "\n", encoding="utf-8"
    )
    out = tmp_path / "site"
    build_site(tmp_path / "source", out)
    page = (out / "runs" / "synthetic-baseline" / "index.html").read_text(encoding="utf-8")
    assert "Task by trial outcome" in page
    assert "Trajectories (30)" in page
    assert "place_order" in page
    # user-controlled text is escaped, never injected as markup
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_render_report_formats(tmp_path: Path) -> None:
    base = _baseline()
    md = render_report(base, "md")
    assert md.startswith("# passk run synthetic-baseline")
    assert "SYNTHETIC" in md
    assert "| 5 |" in md
    as_json = render_report(base, "json")
    assert RunReport.model_validate_json(as_json) == base
    html = render_report(base, "html")
    assert "<!doctype html>" in html and "<svg" in html
    assert 'href="report.json"' not in html  # standalone page, no sibling file
    with pytest.raises(ValueError, match="unknown format"):
        render_report(base, "pdf")  # type: ignore[arg-type]


def test_render_report_marks_failure_categories_in_order() -> None:
    md = render_report(_baseline(), "md")
    names = [c.value for c in FailureCategory if c != FailureCategory.SUCCESS]
    assert "| success |" not in md  # a pass is not a failure bar
    positions = [md.index(f"| {name} |") for name in names]
    assert positions == sorted(positions)


def test_build_site_rejects_empty_and_duplicate_inputs(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no report.json"):
        build_site(empty, tmp_path / "out")
    base = _baseline()
    with pytest.raises(ValueError, match="duplicate run ids"):
        build_site([base, base], tmp_path / "out")
    with pytest.raises(FileNotFoundError):
        build_site(tmp_path / "missing", tmp_path / "out")
