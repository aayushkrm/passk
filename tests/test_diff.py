"""Diff correctness on the SYNTHETIC fixtures, plus report I/O helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from passk.diff import (
    diff,
    is_synthetic,
    load_report,
    load_trajectories,
    mean_metric,
    report_schema_text,
)
from passk.schema import FailureCategory, RunReport, Trajectory

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> RunReport:
    return load_report(FIXTURES / name)


def _trajectories(report: RunReport, successes: dict[str, int]) -> list[Trajectory]:
    """Build trajectories whose first ``successes[task]`` trials succeed, others fail."""
    out: list[Trajectory] = []
    for task in report.tasks:
        for trial in range(report.config.n_trials):
            ok = trial < successes[task.task_id]
            out.append(
                Trajectory(
                    run_id=report.run_id,
                    task_id=task.task_id,
                    trial=trial,
                    seed=trial,
                    agent_id="test",
                    steps=[],
                    final_state={},
                    success=ok,
                    violations=[],
                    failure=FailureCategory.SUCCESS if ok else FailureCategory.WRONG_STATE,
                    wall_s=0.0,
                )
            )
    return out


BASE_IDS = ["t1-001", "t1-002", "t2-001", "t2-002", "t3-001", "t3-002"]


def test_fixtures_are_labelled_synthetic() -> None:
    for name in (
        "report_baseline.json",
        "report_regression.json",
        "report_minor_drop.json",
        "report_other_tasks.json",
    ):
        assert is_synthetic(_load(name).env_info), name


def test_mean_pass_hat_k_matches_hand_values() -> None:
    base = _load("report_baseline.json")
    cand = _load("report_regression.json")
    # baseline successes [5,5,4,4,3,3] out of 5 per task: mean pass^1 = 24/30
    assert mean_metric(base, "pass_hat_k", 1) == pytest.approx(24 / 30)
    # regression successes [2,3,1,2,0,1]: mean pass^1 = 9/30
    assert mean_metric(cand, "pass_hat_k", 1) == pytest.approx(9 / 30)
    # pass^5 is 1 only when all five trials succeed: two tasks at baseline
    assert mean_metric(base, "pass_hat_k", 5) == pytest.approx(2 / 6)
    assert mean_metric(base, "pass_hat_k", 9) is None


def test_diff_default_k_is_largest_shared_k() -> None:
    result = diff(_load("report_baseline.json"), _load("report_regression.json"))
    assert result["k"] == 5
    assert result["k_values"] == [1, 2, 3, 4, 5]


def test_diff_rows_are_sorted_by_largest_regression_first() -> None:
    result = diff(_load("report_baseline.json"), _load("report_regression.json"), k=1)
    deltas = [round(row["delta"], 12) for row in result["tasks"]]
    assert deltas == sorted(deltas)
    # every regression delta at k=1 is -0.6 or -0.4; the three -0.6 rows come first
    assert set(row["task_id"] for row in result["tasks"][:3]) == {"t1-001", "t2-001", "t3-001"}
    assert result["tasks"][0]["delta"] == pytest.approx(-0.6)
    # the first row is one of the three tasks that dropped by 0.6 at k=1
    first = result["tasks"][0]
    assert first["delta"] == pytest.approx(
        first["pass_hat_k_candidate"] - first["pass_hat_k_baseline"]
    )


def test_diff_pass_hat_k_summary_per_k() -> None:
    result = diff(_load("report_baseline.json"), _load("report_regression.json"), k=1)
    per_k = result["pass_hat_k"]
    assert per_k[1]["baseline"] == pytest.approx(24 / 30)
    assert per_k[1]["candidate"] == pytest.approx(9 / 30)
    assert per_k[1]["delta"] == pytest.approx(9 / 30 - 24 / 30)
    assert set(per_k) == {1, 2, 3, 4, 5}


def test_diff_identical_reports_have_zero_deltas() -> None:
    base = _load("report_baseline.json")
    result = diff(base, base.model_copy(update={"run_id": "copy"}))
    assert all(row["delta"] == pytest.approx(0.0) for row in result["tasks"])
    assert all(v["delta"] == pytest.approx(0.0) for v in result["pass_hat_k"].values())


def test_diff_reports_tasks_present_in_one_run_only() -> None:
    result = diff(_load("report_baseline.json"), _load("report_other_tasks.json"))
    assert result["only_in_baseline"] == ["t3-001", "t3-002"]
    assert result["only_in_candidate"] == ["t3-003"]
    assert sorted(row["task_id"] for row in result["tasks"]) == [
        "t1-001",
        "t1-002",
        "t2-001",
        "t2-002",
    ]


def test_diff_rejects_unknown_k_and_disjoint_tasks() -> None:
    base = _load("report_baseline.json")
    with pytest.raises(ValueError, match="not in both runs"):
        diff(base, _load("report_regression.json"), k=6)
    disjoint = base.model_copy(
        update={
            "tasks": [t.model_copy(update={"task_id": f"x-{i}"}) for i, t in enumerate(base.tasks)]
        }
    )
    with pytest.raises(ValueError, match="share no task ids"):
        diff(base, disjoint)


def test_failure_shift_without_trajectories_uses_totals() -> None:
    base = _load("report_baseline.json")
    cand = _load("report_regression.json")
    shift = diff(base, cand)["failure_shift"]
    assert shift["paired"] is False
    assert shift["matrix"] is None
    total_base = sum(v["baseline"] for v in shift["totals"].values())
    total_cand = sum(v["candidate"] for v in shift["totals"].values())
    assert total_base == sum(base.failure_totals.values()) == 30 - 24
    assert total_cand == sum(cand.failure_totals.values()) == 30 - 9


def test_failure_matrix_pairs_trials_by_task_and_trial() -> None:
    base = _load("report_baseline.json")
    cand = _load("report_regression.json")
    base_traj = _trajectories(base, dict(zip(BASE_IDS, [5, 5, 4, 4, 3, 3], strict=True)))
    cand_traj = _trajectories(cand, dict(zip(BASE_IDS, [2, 3, 1, 2, 0, 1], strict=True)))
    shift = diff(base, cand, trajectories_a=base_traj, trajectories_b=cand_traj)["failure_shift"]
    assert shift["paired"] is True
    assert shift["paired_trials"] == 30
    assert shift["unpaired_trials"] == 0
    matrix = shift["matrix"]
    # trials that succeed in both runs: min(baseline, candidate) per task
    assert matrix["success"]["success"] == sum(
        min(a, b) for a, b in zip([5, 5, 4, 4, 3, 3], [2, 3, 1, 2, 0, 1], strict=True)
    )
    # no trial succeeds in the candidate that failed in the baseline
    assert matrix["wrong_state"]["success"] == 0
    assert sum(sum(row.values()) for row in matrix.values()) == 30


def test_load_report_accepts_directory_and_file(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "report.json").write_text((FIXTURES / "report_baseline.json").read_text())
    assert load_report(run_dir).run_id == "synthetic-baseline"
    assert load_report(run_dir / "report.json").run_id == "synthetic-baseline"
    with pytest.raises(FileNotFoundError):
        load_report(tmp_path / "missing")


def test_load_trajectories_reads_jsonl_and_is_none_when_absent(tmp_path: Path) -> None:
    assert load_trajectories(tmp_path) is None
    base = _load("report_baseline.json")
    lines = [t.model_dump_json() for t in _trajectories(base, {tid: 5 for tid in BASE_IDS})]
    (tmp_path / "trajectories.jsonl").write_text("\n".join(lines) + "\n")
    loaded = load_trajectories(tmp_path)
    assert loaded is not None and len(loaded) == 30
    assert load_trajectories(tmp_path / "report.json") is not None


def test_report_schema_text_is_deterministic() -> None:
    assert report_schema_text() == report_schema_text()
