"""Compare two runs and read run artefacts.

``diff(a, b)`` compares a candidate run ``b`` with a baseline ``a``. It reports per-task
pass^k deltas (sorted so the largest regression comes first), the mean pass^k for each k,
and a failure-category shift. The shift is a baseline-by-candidate matrix when both runs'
trajectories are supplied; trials are paired by ``(task_id, trial)``. Without trajectories
only the per-category totals from the reports are available.

The module also holds the report I/O helpers shared by the CLI and the site builder.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from passk.schema import FailureCategory, RunReport, Trajectory

REPORT_FILENAME = "report.json"
TRAJECTORIES_FILENAME = "trajectories.jsonl"
SYNTHETIC_KEY = "synthetic"
SYNTHETIC_LABEL = "SYNTHETIC"


def is_synthetic(env_info: dict[str, Any]) -> bool:
    """Return True when ``env_info`` marks a run as synthetic (scripted or fixture data).

    Args:
        env_info: The ``env_info`` mapping of a RunReport.

    Returns:
        True only when ``env_info["synthetic"]`` is the boolean ``True``.
    """
    return env_info.get(SYNTHETIC_KEY) is True


def load_report(path: Path) -> RunReport:
    """Load a RunReport from a run directory or from a report file.

    Args:
        path: A run directory holding ``report.json``, or the JSON file itself.

    Returns:
        The validated report.

    Raises:
        FileNotFoundError: If no report file exists at ``path``.
        ValueError: If the file does not validate as a RunReport.
    """
    target = path / REPORT_FILENAME if path.is_dir() else path
    if not target.is_file():
        raise FileNotFoundError(f"no {REPORT_FILENAME} found at {path}")
    return RunReport.model_validate_json(target.read_text(encoding="utf-8"))


def load_trajectories(path: Path) -> list[Trajectory] | None:
    """Read ``trajectories.jsonl`` next to a report.

    Args:
        path: A run directory, or the ``report.json`` file inside it.

    Returns:
        The trajectories in file order, or None when the file is absent.
    """
    base = path if path.is_dir() else path.parent
    source = base / TRAJECTORIES_FILENAME
    if not source.is_file():
        return None
    lines = source.read_text(encoding="utf-8").splitlines()
    return [Trajectory.model_validate_json(line) for line in lines if line.strip()]


def report_schema_text() -> str:
    """Return the JSON Schema of RunReport in the form committed to the repository.

    Returns:
        Sorted, two-space-indented JSON with a trailing newline.
    """
    return json.dumps(RunReport.model_json_schema(), indent=2, sort_keys=True) + "\n"


def mean_metric(
    report: RunReport, metric: Literal["pass_hat_k", "pass_at_k"], k: int
) -> float | None:
    """Mean of a per-task metric at ``k`` over the tasks that report it.

    Args:
        report: The run to summarise.
        metric: ``"pass_hat_k"`` or ``"pass_at_k"``.
        k: The k to average.

    Returns:
        The equal-weight mean over tasks, or None when no task has a value at ``k``.
    """
    values = [getattr(task, metric)[k] for task in report.tasks if k in getattr(task, metric)]
    if not values:
        return None
    return sum(values) / len(values)


def _common_k(a: RunReport, b: RunReport) -> list[int]:
    return sorted(set(a.config.k_values) & set(b.config.k_values))


def _failure_shift(
    a: RunReport,
    b: RunReport,
    trajectories_a: Sequence[Trajectory] | None,
    trajectories_b: Sequence[Trajectory] | None,
) -> dict[str, Any]:
    categories = [category.value for category in FailureCategory]
    extra = sorted((set(a.failure_totals) | set(b.failure_totals)) - set(categories))
    names = categories + extra
    totals = {
        name: {
            "baseline": a.failure_totals.get(name, 0),
            "candidate": b.failure_totals.get(name, 0),
            "delta": b.failure_totals.get(name, 0) - a.failure_totals.get(name, 0),
        }
        for name in names
    }
    shift: dict[str, Any] = {
        "categories": names,
        "totals": totals,
        "paired": False,
        "matrix": None,
        "paired_trials": 0,
        "unpaired_trials": 0,
    }
    if trajectories_a is None or trajectories_b is None:
        return shift
    index_a = {(t.task_id, t.trial): t for t in trajectories_a}
    index_b = {(t.task_id, t.trial): t for t in trajectories_b}
    shared = sorted(set(index_a) & set(index_b))
    matrix: dict[str, dict[str, int]] = {row: dict.fromkeys(names, 0) for row in names}
    for key in shared:
        matrix[index_a[key].failure.value][index_b[key].failure.value] += 1
    shift.update(
        paired=True,
        matrix=matrix,
        paired_trials=len(shared),
        unpaired_trials=len(set(index_a) ^ set(index_b)),
    )
    return shift


def diff(
    a: RunReport,
    b: RunReport,
    k: int | None = None,
    *,
    trajectories_a: Sequence[Trajectory] | None = None,
    trajectories_b: Sequence[Trajectory] | None = None,
) -> dict[str, Any]:
    """Compare candidate run ``b`` with baseline run ``a``.

    Tasks are matched by ``task_id``. Tasks present in only one run are listed and left out
    of the comparison.

    Args:
        a: Baseline run report.
        b: Candidate run report.
        k: The k to rank tasks by. Defaults to the largest k that both configs list.
        trajectories_a: Baseline trajectories, for a paired failure matrix.
        trajectories_b: Candidate trajectories, for a paired failure matrix.

    Returns:
        A JSON-serialisable dict with keys ``k``, ``k_values``, ``synthetic``,
        ``baseline``, ``candidate``, ``only_in_baseline``, ``only_in_candidate``,
        ``pass_hat_k`` (per k: baseline, candidate, delta), ``tasks`` (sorted by
        ascending delta, so the largest regression is first) and ``failure_shift``.

    Raises:
        ValueError: If the runs share no k value, ``k`` is not shared, the runs share
            no task ids, or a shared task lacks pass^k at ``k``.
    """
    shared_k = _common_k(a, b)
    if not shared_k:
        raise ValueError("the two runs share no k value in their configs")
    if k is None:
        k = shared_k[-1]
    elif k not in shared_k:
        raise ValueError(f"k={k} is not in both runs; shared k values are {shared_k}")

    stats_a = {t.task_id: t for t in a.tasks}
    stats_b = {t.task_id: t for t in b.tasks}
    shared_tasks = sorted(set(stats_a) & set(stats_b))
    if not shared_tasks:
        raise ValueError("the two runs share no task ids")

    rows: list[dict[str, Any]] = []
    for task_id in shared_tasks:
        ta, tb = stats_a[task_id], stats_b[task_id]
        if k not in ta.pass_hat_k or k not in tb.pass_hat_k:
            raise ValueError(f"task {task_id} has no pass^{k} in one of the runs")
        base_value, cand_value = ta.pass_hat_k[k], tb.pass_hat_k[k]
        rows.append(
            {
                "task_id": task_id,
                "tier": tb.tier,
                "c_baseline": ta.c,
                "c_candidate": tb.c,
                "n_baseline": ta.n,
                "n_candidate": tb.n,
                "pass_hat_k_baseline": base_value,
                "pass_hat_k_candidate": cand_value,
                "delta": cand_value - base_value,
            }
        )
    # Rounding the key makes float ties (for example -0.6 computed two ways) sort by task id.
    rows.sort(key=lambda row: (round(row["delta"], 12), row["task_id"]))

    per_k: dict[int, dict[str, float]] = {}
    for kk in shared_k:
        base_mean = _mean_over(shared_tasks, stats_a, kk)
        cand_mean = _mean_over(shared_tasks, stats_b, kk)
        if base_mean is None or cand_mean is None:
            continue
        per_k[kk] = {
            "baseline": base_mean,
            "candidate": cand_mean,
            "delta": cand_mean - base_mean,
        }

    return {
        "k": k,
        "k_values": shared_k,
        "synthetic": is_synthetic(a.env_info) or is_synthetic(b.env_info),
        "baseline": {"run_id": a.run_id, "synthetic": is_synthetic(a.env_info)},
        "candidate": {"run_id": b.run_id, "synthetic": is_synthetic(b.env_info)},
        "only_in_baseline": sorted(set(stats_a) - set(stats_b)),
        "only_in_candidate": sorted(set(stats_b) - set(stats_a)),
        "pass_hat_k": per_k,
        "tasks": rows,
        "failure_shift": _failure_shift(a, b, trajectories_a, trajectories_b),
    }


def _mean_over(task_ids: Sequence[str], stats: dict[str, Any], k: int) -> float | None:
    values = [
        stats[task_id].pass_hat_k[k] for task_id in task_ids if k in stats[task_id].pass_hat_k
    ]
    if len(values) != len(task_ids):
        return None
    return sum(values) / len(values)
