"""Compare two runs: per-task pass^k deltas and a failure-category shift matrix."""

from typing import Any

from passk.schema import RunReport


def diff(a: RunReport, b: RunReport, k: int | None = None) -> dict[str, Any]:
    """Compare run ``b`` (candidate) with run ``a`` (baseline).

    Args:
        a: Baseline run report.
        b: Candidate run report. Tasks are matched by ``task_id``.
        k: The k to compare. Defaults to the largest k in both configs.

    Returns:
        A JSON-serialisable dict with per-task pass^k deltas and the failure-category
        shift matrix (baseline category -> candidate category counts).
    """
    raise NotImplementedError("diff: scheduled for work package P-E")
