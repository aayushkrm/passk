"""CI regression gate.

A candidate fails if pass^k@K drops by more than ``max_drop`` AND the paired-bootstrap
CI for the drop excludes 0 (when ``require_significance`` is set), or if the candidate's
mean pass^k@K falls below the absolute floor ``min_pass_hat_k``.

The paired bootstrap resamples tasks. For each task it pairs the baseline and candidate
pass^k@K values, using ``passk.metrics.paired_bootstrap`` with a fixed seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from passk import metrics
from passk.numfmt import fixed
from passk.schema import GateDecision, GatePolicy, RunReport, TaskStats

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
POLICY_KEYS = frozenset({"k", "max_drop", "min_pass_hat_k", "require_significance", "alpha"})


def f3(value: float, *, signed: bool = False) -> str:
    """Format a value with three decimals, stable across platforms."""
    return fixed(value, 3, signed=signed)


def load_policy(path: Path) -> GatePolicy:
    """Load a gate policy from a YAML mapping.

    The file holds the ``GatePolicy`` fields at top level: ``k``, ``max_drop``,
    ``min_pass_hat_k`` (optional), ``require_significance`` (default true) and ``alpha``
    (default 0.05). Unknown keys are rejected so that typos fail loudly.

    Args:
        path: Path to the YAML file.

    Returns:
        The validated policy.

    Raises:
        ValueError: If the file is not a mapping, has unknown keys, or fails validation.
    """
    data: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: policy must be a YAML mapping")
    unknown = sorted(str(key) for key in set(data) - POLICY_KEYS)
    if unknown:
        raise ValueError(f"{path}: unknown policy keys {unknown}; allowed {sorted(POLICY_KEYS)}")
    try:
        policy = GatePolicy.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{path}: invalid policy: {exc}") from exc
    _check_policy(policy)
    return policy


def _check_policy(policy: GatePolicy) -> None:
    if policy.k < 1:
        raise ValueError("policy k must be at least 1")
    if policy.max_drop < 0:
        raise ValueError("policy max_drop must be non-negative")
    if not 0.0 < policy.alpha < 1.0:
        raise ValueError("policy alpha must lie strictly between 0 and 1")


def _paired_ci(a: Sequence[float], b: Sequence[float], alpha: float) -> tuple[float, float, float]:
    """Return ``(mean_delta, ci_low, ci_high)`` for ``b - a`` with a seeded paired bootstrap."""
    return metrics.paired_bootstrap(
        a, b, n_boot=BOOTSTRAP_RESAMPLES, alpha=alpha, seed=BOOTSTRAP_SEED
    )


def _values_at_k(stats: Sequence[TaskStats], k: int, label: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for task in stats:
        if k not in task.pass_hat_k:
            raise ValueError(f"{label} task {task.task_id} has no pass^{k} (n={task.n})")
        values[task.task_id] = task.pass_hat_k[k]
    return values


def gate(baseline: RunReport, candidate: RunReport, policy: GatePolicy) -> GateDecision:
    """Decide whether ``candidate`` regresses against ``baseline`` under ``policy``.

    Args:
        baseline: Reference run report.
        candidate: New run report.
        policy: Gate thresholds.

    Returns:
        The decision with the paired delta (candidate minus baseline) and its CI.

    Raises:
        ValueError: If the two runs do not share the same task ids, a task lacks pass^k at
            ``policy.k``, the run has no tasks, or the policy is invalid.
    """
    _check_policy(policy)
    k = policy.k
    base = _values_at_k(baseline.tasks, k, "baseline")
    cand = _values_at_k(candidate.tasks, k, "candidate")
    if not base:
        raise ValueError("baseline report has no tasks")
    if set(base) != set(cand):
        only_base = sorted(set(base) - set(cand))
        only_cand = sorted(set(cand) - set(base))
        raise ValueError(
            f"task ids differ: only in baseline {only_base}, only in candidate {only_cand}"
        )

    base_n = {s.task_id: s.n for s in baseline.tasks}
    cand_n = {s.task_id: s.n for s in candidate.tasks}
    if base_n != cand_n:
        raise ValueError("baseline and candidate ran different trial counts per task")
    ids = sorted(base)
    a = [base[task_id] for task_id in ids]
    b = [cand[task_id] for task_id in ids]
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    delta, ci_low, ci_high = _paired_ci(a, b, policy.alpha)

    level = f"{(1.0 - policy.alpha) * 100:g}%"
    drop = 0.0 - delta  # written this way so a zero delta never prints as -0.000
    reasons = [
        f"pass^{k}: baseline {f3(mean_a)}, candidate {f3(mean_b)}, "
        f"delta {f3(delta, signed=True)} "
        f"({level} CI [{f3(ci_low, signed=True)}, {f3(ci_high, signed=True)}], {len(ids)} tasks)"
    ]
    failed = False
    if drop > policy.max_drop:
        significant = ci_high < 0.0
        if not policy.require_significance:
            failed = True
            reasons.append(
                f"regression: drop {f3(drop)} exceeds max_drop {f3(policy.max_drop)} "
                "(significance not required)"
            )
        elif significant:
            failed = True
            reasons.append(
                f"regression: drop {f3(drop)} exceeds max_drop {f3(policy.max_drop)} "
                f"and the {level} CI excludes 0"
            )
        else:
            reasons.append(
                f"not a regression: drop {f3(drop)} exceeds max_drop "
                f"{f3(policy.max_drop)} but the {level} CI includes 0"
            )
    else:
        reasons.append(f"drop {f3(drop)} is within max_drop {f3(policy.max_drop)}")

    if policy.min_pass_hat_k is not None:
        floor = policy.min_pass_hat_k
        if mean_b < floor:
            failed = True
            reasons.append(
                f"floor: candidate pass^{k} {f3(mean_b)} is below min_pass_hat_k {f3(floor)}"
            )
        else:
            reasons.append(
                f"floor: candidate pass^{k} {f3(mean_b)} meets min_pass_hat_k {f3(floor)}"
            )

    return GateDecision(
        passed=not failed,
        reasons=reasons,
        delta=float(delta),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
    )
