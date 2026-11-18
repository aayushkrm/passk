"""Reliability metrics: pass@k, pass^k, per-task aggregation and bootstrap intervals.

Definitions (n trials per task, c successes):

- ``pass_at_k(n, c, k) = 1 - C(n-c, k) / C(n, k)``: chance that at least one of k
  trials drawn without replacement succeeds.
- ``pass_hat_k(n, c, k) = C(c, k) / C(n, k)``: the unbiased pass^k estimator from
  tau-bench. It is the chance that all k trials drawn without replacement succeed.

Both estimators are computed from exact integers and rounded once by Python's int
division, so they are monotone in k in floating point as well as on paper.

Intervals are percentile bootstraps over tasks. Every resampling call takes an explicit
seed and creates its own ``numpy.random.Generator``; no global random state is touched.
"""

import math
from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np

from passk.schema import FailureCategory, TaskStats, Trajectory

_BOOTSTRAP_CHUNK = 1_000


def _check_nck(n: int, c: int, k: int) -> None:
    """Validate the ``(n, c, k)`` triple shared by both estimators.

    Args:
        n: Number of trials.
        c: Number of successful trials.
        k: Number of trials drawn.

    Raises:
        ValueError: If ``n < 1``, ``c`` is outside ``[0, n]`` or ``k`` is outside ``[1, n]``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must satisfy 0 <= c <= n, got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n, got k={k}, n={n}")


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of the chance that at least one of k trials succeeds.

    Args:
        n: Number of trials.
        c: Number of successful trials, ``0 <= c <= n``.
        k: Number of trials drawn, ``1 <= k <= n``.

    Returns:
        ``1 - C(n-c, k) / C(n, k)``. For example ``pass_at_k(5, 0, k) == 0``.

    Raises:
        ValueError: If the arguments are out of range (see ``_check_nck``).
    """
    _check_nck(n, c, k)
    total = math.comb(n, k)
    return (total - math.comb(n - c, k)) / total


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of the chance that all of k trials succeed.

    Args:
        n: Number of trials.
        c: Number of successful trials, ``0 <= c <= n``.
        k: Number of trials drawn, ``1 <= k <= n``.

    Returns:
        ``C(c, k) / C(n, k)``. For example ``pass_hat_k(5, 3, 2) == 0.3``.

    Raises:
        ValueError: If the arguments are out of range (see ``_check_nck``).
    """
    _check_nck(n, c, k)
    return math.comb(c, k) / math.comb(n, k)


def aggregate_task(
    task_id: str,
    tier: Literal[1, 2, 3],
    trials: Sequence[Trajectory],
    k_values: Sequence[int],
) -> TaskStats:
    """Aggregate the trials of one task into a ``TaskStats``.

    Only values of k with ``k <= n`` are reported, because pass@k and pass^k are undefined
    for k above the number of trials. ``failures`` counts the failure categories of the
    unsuccessful trials only; successful trials are not counted there.

    Args:
        task_id: Identifier of the task. Every trajectory must carry this id.
        tier: Tier of the task, copied into the result.
        trials: The trajectories of this task, one per trial.
        k_values: Values of k to report. Values greater than ``n`` are skipped.

    Returns:
        A ``TaskStats`` with ``n = len(trials)`` and ``c`` successful trials.

    Raises:
        ValueError: If ``trials`` is empty, a trajectory belongs to another task, a
            trajectory's ``success`` flag disagrees with its ``failure`` category, or a
            value in ``k_values`` is below 1.
    """
    if not trials:
        raise ValueError(f"task {task_id!r} has no trials")
    if any(k < 1 for k in k_values):
        raise ValueError(f"k values must be >= 1, got {list(k_values)}")

    failures: Counter[str] = Counter()
    for trial in trials:
        if trial.task_id != task_id:
            raise ValueError(f"trajectory for task {trial.task_id!r} passed for {task_id!r}")
        if trial.success != (trial.failure == FailureCategory.SUCCESS):
            raise ValueError(
                f"trajectory {trial.run_id}/{trial.task_id}/{trial.trial}: success="
                f"{trial.success} disagrees with failure={trial.failure.value}"
            )
        if not trial.success:
            failures[trial.failure.value] += 1

    n = len(trials)
    c = sum(1 for trial in trials if trial.success)
    ks = sorted({k for k in k_values if k <= n})
    return TaskStats(
        task_id=task_id,
        tier=tier,
        n=n,
        c=c,
        pass_at_k={k: pass_at_k(n, c, k) for k in ks},
        pass_hat_k={k: pass_hat_k(n, c, k) for k in ks},
        failures=dict(failures),
    )


def _mean_ci(
    values: Sequence[float], n_boot: int, alpha: float, seed: int
) -> tuple[float, float, float]:
    """Percentile bootstrap over the entries of ``values`` for their mean.

    Args:
        values: Non-empty sequence of per-task values.
        n_boot: Number of bootstrap resamples, at least 1.
        alpha: Two-sided significance level, in ``(0, 1)``.
        seed: Seed for the resampling generator.

    Returns:
        ``(mean, ci_low, ci_high)`` where the interval is the ``alpha/2`` and
        ``1 - alpha/2`` quantiles of the resampled means.

    Raises:
        ValueError: If ``values`` is empty or the other arguments are out of range.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=float)
    for start in range(0, n_boot, _BOOTSTRAP_CHUNK):
        stop = min(start + _BOOTSTRAP_CHUNK, n_boot)
        idx = rng.integers(0, arr.size, size=(stop - start, arr.size))
        boot_means[start:stop] = arr[idx].mean(axis=1)
    low, high = np.quantile(boot_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(arr.mean()), float(low), float(high)


def bootstrap_ci(
    values: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap over tasks for the mean of ``values``.

    Args:
        values: Per-task metric values.
        n_boot: Number of bootstrap resamples.
        alpha: Two-sided significance level (0.05 gives a 95% interval).
        seed: Seed for the resampling RNG. The result must be reproducible.

    Returns:
        ``(mean, ci_low, ci_high)``.

    Raises:
        ValueError: If ``values`` is empty or the other arguments are out of range.
    """
    return _mean_ci(values, n_boot, alpha, seed)


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap over tasks for the mean difference ``b - a``.

    The same task indices are resampled for both runs, so the interval reflects the
    per-task differences rather than two independent samples.

    Args:
        a: Per-task metric values for the baseline run.
        b: Per-task metric values for the candidate run, aligned with ``a``.
        n_boot: Number of bootstrap resamples.
        alpha: Two-sided significance level.
        seed: Seed for the resampling RNG.

    Returns:
        ``(mean_delta, ci_low, ci_high)`` for ``b - a``.

    Raises:
        ValueError: If ``a`` and ``b`` differ in length, or are empty.
    """
    if len(a) != len(b):
        raise ValueError(f"paired sequences differ in length: {len(a)} != {len(b)}")
    diffs = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return _mean_ci(diffs.tolist(), n_boot, alpha, seed)


def summarize_tasks(
    tasks: Sequence[TaskStats],
    k_values: Sequence[int],
    *,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[int, dict[str, float]]]:
    """Build the ``RunReport.summary`` mapping across tasks.

    For each k, the mean of the per-task pass@k and pass^k is reported with a seeded
    percentile bootstrap over tasks. Only tasks with ``n >= k`` contribute to that k.
    The same seed is used for every k and metric, so their resamples use the same task
    indices and the numbers are comparable across k.

    Args:
        tasks: Per-task statistics.
        k_values: Values of k to summarise.
        n_boot: Number of bootstrap resamples.
        alpha: Two-sided significance level.
        seed: Seed for the resampling RNG.

    Returns:
        ``{"pass_at_k": {k: {"mean", "ci_low", "ci_high"}}, "pass_hat_k": {...}}``.
        A k that no task can support (every ``n < k``) is omitted.

    Raises:
        ValueError: If a value in ``k_values`` is below 1 or the bootstrap arguments are
            out of range.
    """
    if any(k < 1 for k in k_values):
        raise ValueError(f"k values must be >= 1, got {list(k_values)}")

    summary: dict[str, dict[int, dict[str, float]]] = {"pass_at_k": {}, "pass_hat_k": {}}
    for k in sorted(set(k_values)):
        eligible = [task for task in tasks if task.n >= k]
        if not eligible:
            continue
        for metric, per_task in (
            ("pass_at_k", [task.pass_at_k[k] for task in eligible]),
            ("pass_hat_k", [task.pass_hat_k[k] for task in eligible]),
        ):
            mean, ci_low, ci_high = _mean_ci(per_task, n_boot, alpha, seed)
            summary[metric][k] = {"mean": mean, "ci_low": ci_low, "ci_high": ci_high}
    return summary
