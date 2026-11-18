"""Reliability metrics: pass@k, pass^k and bootstrap intervals.

Definitions (n trials per task, c successes):
- ``pass_at_k(n, c, k) = 1 - C(n-c, k) / C(n, k)``
- ``pass_hat_k(n, c, k) = C(c, k) / C(n, k)``, the unbiased pass^k estimator from tau-bench.
"""

from collections.abc import Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of the chance that at least one of k trials succeeds.

    Args:
        n: Number of trials.
        c: Number of successful trials, ``0 <= c <= n``.
        k: Number of trials drawn, ``1 <= k <= n``.

    Returns:
        ``1 - C(n-c, k) / C(n, k)``.
    """
    raise NotImplementedError("pass_at_k: scheduled for work package P-A")


def pass_hat_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of the chance that all of k trials succeed.

    Args:
        n: Number of trials.
        c: Number of successful trials, ``0 <= c <= n``.
        k: Number of trials drawn, ``1 <= k <= n``.

    Returns:
        ``C(c, k) / C(n, k)``. For example ``pass_hat_k(5, 3, 2) == 0.3``.
    """
    raise NotImplementedError("pass_hat_k: scheduled for work package P-A")


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
    """
    raise NotImplementedError("bootstrap_ci: scheduled for work package P-A")


def paired_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Paired bootstrap over tasks for the mean difference ``b - a``.

    Args:
        a: Per-task metric values for the baseline run.
        b: Per-task metric values for the candidate run, aligned with ``a``.
        n_boot: Number of bootstrap resamples.
        alpha: Two-sided significance level.
        seed: Seed for the resampling RNG.

    Returns:
        ``(mean_delta, ci_low, ci_high)`` for ``b - a``.

    Raises:
        ValueError: If ``a`` and ``b`` differ in length.
    """
    raise NotImplementedError("paired_bootstrap: scheduled for work package P-A")
