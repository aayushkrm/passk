"""Tests for pass@k, pass^k, per-task aggregation and bootstrap intervals."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from passk.metrics import (
    aggregate_task,
    bootstrap_ci,
    paired_bootstrap,
    pass_at_k,
    pass_hat_k,
    summarize_tasks,
)
from passk.schema import (
    AgentAction,
    FailureCategory,
    Step,
    TaskStats,
    Trajectory,
)


def _trajectory(task_id: str, trial: int, success: bool) -> Trajectory:
    """Build a minimal trajectory with a consistent success/failure pair."""
    return Trajectory(
        run_id="r",
        task_id=task_id,
        trial=trial,
        seed=trial,
        agent_id="test",
        steps=[Step(index=0, action=AgentAction(kind="final", message="done"), latency_s=0.0)],
        final_state={},
        success=success,
        violations=[],
        failure=FailureCategory.SUCCESS if success else FailureCategory.WRONG_STATE,
        wall_s=0.0,
    )


# Exact values -----------------------------------------------------------------------


def test_pass_hat_k_known_value() -> None:
    assert pass_hat_k(5, 3, 2) == 0.3
    assert pass_hat_k(5, 3, 2) == math.comb(3, 2) / math.comb(5, 2)


def test_pass_at_k_known_value() -> None:
    assert pass_at_k(5, 3, 2) == pytest.approx(0.9, abs=0)
    assert pass_at_k(5, 3, 2) == 1 - math.comb(2, 2) / math.comb(5, 2)


def test_pass_hat_k_sequence_for_five_trials_three_successes() -> None:
    values = [pass_hat_k(5, 3, k) for k in range(1, 6)]
    assert values == pytest.approx([0.6, 0.3, 0.1, 0.0, 0.0], abs=0)


def test_no_successes_gives_zero_pass_at_k() -> None:
    for k in range(1, 6):
        assert pass_at_k(5, 0, k) == 0.0
        assert pass_hat_k(5, 0, k) == 0.0


def test_all_successes_gives_one() -> None:
    for k in range(1, 6):
        assert pass_hat_k(5, 5, k) == 1.0
        assert pass_at_k(5, 5, k) == 1.0


def test_k_equal_one_is_success_rate() -> None:
    assert pass_at_k(7, 4, 1) == pytest.approx(4 / 7, abs=0)
    assert pass_hat_k(7, 4, 1) == pytest.approx(4 / 7, abs=0)


# Monotonicity -----------------------------------------------------------------------


def test_monotonicity_over_grid() -> None:
    for n in range(1, 13):
        for c in range(0, n + 1):
            hat = [pass_hat_k(n, c, k) for k in range(1, n + 1)]
            at = [pass_at_k(n, c, k) for k in range(1, n + 1)]
            assert all(0.0 <= v <= 1.0 for v in hat + at)
            assert all(later <= earlier for earlier, later in zip(hat, hat[1:], strict=False))
            assert all(later >= earlier for earlier, later in zip(at, at[1:], strict=False))


# Argument validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    "args",
    [(5, 3, 6), (5, 6, 2), (5, -1, 2), (5, 3, 0), (0, 0, 1)],
)
def test_out_of_range_arguments_raise(args: tuple[int, int, int]) -> None:
    with pytest.raises(ValueError):
        pass_hat_k(*args)
    with pytest.raises(ValueError):
        pass_at_k(*args)


# Brute-force enumeration -----------------------------------------------------------


def _enumerate_pass_hat(n: int, c: int, k: int) -> float:
    """Fraction of k-subsets of a fixed outcome vector in which every trial succeeds."""
    outcomes = [1] * c + [0] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    return sum(all(outcomes[i] for i in s) for s in subsets) / len(subsets)


def _enumerate_pass_at(n: int, c: int, k: int) -> float:
    """Fraction of k-subsets of a fixed outcome vector with at least one success."""
    outcomes = [1] * c + [0] * (n - c)
    subsets = list(itertools.combinations(range(n), k))
    return sum(any(outcomes[i] for i in s) for s in subsets) / len(subsets)


@pytest.mark.parametrize("n", range(1, 8))
def test_estimators_match_brute_force_enumeration(n: int) -> None:
    for c in range(n + 1):
        for k in range(1, n + 1):
            assert pass_hat_k(n, c, k) == pytest.approx(_enumerate_pass_hat(n, c, k), abs=1e-12)
            assert pass_at_k(n, c, k) == pytest.approx(_enumerate_pass_at(n, c, k), abs=1e-12)


def test_estimators_match_seeded_monte_carlo() -> None:
    """Draw k trials without replacement at random and compare the hit rates."""
    n, c, k, draws = 10, 6, 3, 200_000
    rng = np.random.default_rng(12345)
    outcomes = np.array([1] * c + [0] * (n - c), dtype=bool)
    picks = np.argsort(rng.random((draws, n)), axis=1)[:, :k]
    hits = outcomes[picks]
    assert float(hits.all(axis=1).mean()) == pytest.approx(pass_hat_k(n, c, k), abs=0.005)
    assert float(hits.any(axis=1).mean()) == pytest.approx(pass_at_k(n, c, k), abs=0.005)


# Per-task aggregation ---------------------------------------------------------------


def test_aggregate_task_counts_and_skips_k_above_n() -> None:
    trials = [_trajectory("t1", i, success=i < 3) for i in range(5)]
    stats = aggregate_task("t1", 2, trials, k_values=[1, 2, 5, 6])
    assert stats.n == 5
    assert stats.c == 3
    assert stats.tier == 2
    assert set(stats.pass_hat_k) == {1, 2, 5}
    assert set(stats.pass_at_k) == {1, 2, 5}
    assert stats.pass_hat_k[2] == pytest.approx(0.3, abs=0)
    assert stats.failures == {"wrong_state": 2}


def test_aggregate_task_counts_failure_categories_of_failed_trials_only() -> None:
    trials = [_trajectory("t1", 0, True), _trajectory("t1", 1, False)]
    trials[1] = trials[1].model_copy(update={"failure": FailureCategory.LOOP})
    stats = aggregate_task("t1", 1, trials, k_values=[1])
    assert stats.failures == {"loop": 1}


def test_aggregate_task_rejects_mixed_tasks_and_inconsistent_labels() -> None:
    with pytest.raises(ValueError):
        aggregate_task("t1", 1, [_trajectory("t2", 0, True)], k_values=[1])
    bad = _trajectory("t1", 0, True).model_copy(update={"failure": FailureCategory.LOOP})
    with pytest.raises(ValueError):
        aggregate_task("t1", 1, [bad], k_values=[1])
    with pytest.raises(ValueError):
        aggregate_task("t1", 1, [], k_values=[1])


# Bootstrap --------------------------------------------------------------------------


def test_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    values = [0.1, 0.4, 0.5, 0.9, 1.0, 0.0, 0.7]
    assert bootstrap_ci(values, seed=7) == bootstrap_ci(values, seed=7)


def test_bootstrap_does_not_touch_global_random_state() -> None:
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    bootstrap_ci([0.2, 0.8, 0.5], n_boot=200, seed=3)
    after = np.random.get_state()[1]
    assert np.array_equal(before, after)


def test_bootstrap_mean_and_interval_order() -> None:
    values = [0.2, 0.4, 0.6, 0.8]
    mean, low, high = bootstrap_ci(values, seed=1)
    assert mean == pytest.approx(0.5, abs=0)
    assert low <= mean <= high


def test_bootstrap_constant_values_collapse() -> None:
    assert bootstrap_ci([0.25] * 6, seed=0) == (0.25, 0.25, 0.25)


def test_bootstrap_width_close_to_normal_approximation() -> None:
    rng = np.random.default_rng(99)
    values = rng.normal(0.0, 1.0, size=400).tolist()
    mean, low, high = bootstrap_ci(values, n_boot=10_000, seed=5)
    se = float(np.std(values, ddof=1)) / math.sqrt(len(values))
    assert (high - low) == pytest.approx(2 * 1.96 * se, rel=0.1)
    assert low < mean < high


def test_bootstrap_rejects_empty_and_bad_arguments() -> None:
    with pytest.raises(ValueError):
        bootstrap_ci([], seed=0)
    with pytest.raises(ValueError):
        bootstrap_ci([0.5], n_boot=0)
    with pytest.raises(ValueError):
        bootstrap_ci([0.5], alpha=1.0)


def test_paired_bootstrap_delta_and_sign() -> None:
    a = [0.2, 0.4, 0.5, 0.3, 0.6]
    b = [x + 0.25 for x in a]
    delta, low, high = paired_bootstrap(a, b, seed=2)
    assert delta == pytest.approx(0.25, abs=1e-12)
    assert low == pytest.approx(0.25, abs=1e-12)
    assert high == pytest.approx(0.25, abs=1e-12)


def test_paired_bootstrap_interval_excludes_zero_for_a_clear_shift() -> None:
    rng = np.random.default_rng(4)
    a = rng.uniform(0.2, 0.8, size=30)
    b = a + 0.1 + rng.normal(0.0, 0.02, size=30)
    delta, low, high = paired_bootstrap(a.tolist(), b.tolist(), seed=0)
    assert delta > 0
    assert low > 0


def test_paired_bootstrap_is_reproducible_and_checks_lengths() -> None:
    a = [0.1, 0.5, 0.9]
    b = [0.2, 0.4, 0.8]
    assert paired_bootstrap(a, b, seed=11) == paired_bootstrap(a, b, seed=11)
    with pytest.raises(ValueError):
        paired_bootstrap([0.1, 0.2], [0.1])


# Summary ----------------------------------------------------------------------------


def _task(task_id: str, n: int, c: int, ks: list[int]) -> TaskStats:
    return TaskStats(
        task_id=task_id,
        tier=1,
        n=n,
        c=c,
        pass_at_k={k: pass_at_k(n, c, k) for k in ks if k <= n},
        pass_hat_k={k: pass_hat_k(n, c, k) for k in ks if k <= n},
        failures={},
    )


def test_summarize_tasks_shape_and_means() -> None:
    tasks = [_task("a", 5, 5, [1, 2]), _task("b", 5, 0, [1, 2])]
    summary = summarize_tasks(tasks, [1, 2], n_boot=500, seed=0)
    assert set(summary) == {"pass_at_k", "pass_hat_k"}
    assert set(summary["pass_hat_k"]) == {1, 2}
    assert summary["pass_hat_k"][1]["mean"] == pytest.approx(0.5, abs=0)
    assert summary["pass_hat_k"][2]["mean"] == pytest.approx(0.5, abs=0)
    entry = summary["pass_at_k"][1]
    assert set(entry) == {"mean", "ci_low", "ci_high"}
    assert entry["ci_low"] <= entry["mean"] <= entry["ci_high"]


def test_summarize_tasks_only_uses_tasks_that_support_k() -> None:
    tasks = [_task("a", 5, 5, [1, 2, 3]), _task("b", 2, 1, [1, 2])]
    summary = summarize_tasks(tasks, [1, 3], n_boot=200, seed=0)
    assert set(summary["pass_hat_k"]) == {1, 3}
    # Only task "a" has n >= 3, so the mean for k=3 is that task's value.
    assert summary["pass_hat_k"][3]["mean"] == pytest.approx(1.0, abs=0)


def test_summarize_tasks_omits_unsupported_k() -> None:
    tasks = [_task("a", 2, 1, [1])]
    summary = summarize_tasks(tasks, [1, 4], n_boot=100, seed=0)
    assert set(summary["pass_hat_k"]) == {1}


def test_summarize_tasks_is_reproducible() -> None:
    tasks = [_task("a", 5, 3, [1, 2]), _task("b", 5, 1, [1, 2]), _task("c", 5, 4, [1, 2])]
    assert summarize_tasks(tasks, [1, 2], seed=8) == summarize_tasks(tasks, [1, 2], seed=8)
