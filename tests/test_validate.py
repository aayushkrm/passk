"""Calibration logic (E1 and E2) against a fast analytic fake simulator. No runner needed."""

from __future__ import annotations

import numpy as np
import pytest

from passk import validate
from passk.schema import (
    AgentConfig,
    Assertion,
    RunConfig,
    RunReport,
    Task,
    TaskStats,
    ToolCall,
)
from passk.validate import (
    analytic_pass_hat_k,
    check_estimator,
    pass_hat_from_counts,
    run_validation,
    wilson_interval,
)

LENGTHS = {"t1-001": 1, "t1-002": 1, "t2-001": 2, "t2-002": 2, "t3-001": 3, "t3-002": 3}


def _task(task_id: str, steps: int) -> Task:
    call = ToolCall(call_id="c", name="get_compound", arguments={})
    return Task(
        id=task_id,
        tier=1,
        instruction="x",
        initial_state={},
        goal=[Assertion(path="a", op="equals", value=1)],
        forbidden=[],
        reference_solution=[call] * steps,
    )


TASKS = [_task(tid, m) for tid, m in LENGTHS.items()]


def _fake_report(p: float, seed: int, n: int = validate.N_TRIALS) -> RunReport:
    """Fast stand-in for the runner: binomial trial counts with success q_t = (1-p)^m_t."""
    rng = np.random.default_rng(seed)
    stats: list[TaskStats] = []
    for task in TASKS:
        q = (1.0 - p) ** len(task.reference_solution)
        c = int(rng.binomial(n, q))
        stats.append(
            TaskStats(
                task_id=task.id,
                tier=task.tier,
                n=n,
                c=c,
                pass_at_k={},
                pass_hat_k={k: pass_hat_from_counts(n, c, k) for k in range(1, 6)},
                failures={},
            )
        )
    return RunReport(
        run_id=f"fake-p{p}-s{seed}",
        config=RunConfig(
            run_id=f"fake-p{p}-s{seed}",
            tasks_glob="unused",
            agent=AgentConfig(kind="noisy_oracle", params={"p_error": p, "seed": seed}),
            n_trials=n,
            base_seed=seed,
            transport="inprocess",
        ),
        env_info={"synthetic": True},
        tasks=stats,
        summary={},
        failure_totals={},
        tokens={},
        wall_s=0.0,
    )


def test_pass_hat_from_counts_matches_definition() -> None:
    assert pass_hat_from_counts(5, 3, 2) == pytest.approx(0.3)
    assert pass_hat_from_counts(5, 2, 3) == 0.0
    assert pass_hat_from_counts(20, 20, 5) == pytest.approx(1.0)


def test_analytic_pass_hat_k_is_mean_of_q_t_to_the_k() -> None:
    # q_t = (1 - 0.1) ** m_t; pass^k = mean_t q_t ** k
    expected = np.mean([(0.9**m) ** 2 for m in LENGTHS.values()])
    assert analytic_pass_hat_k(TASKS, 0.1, 2) == pytest.approx(float(expected))
    assert analytic_pass_hat_k(TASKS, 0.0, 5) == pytest.approx(1.0)


def test_analytic_requires_tasks() -> None:
    with pytest.raises(ValueError, match="no tasks"):
        analytic_pass_hat_k([], 0.1, 1)


def test_wilson_interval_bounds() -> None:
    low, high = wilson_interval(0, 200)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < high < 0.05
    low, high = wilson_interval(200, 200)
    assert high == 1.0 and low > 0.95
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_estimator_check_with_exact_counts_has_small_error() -> None:
    result = check_estimator(lambda p, s: _fake_report(p, s), TASKS, 0.0, seed=1)
    # p = 0 means every trial succeeds, so the estimator is exact
    assert result["max_abs_error"] == pytest.approx(0.0)


def test_estimator_check_flags_wrong_trial_count() -> None:
    with pytest.raises(ValueError, match="trials per task"):
        check_estimator(lambda p, s: _fake_report(p, s, n=5), TASKS, 0.1, seed=1)


def test_estimator_check_flags_task_mismatch() -> None:
    with pytest.raises(ValueError, match="task ids differ"):
        check_estimator(lambda p, s: _fake_report(p, s), TASKS[:3], 0.1, seed=1)


def test_estimator_check_flags_stored_value_disagreeing_with_counts() -> None:
    def broken(p: float, seed: int) -> RunReport:
        report = _fake_report(p, seed)
        bad = report.tasks[0].model_copy(update={"pass_hat_k": {1: 0.123}})
        return report.model_copy(update={"tasks": [bad, *report.tasks[1:]]})

    with pytest.raises(ValueError, match="disagrees with counts"):
        check_estimator(broken, TASKS, 0.1, seed=1)


def test_run_validation_reports_e1_and_e2_labelled_synthetic() -> None:
    result = run_validation(lambda p, s: _fake_report(p, s), TASKS, pairs=40, base_seed=3)
    assert result["label"] == "SYNTHETIC"
    e1 = result["e1"]
    assert [level["p"] for level in e1["levels"]] == [0.0, 0.05, 0.10, 0.20]
    assert e1["n_trials"] == 20
    # with 20 trials per task the estimator should track the analytic value closely
    assert e1["max_abs_error"] < 0.15
    e2 = result["e2"]
    assert e2["pairs"] == 40
    assert e2["aa"]["false_alarms"] <= 4
    assert e2["ab"]["power"] > 0.5
    assert e2["ab"]["ci95"][0] <= e2["ab"]["power"] <= e2["ab"]["ci95"][1]


def test_run_validation_is_seeded() -> None:
    first = run_validation(lambda p, s: _fake_report(p, s), TASKS, pairs=10, base_seed=5)
    second = run_validation(lambda p, s: _fake_report(p, s), TASKS, pairs=10, base_seed=5)
    assert first["e1"] == second["e1"]
    assert first["e2"] == second["e2"]


def test_run_validation_rejects_zero_pairs() -> None:
    with pytest.raises(ValueError, match="pairs"):
        run_validation(lambda p, s: _fake_report(p, s), TASKS, pairs=0)
