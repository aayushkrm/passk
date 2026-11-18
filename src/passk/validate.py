"""Calibration of the estimator and the gate (DECISION.md section 2.1, experiments E1 and E2).

Everything here runs on scripted agents, so the output is SYNTHETIC. It tests the tool
against a known ground truth, not a model.

E1 (estimator): a NoisyOracle with error rate p, run for ``N_TRIALS`` (20) trials per task.
Each reference step is corrupted with probability p, so task t succeeds with
q_t = (1 - p) ** m_t, where m_t is the number of reference steps. The true pass^k is
mean_t q_t ** k. The measured pass^k is the unbiased estimator over the trials, averaged
over tasks. The max absolute error is reported for k = 1..5, together with a z-score: the
error divided by its exact sampling standard error under the model. Under the model the
per-task estimator has a known binomial variance, so the standard error of the mean over
tasks is sqrt(sum_t Var_t) / T.

E2 (gate): ``pairs`` seeded A/A comparisons (p = 0.10 against p = 0.10) give the false-alarm
rate. The same number of A/B comparisons (p = 0.05 baseline against p = 0.15 candidate) give
power. Both rates carry Wilson 95% intervals. E2 uses ``E2_TRIALS`` (5) trials per task, the
same n as the main experiment E3, so the gate is calibrated at the operating point it is
used at.

The simulator is injected as ``simulate_run(p, seed) -> RunReport``. The CLI supplies a
default that runs the real runner without writing to disk. Tests supply a fast fake.
Simulations can run in worker processes (``workers > 1``); each job carries its own seed,
so the results do not depend on the number of workers.
"""

from __future__ import annotations

import concurrent.futures
import functools
import math
import time
from collections.abc import Callable, Sequence
from typing import Any

from passk import runner
from passk.gate import gate
from passk.schema import AgentConfig, GatePolicy, RunConfig, RunReport, Task

N_TRIALS = 20
E2_TRIALS = 5
K_MAX = 5
NOISE_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
AA_P = 0.10
AB_P: tuple[float, float] = (0.05, 0.15)
DEFAULT_PAIRS = 200
DEFAULT_POLICY = GatePolicy(k=3, max_drop=0.10)
Z_LIMIT = 3.0
"""|z| above this flags an E1 cell as inconsistent with the analytic value (about a
two-sided 0.05 test Bonferroni-corrected over the 20 cells of the E1 table)."""
SYNTHETIC = "SYNTHETIC"
_Z95 = 1.959963984540054

SimulateRun = Callable[[float, int], RunReport]


def pass_hat_from_counts(n: int, c: int, k: int) -> float:
    """Unbiased pass^k from trial counts, computed independently of the runner.

    Args:
        n: Trials.
        c: Successful trials.
        k: Number of trials drawn.

    Returns:
        ``C(c, k) / C(n, k)``, which is 0 when ``k > c``.
    """
    return math.comb(c, k) / math.comb(n, k)


def analytic_pass_hat_k(tasks: Sequence[Task], p: float, k: int) -> float:
    """True pass^k for NoisyOracle at error rate ``p``.

    Args:
        tasks: Tasks; ``len(task.reference_solution)`` is m_t.
        p: Per-step error probability.
        k: The k value.

    Returns:
        ``mean_t (1 - p) ** (m_t * k)``.
    """
    if not tasks:
        raise ValueError("no tasks to compute the analytic value over")
    return sum((1.0 - p) ** (len(t.reference_solution) * k) for t in tasks) / len(tasks)


def estimator_variance(n: int, q: float, k: int) -> float:
    """Exact variance of ``pass_hat_from_counts(n, c, k)`` when ``c ~ Binomial(n, q)``.

    Args:
        n: Trials per task.
        q: Per-trial success probability.
        k: The k value, 1 <= k <= n.

    Returns:
        The variance of the unbiased pass^k estimator for one task.
    """
    mean = 0.0
    second = 0.0
    denom = math.comb(n, k)
    for c in range(k, n + 1):
        weight = math.comb(n, c) * (q**c) * ((1.0 - q) ** (n - c))
        estimate = math.comb(c, k) / denom
        mean += weight * estimate
        second += weight * estimate * estimate
    return max(0.0, second - mean * mean)


def expected_standard_error(tasks: Sequence[Task], p: float, k: int, n: int) -> float:
    """Standard error of the task-averaged pass^k under the NoisyOracle model.

    Args:
        tasks: Tasks; each has q_t = (1 - p) ** m_t.
        p: Per-step error probability.
        k: The k value.
        n: Trials per task.

    Returns:
        ``sqrt(sum_t Var_t) / T``. Zero when every task is deterministic.
    """
    variance = sum(estimator_variance(n, (1.0 - p) ** len(t.reference_solution), k) for t in tasks)
    return math.sqrt(variance) / len(tasks)


def wilson_interval(successes: int, n: int, z: float = _Z95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Args:
        successes: Count of events.
        n: Number of trials.
        z: Normal quantile (1.96 for 95%).

    Returns:
        ``(low, high)``; ``(0.0, 1.0)`` when ``n`` is zero.
    """
    if n == 0:
        return (0.0, 1.0)
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * math.sqrt(phat * (1.0 - phat) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def check_estimator(
    simulate_run: SimulateRun, tasks: Sequence[Task], p: float, seed: int
) -> dict[str, Any]:
    """Run one E1 level and compare measured pass^k with the analytic value.

    Args:
        simulate_run: Injected simulator.
        tasks: Tasks the simulator runs.
        p: Error rate.
        seed: Seed passed to the simulator.

    Returns:
        A dict with ``p``, ``seed``, ``pass_hat_k`` (per k: measured, analytic, abs_error,
        standard_error, z) and ``max_abs_error`` and ``max_abs_z`` over k.

    Raises:
        ValueError: If the simulator does not run ``N_TRIALS`` trials per task, does not
            cover the given tasks, or its stored pass^k disagrees with the counts.
    """
    return _check_report_for_e1(simulate_run(p, seed), tasks, p, seed)


def _check_report_for_e1(
    report: RunReport, tasks: Sequence[Task], p: float, seed: int
) -> dict[str, Any]:
    """Compare one simulated E1 report with the analytic pass^k. See ``check_estimator``."""
    if any(stats.n != N_TRIALS for stats in report.tasks):
        raise ValueError(f"simulator must run {N_TRIALS} trials per task")
    expected_ids = {t.id for t in tasks}
    if {s.task_id for s in report.tasks} != expected_ids:
        raise ValueError("simulator task ids differ from the task set used for E1")

    per_k: dict[int, dict[str, float]] = {}
    for k in range(1, K_MAX + 1):
        measured_values: list[float] = []
        for stats in report.tasks:
            independent = pass_hat_from_counts(stats.n, stats.c, k)
            stored = stats.pass_hat_k.get(k)
            if stored is not None and abs(stored - independent) > 1e-9:
                raise ValueError(
                    f"task {stats.task_id}: stored pass^{k} {stored} disagrees with counts "
                    f"({independent})"
                )
            measured_values.append(independent)
        measured = sum(measured_values) / len(measured_values)
        analytic = analytic_pass_hat_k(tasks, p, k)
        error = abs(measured - analytic)
        se = expected_standard_error(tasks, p, k, N_TRIALS)
        z = error / se if se > 0.0 else 0.0
        per_k[k] = {
            "measured": measured,
            "analytic": analytic,
            "abs_error": error,
            "standard_error": se,
            "z": z,
        }
    return {
        "p": p,
        "seed": seed,
        "pass_hat_k": per_k,
        "max_abs_error": max(v["abs_error"] for v in per_k.values()),
        "max_abs_z": max(v["z"] for v in per_k.values()),
    }


def _pair_seed(base_seed: int, offset: int, index: int, arm: int) -> int:
    return base_seed + offset + 2 * index + arm


def _apply_job(job: tuple[SimulateRun, float, int]) -> RunReport:
    """Run one simulation job. Module-level so that worker processes can import it."""
    simulate, p, seed = job
    return simulate(p, seed)


def _simulate_all(
    simulate: SimulateRun, jobs: Sequence[tuple[float, int]], workers: int
) -> list[RunReport]:
    """Run ``simulate(p, seed)`` for every job, in order, optionally in worker processes."""
    payload = [(simulate, p, seed) for p, seed in jobs]
    if workers <= 1 or len(payload) <= 1:
        return [_apply_job(item) for item in payload]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_apply_job, payload, chunksize=4))


def run_validation(
    simulate_run: SimulateRun,
    tasks: Sequence[Task],
    *,
    policy: GatePolicy = DEFAULT_POLICY,
    pairs: int = DEFAULT_PAIRS,
    base_seed: int = 0,
    e2_simulate: SimulateRun | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Run experiments E1 and E2 with injected simulators.

    Args:
        simulate_run: Function ``(p, seed) -> RunReport`` that runs ``N_TRIALS`` trials per
            task on ``tasks``. Used for E1.
        tasks: The task set, used for the analytic values.
        policy: Gate policy used in E2.
        pairs: Number of A/A and of A/B comparisons.
        base_seed: Offset for every seed in this calibration.
        e2_simulate: Simulator for E2 (``E2_TRIALS`` trials per task). Defaults to
            ``simulate_run``, in which case E2 uses ``N_TRIALS`` trials.
        workers: Worker processes for the simulations. The result does not depend on it.

    Returns:
        A JSON-serialisable dict labelled SYNTHETIC with keys ``e1``, ``e2`` and
        ``runtime_s``.

    Raises:
        ValueError: On an invalid simulator output, an empty task set or ``pairs < 1``.
    """
    if pairs < 1:
        raise ValueError("pairs must be at least 1")
    started = time.perf_counter()
    e2_sim = e2_simulate if e2_simulate is not None else simulate_run

    e1_jobs = [(p, base_seed + index) for index, p in enumerate(NOISE_LEVELS)]
    e1_reports = _simulate_all(simulate_run, e1_jobs, workers)
    levels = [
        _check_report_for_e1(report, tasks, p, seed)
        for (p, seed), report in zip(e1_jobs, e1_reports, strict=True)
    ]
    e1 = {
        "n_trials": N_TRIALS,
        "k_values": list(range(1, K_MAX + 1)),
        "levels": levels,
        "max_abs_error": max(level["max_abs_error"] for level in levels),
        "max_abs_z": max(level["max_abs_z"] for level in levels),
    }

    aa_jobs: list[tuple[float, int]] = []
    for index in range(pairs):
        aa_jobs.append((AA_P, _pair_seed(base_seed, 10_000, index, 0)))
        aa_jobs.append((AA_P, _pair_seed(base_seed, 10_000, index, 1)))
    aa_reports = _simulate_all(e2_sim, aa_jobs, workers)
    aa_failures = sum(
        1
        for index in range(pairs)
        if not gate(aa_reports[2 * index], aa_reports[2 * index + 1], policy).passed
    )

    ab_jobs: list[tuple[float, int]] = []
    for index in range(pairs):
        ab_jobs.append((AB_P[0], _pair_seed(base_seed, 20_000, index, 0)))
        ab_jobs.append((AB_P[1], _pair_seed(base_seed, 20_000, index, 1)))
    ab_reports = _simulate_all(e2_sim, ab_jobs, workers)
    ab_detections = sum(
        1
        for index in range(pairs)
        if not gate(ab_reports[2 * index], ab_reports[2 * index + 1], policy).passed
    )

    aa_low, aa_high = wilson_interval(aa_failures, pairs)
    ab_low, ab_high = wilson_interval(ab_detections, pairs)
    e2 = {
        "policy": policy.model_dump(mode="json"),
        "pairs": pairs,
        "trials_per_task": aa_reports[0].tasks[0].n,
        "aa": {
            "p": AA_P,
            "false_alarms": aa_failures,
            "rate": aa_failures / pairs,
            "ci95": [aa_low, aa_high],
        },
        "ab": {
            "p_baseline": AB_P[0],
            "p_candidate": AB_P[1],
            "detections": ab_detections,
            "power": ab_detections / pairs,
            "ci95": [ab_low, ab_high],
        },
    }
    return {
        "label": SYNTHETIC,
        "note": "Scripted NoisyOracle agents. Calibrates the tool, not a model.",
        "e1": e1,
        "e2": e2,
        "runtime_s": time.perf_counter() - started,
    }


_TASKS_BY_GLOB: dict[str, list[Task]] = {}


def _tasks_for(tasks_glob: str) -> list[Task]:
    """Parse the task files once per process; parsing costs about a second per run otherwise."""
    if tasks_glob not in _TASKS_BY_GLOB:
        from passk.envs.base import load_tasks

        _TASKS_BY_GLOB[tasks_glob] = load_tasks(tasks_glob)
    return _TASKS_BY_GLOB[tasks_glob]


def simulate_labstock(tasks_glob: str, n_trials: int, p: float, seed: int) -> RunReport:
    """Run the real runner on NoisyOracle agents for one (p, seed) and return the report.

    Nothing is written to disk and no environment record is collected: only the report
    is kept. Module-level, so worker processes can call it.

    Args:
        tasks_glob: Glob of the task YAML files.
        n_trials: Trials per task.
        p: Per-step error probability.
        seed: Base seed of the run (episode seeds derive from it).

    Returns:
        The ``RunReport`` of the run.
    """
    config = RunConfig(
        run_id=f"validate-p{round(p * 100):03d}-s{seed}-n{n_trials}",
        tasks_glob=tasks_glob,
        agent=AgentConfig(kind="noisy_oracle", params={"p_error": p, "seed": seed}),
        n_trials=n_trials,
        base_seed=seed,
        transport="inprocess",
    )
    return runner.run(
        config,
        write=False,
        light=True,
        deps=runner.RunDeps(tasks=_tasks_for(tasks_glob)),
    )


def make_labstock_simulator(tasks_glob: str, n_trials: int = N_TRIALS) -> SimulateRun:
    """Build a picklable simulator ``(p, seed) -> RunReport`` over the LabStock runner.

    Args:
        tasks_glob: Glob of the task YAML files.
        n_trials: Trials per task.

    Returns:
        A ``functools.partial`` of ``simulate_labstock``.
    """
    return functools.partial(simulate_labstock, tasks_glob, n_trials)
