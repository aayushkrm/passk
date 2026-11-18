"""Calibration of the estimator and the gate (DECISION.md section 2.1, experiments E1 and E2).

Everything here runs on scripted agents, so the output is SYNTHETIC. It tests the tool
against a known ground truth, not a model.

E1 (estimator): a NoisyOracle with error rate p, run for ``N_TRIALS`` trials per task.
Each reference step is corrupted with probability p, so task t succeeds with
q_t = (1 - p) ** m_t, where m_t is the number of reference steps. The true pass^k is
mean_t q_t ** k. The measured pass^k is the unbiased estimator over the trials, averaged
over tasks. The max absolute error is reported for k = 1..5.

E2 (gate): ``pairs`` seeded A/A comparisons (p = 0.10 against p = 0.10) give the false-alarm
rate. The same number of A/B comparisons (p = 0.05 baseline against p = 0.15 candidate) give
power. Both rates carry Wilson 95% intervals.

The simulator is injected as ``simulate_run(p, seed) -> RunReport``. The CLI supplies a
default that runs the real runner. Tests supply a fast fake.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from passk import runner
from passk.gate import gate
from passk.schema import AgentConfig, GatePolicy, RunConfig, RunReport, Task

N_TRIALS = 20
K_MAX = 5
NOISE_LEVELS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
AA_P = 0.10
AB_P: tuple[float, float] = (0.05, 0.15)
DEFAULT_PAIRS = 200
DEFAULT_POLICY = GatePolicy(k=3, max_drop=0.10)
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
        A dict with ``p``, ``seed``, ``pass_hat_k`` (per k: measured, analytic, abs_error)
        and ``max_abs_error`` over k.

    Raises:
        ValueError: If the simulator does not run ``N_TRIALS`` trials per task, does not
            cover the given tasks, or its stored pass^k disagrees with the counts.
    """
    report = simulate_run(p, seed)
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
        per_k[k] = {
            "measured": measured,
            "analytic": analytic,
            "abs_error": abs(measured - analytic),
        }
    return {
        "p": p,
        "seed": seed,
        "pass_hat_k": per_k,
        "max_abs_error": max(v["abs_error"] for v in per_k.values()),
    }


def _pair_seed(base_seed: int, offset: int, index: int, arm: int) -> int:
    return base_seed + offset + 2 * index + arm


def run_validation(
    simulate_run: SimulateRun,
    tasks: Sequence[Task],
    *,
    policy: GatePolicy = DEFAULT_POLICY,
    pairs: int = DEFAULT_PAIRS,
    base_seed: int = 0,
) -> dict[str, Any]:
    """Run experiments E1 and E2 with an injected simulator.

    Args:
        simulate_run: Function ``(p, seed) -> RunReport`` that runs ``N_TRIALS`` trials per
            task on ``tasks``.
        tasks: The task set, used for the analytic values.
        policy: Gate policy used in E2.
        pairs: Number of A/A and of A/B comparisons.
        base_seed: Offset for every seed in this calibration.

    Returns:
        A JSON-serialisable dict labelled SYNTHETIC with keys ``e1``, ``e2`` and
        ``runtime_s``.

    Raises:
        ValueError: On an invalid simulator output, an empty task set or ``pairs < 1``.
    """
    if pairs < 1:
        raise ValueError("pairs must be at least 1")
    started = time.perf_counter()

    levels = [
        check_estimator(simulate_run, tasks, p, base_seed + index)
        for index, p in enumerate(NOISE_LEVELS)
    ]
    e1 = {
        "n_trials": N_TRIALS,
        "k_values": list(range(1, K_MAX + 1)),
        "levels": levels,
        "max_abs_error": max(level["max_abs_error"] for level in levels),
    }

    aa_failures = 0
    for index in range(pairs):
        base = simulate_run(AA_P, _pair_seed(base_seed, 10_000, index, 0))
        cand = simulate_run(AA_P, _pair_seed(base_seed, 10_000, index, 1))
        if not gate(base, cand, policy).passed:
            aa_failures += 1

    ab_detections = 0
    for index in range(pairs):
        base = simulate_run(AB_P[0], _pair_seed(base_seed, 20_000, index, 0))
        cand = simulate_run(AB_P[1], _pair_seed(base_seed, 20_000, index, 1))
        if not gate(base, cand, policy).passed:
            ab_detections += 1

    aa_low, aa_high = wilson_interval(aa_failures, pairs)
    ab_low, ab_high = wilson_interval(ab_detections, pairs)
    e2 = {
        "policy": policy.model_dump(mode="json"),
        "pairs": pairs,
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


def make_labstock_simulator(tasks_glob: str, workdir: Path) -> SimulateRun:
    """Build the default simulator, which runs the real runner on NoisyOracle agents.

    Each run is written under ``workdir`` and removed after its report is read.

    Args:
        tasks_glob: Glob of the task YAML files.
        workdir: Scratch directory for run output.

    Returns:
        A function ``(p, seed) -> RunReport``.
    """

    def simulate(p: float, seed: int) -> RunReport:
        run_id = f"validate-p{round(p * 100):03d}-s{seed}"
        config = RunConfig(
            run_id=run_id,
            tasks_glob=tasks_glob,
            agent=AgentConfig(kind="noisy_oracle", params={"p_error": p, "seed": seed}),
            n_trials=N_TRIALS,
            base_seed=seed,
            transport="inprocess",
        )
        try:
            return runner.run(config, out_root=workdir)
        finally:
            shutil.rmtree(workdir / run_id, ignore_errors=True)

    return simulate


def scratch_dir() -> tempfile.TemporaryDirectory[str]:
    """Create a temporary directory for simulator output; the caller cleans it up."""
    return tempfile.TemporaryDirectory(prefix="passk-validate-")
