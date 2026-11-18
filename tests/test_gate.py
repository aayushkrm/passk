"""Gate decisions on the SYNTHETIC fixtures, policy loading and the bootstrap fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from passk import gate as gate_module
from passk import metrics
from passk.diff import load_report
from passk.gate import gate, load_policy
from passk.schema import GatePolicy, RunReport

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> RunReport:
    return load_report(FIXTURES / name)


@pytest.fixture
def baseline() -> RunReport:
    return _load("report_baseline.json")


def test_same_run_passes_with_zero_delta(baseline: RunReport) -> None:
    decision = gate(baseline, baseline, GatePolicy(k=1, max_drop=0.05))
    assert decision.passed is True
    assert decision.delta == pytest.approx(0.0)
    assert decision.ci_low == pytest.approx(0.0)
    assert decision.ci_high == pytest.approx(0.0)
    assert any("within max_drop" in reason for reason in decision.reasons)


def test_significant_regression_fails(baseline: RunReport) -> None:
    candidate = _load("report_regression.json")
    decision = gate(baseline, candidate, GatePolicy(k=1, max_drop=0.10))
    assert decision.passed is False
    # per-task deltas at k=1 are -0.6 (three tasks) and -0.4 (three tasks): mean -0.5
    assert decision.delta == pytest.approx(-0.5)
    assert decision.ci_high < 0.0
    assert any(r.startswith("regression:") for r in decision.reasons)


def test_drop_within_max_drop_passes(baseline: RunReport) -> None:
    candidate = _load("report_regression.json")
    decision = gate(baseline, candidate, GatePolicy(k=1, max_drop=0.60))
    assert decision.passed is True


def test_drop_without_significance_is_not_a_regression(baseline: RunReport) -> None:
    # mixed per-task changes: mean drop 0.067 at k=1, CI includes 0
    candidate = _load("report_minor_drop.json")
    policy = GatePolicy(k=1, max_drop=0.05, require_significance=True)
    decision = gate(baseline, candidate, policy)
    assert decision.passed is True
    assert decision.ci_low < 0.0 < decision.ci_high
    assert any("not a regression" in reason for reason in decision.reasons)


def test_drop_fails_when_significance_is_not_required(baseline: RunReport) -> None:
    candidate = _load("report_minor_drop.json")
    policy = GatePolicy(k=1, max_drop=0.05, require_significance=False)
    decision = gate(baseline, candidate, policy)
    assert decision.passed is False
    assert any("significance not required" in reason for reason in decision.reasons)


def test_floor_breach_fails_even_without_a_drop(baseline: RunReport) -> None:
    # baseline pass^1 is 24/30 = 0.8, below the floor of 0.9
    decision = gate(baseline, baseline, GatePolicy(k=1, max_drop=0.5, min_pass_hat_k=0.9))
    assert decision.passed is False
    assert any(r.startswith("floor:") and "below" in r for r in decision.reasons)


def test_floor_met_passes(baseline: RunReport) -> None:
    decision = gate(baseline, baseline, GatePolicy(k=1, max_drop=0.5, min_pass_hat_k=0.5))
    assert decision.passed is True
    assert any("meets min_pass_hat_k" in r for r in decision.reasons)


def test_task_id_mismatch_raises(baseline: RunReport) -> None:
    other = _load("report_other_tasks.json")
    with pytest.raises(ValueError, match="task ids differ"):
        gate(baseline, other, GatePolicy(k=1, max_drop=0.1))


def test_k_not_in_task_stats_raises(baseline: RunReport) -> None:
    with pytest.raises(ValueError, match="has no pass\\^9"):
        gate(baseline, baseline, GatePolicy(k=9, max_drop=0.1))


def test_invalid_alpha_raises(baseline: RunReport) -> None:
    with pytest.raises(ValueError, match="alpha"):
        gate(baseline, baseline, GatePolicy(k=1, max_drop=0.1, alpha=1.5))


def test_gate_is_reproducible(baseline: RunReport) -> None:
    candidate = _load("report_minor_drop.json")
    policy = GatePolicy(k=2, max_drop=0.05)
    assert gate(baseline, candidate, policy) == gate(baseline, candidate, policy)


def test_falls_back_to_local_bootstrap_when_metrics_stub_raises(
    baseline: RunReport, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = _load("report_regression.json")
    policy = GatePolicy(k=1, max_drop=0.10)
    expected = gate(baseline, candidate, policy)

    def not_yet(*_args: object, **_kwargs: object) -> tuple[float, float, float]:
        raise NotImplementedError("paired_bootstrap: scheduled for work package P-A")

    monkeypatch.setattr(metrics, "paired_bootstrap", not_yet)
    assert gate(baseline, candidate, policy) == expected


def test_uses_metrics_bootstrap_when_implemented(
    baseline: RunReport, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, float, int]] = []

    def fake(
        a: list[float], b: list[float], n_boot: int = 0, alpha: float = 0.0, seed: int = 0
    ) -> tuple[float, float, float]:
        calls.append((n_boot, alpha, seed))
        return (0.0, -0.01, 0.01)

    monkeypatch.setattr(metrics, "paired_bootstrap", fake)
    decision = gate(baseline, baseline, GatePolicy(k=1, max_drop=0.1))
    assert calls == [(gate_module.BOOTSTRAP_RESAMPLES, 0.05, gate_module.BOOTSTRAP_SEED)]
    assert decision.passed is True


def test_load_policy_reads_all_fields(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text(
        "k: 3\nmax_drop: 0.1\nmin_pass_hat_k: 0.5\nrequire_significance: false\nalpha: 0.1\n",
        encoding="utf-8",
    )
    policy = load_policy(path)
    assert policy == GatePolicy(
        k=3, max_drop=0.1, min_pass_hat_k=0.5, require_significance=False, alpha=0.1
    )


def test_load_policy_defaults(tmp_path: Path) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text("k: 5\nmax_drop: 0.2\n", encoding="utf-8")
    policy = load_policy(path)
    assert policy.min_pass_hat_k is None
    assert policy.require_significance is True
    assert policy.alpha == 0.05


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("k: 3\nmax_drop: 0.1\nmax_dorp: 0.2\n", "unknown policy keys"),
        ("- just\n- a list\n", "must be a YAML mapping"),
        ("k: 0\nmax_drop: 0.1\n", "k must be at least 1"),
        ("k: 3\nmax_drop: -0.1\n", "max_drop must be non-negative"),
        ("k: 3\n", "invalid policy"),
    ],
)
def test_load_policy_rejects_bad_files(tmp_path: Path, text: str, message: str) -> None:
    path = tmp_path / "gate.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_policy(path)
