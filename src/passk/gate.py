"""CI regression gate.

A candidate fails if pass^k@K drops by more than ``max_drop`` AND the paired-bootstrap
CI for the drop excludes 0 (when ``require_significance`` is set), or if the candidate's
pass^k@K falls below the absolute floor ``min_pass_hat_k``.
"""

from passk.schema import GateDecision, GatePolicy, RunReport


def gate(baseline: RunReport, candidate: RunReport, policy: GatePolicy) -> GateDecision:
    """Decide whether ``candidate`` regresses against ``baseline`` under ``policy``.

    Args:
        baseline: Reference run report.
        candidate: New run report.
        policy: Gate thresholds.

    Returns:
        The decision with the paired delta and its CI.

    Raises:
        ValueError: If the two runs do not share the same task ids.
    """
    raise NotImplementedError("gate: scheduled for work package P-E")
