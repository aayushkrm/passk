"""Assign one primary ``FailureCategory`` to a trajectory.

Precedence (first match wins): ``policy_violation`` > ``schema_error`` >
``tool_error_unrecovered`` > ``false_success_claim`` > ``loop`` > ``max_turns`` >
``wrong_state`` > ``success``.
"""

from collections.abc import Sequence

from passk.schema import FailureCategory, Step


def is_loop(steps: Sequence[Step], min_repeat: int = 3) -> bool:
    """Return True if the last ``min_repeat`` tool calls are identical consecutive calls.

    Args:
        steps: Steps of the episode.
        min_repeat: Number of identical consecutive calls that counts as a loop.

    Returns:
        Whether the episode ends in a loop.
    """
    raise NotImplementedError("is_loop: scheduled for work package P-A")


def classify(
    steps: Sequence[Step],
    *,
    violations: Sequence[str],
    goal_met: bool,
    claimed_done: bool,
    max_turns: int,
) -> FailureCategory:
    """Return the primary failure category for one episode.

    Args:
        steps: Steps of the episode.
        violations: Policy violation messages from the environment.
        goal_met: Whether all goal assertions hold and no forbidden assertion holds.
        claimed_done: Whether the agent ended with a ``final`` action.
        max_turns: Turn budget of the task.

    Returns:
        ``FailureCategory.SUCCESS`` only when ``goal_met`` and there are no violations.
    """
    raise NotImplementedError("classify: scheduled for work package P-A")
