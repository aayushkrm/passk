"""Assign one primary ``FailureCategory`` to a trajectory.

Checking order (first match wins): ``policy_violation`` > ``success`` > ``schema_error`` >
``tool_error_unrecovered`` > ``false_success_claim`` > ``loop`` > ``max_turns`` >
``wrong_state``. ``success`` sits second because a met goal with no violation is a pass,
whatever errors were recovered on the way.

Rules, each evaluated on the data of one episode:

1. ``policy_violation``: the environment recorded at least one violation message.
2. ``success``: the goal assertions hold and no violation was recorded. This check comes
   before the failure chain below because ``Trajectory.failure`` must be ``success``
   whenever ``Trajectory.success`` is True (see ``schema.py``). A recovered error or a
   loop inside a passing episode is therefore not a failure label.
3. ``schema_error``: some tool result has ``ok=False`` and an error message matching one
   of the invalid-call patterns in ``_SCHEMA_ERROR``, such as "invalid argument",
   "unknown tool" or "missing required". Matching is case-insensitive and applies to the
   whole episode, recovered or not.
4. ``tool_error_unrecovered``: some tool result has ``ok=False`` and no later tool result
   with the same tool name has ``ok=True``. A failed call that the agent later retried
   successfully is recovered and does not count.
5. ``false_success_claim``: the agent ended with a ``final`` action while the goal is not
   met. The agent said it was done and it was not.
6. ``loop``: at least ``min_repeat`` (default 3) consecutive tool calls with identical
   name and arguments, anywhere in the episode (see ``is_loop``).
7. ``max_turns``: the episode used all of its turn budget (``len(steps) >= max_turns``).
8. ``wrong_state``: none of the above. The episode stopped, the goal is not met and no
   earlier rule explains why.
"""

import re
from collections.abc import Sequence

from passk.schema import FailureCategory, Step, ToolCall, Trajectory

_SCHEMA_ERROR = re.compile(
    r"invalid argument|unexpected argument|unknown argument|missing required|"
    r"unknown tool|validation error|schema|invalid type|wrong type",
    re.IGNORECASE,
)


def _tool_call(step: Step) -> ToolCall | None:
    """Return the tool call of a step, or None for a ``final`` action.

    Args:
        step: One turn of the episode.

    Returns:
        The ``ToolCall`` when the action is a tool call, otherwise None.
    """
    return step.action.tool_call


def _has_schema_error(steps: Sequence[Step]) -> bool:
    """Return True if any failed tool result matches the invalid-call patterns.

    Args:
        steps: Steps of the episode.

    Returns:
        Whether a ``ToolResult`` with ``ok=False`` has an error message matching
        ``_SCHEMA_ERROR``.
    """
    for step in steps:
        result = step.result
        if result is not None and not result.ok and result.error:
            if _SCHEMA_ERROR.search(result.error):
                return True
    return False


def _has_unrecovered_tool_error(steps: Sequence[Step]) -> bool:
    """Return True if a failed tool call was never followed by a successful retry.

    A failure is recovered when a later step calls a tool with the same name and gets
    ``ok=True``. Any failure without such a later success makes the episode unrecovered.

    Args:
        steps: Steps of the episode.

    Returns:
        Whether at least one tool failure is unrecovered.
    """
    for index, step in enumerate(steps):
        result = step.result
        if result is None or result.ok:
            continue
        call = _tool_call(step)
        if call is None:
            return True
        recovered = False
        for later in steps[index + 1 :]:
            later_call = _tool_call(later)
            later_result = later.result
            if (
                later_call is not None
                and later_call.name == call.name
                and later_result is not None
                and later_result.ok
            ):
                recovered = True
                break
        if not recovered:
            return True
    return False


def is_loop(steps: Sequence[Step], min_repeat: int = 3) -> bool:
    """Return True if the episode contains ``min_repeat`` identical consecutive tool calls.

    Two calls are identical when they have the same ``name`` and the same ``arguments``.
    The run may appear anywhere in the episode. A ``final`` action breaks a run.

    Args:
        steps: Steps of the episode.
        min_repeat: Number of identical consecutive calls that counts as a loop, at least 2.

    Returns:
        Whether the episode contains such a run.

    Raises:
        ValueError: If ``min_repeat`` is below 2.
    """
    if min_repeat < 2:
        raise ValueError(f"min_repeat must be >= 2, got {min_repeat}")
    run = 0
    previous: ToolCall | None = None
    for step in steps:
        call = _tool_call(step)
        if call is None:
            run = 0
            previous = None
            continue
        if (
            previous is not None
            and call.name == previous.name
            and call.arguments == previous.arguments
        ):
            run += 1
        else:
            run = 1
        previous = call
        if run >= min_repeat:
            return True
    return False


def classify(
    steps: Sequence[Step],
    *,
    violations: Sequence[str],
    goal_met: bool,
    claimed_done: bool,
    max_turns: int,
) -> FailureCategory:
    """Return the primary failure category for one episode.

    The rules are listed in the module docstring. The first matching rule wins.

    Args:
        steps: Steps of the episode.
        violations: Policy violation messages from the environment.
        goal_met: Whether all goal assertions hold and no forbidden assertion holds.
        claimed_done: Whether the agent ended with a ``final`` action.
        max_turns: Turn budget of the task.

    Returns:
        ``FailureCategory.POLICY_VIOLATION`` if there are violations, otherwise
        ``FailureCategory.SUCCESS`` if ``goal_met``, otherwise the first failure rule that
        holds, falling back to ``FailureCategory.WRONG_STATE``.
    """
    if violations:
        return FailureCategory.POLICY_VIOLATION
    if goal_met:
        return FailureCategory.SUCCESS
    if _has_schema_error(steps):
        return FailureCategory.SCHEMA_ERROR
    if _has_unrecovered_tool_error(steps):
        return FailureCategory.TOOL_ERROR_UNRECOVERED
    if claimed_done:
        return FailureCategory.FALSE_SUCCESS_CLAIM
    if is_loop(steps):
        return FailureCategory.LOOP
    if len(steps) >= max_turns:
        return FailureCategory.MAX_TURNS
    return FailureCategory.WRONG_STATE


def classify_trajectory(trajectory: Trajectory, max_turns: int) -> FailureCategory:
    """Classify a recorded trajectory by deriving the ``classify`` arguments from it.

    ``goal_met`` is ``trajectory.success``, and ``claimed_done`` is True when the last
    step is a ``final`` action.

    Args:
        trajectory: The recorded episode.
        max_turns: Turn budget of the task the episode ran against.

    Returns:
        The primary failure category, as ``classify`` returns it.
    """
    claimed_done = bool(trajectory.steps) and trajectory.steps[-1].action.kind == "final"
    return classify(
        trajectory.steps,
        violations=trajectory.violations,
        goal_met=trajectory.success,
        claimed_done=claimed_done,
        max_turns=max_turns,
    )
