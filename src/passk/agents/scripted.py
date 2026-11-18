"""Deterministic agents: oracle, noisy oracle and replay from JSONL.

``OracleAgent`` replays the task's reference solution. When a call fails with the transient
error it re-issues the same call once. ``NoisyOracleAgent`` follows the oracle but, with
probability ``p_error`` per step, replaces the step with an irrecoverable policy-violating
order. ``ReplayAgent`` replays the actions recorded in a trajectory JSONL file.

Everything here is SYNTHETIC: scripted output is never a model result.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from passk.schema import AgentAction, Step, Task, ToolCall, ToolSpec, Trajectory

_TRANSIENT = "transient"
_BUST_GRAMS = 1_000_000.0


def _reference_position(history: list[Step]) -> int:
    """Return how many reference calls have been completed so far.

    A transient failure does not advance the position, so the same reference call is
    retried. Every other step does.
    """
    position = 0
    for step in history:
        result = step.result
        if result is not None and not result.ok and result.error == _TRANSIENT:
            continue
        position += 1
    return position


def _final(message: str) -> AgentAction:
    """Build a final action."""
    return AgentAction(kind="final", tool_call=None, message=message)


class OracleAgent:
    """Replays the task's ``reference_solution`` and then returns a final action."""

    def act(self, task: Task, history: list[Step], tools: list[ToolSpec]) -> AgentAction:
        """Return the next reference call, or a final action when the solution is done.

        A call that failed transiently is re-issued with a fresh ``call_id``.

        Args:
            task: Task being solved.
            history: Steps taken so far.
            tools: Tools available (unused by the oracle).

        Returns:
            The next action.
        """
        del tools
        position = _reference_position(history)
        if position >= len(task.reference_solution):
            return _final("done")
        reference = task.reference_solution[position]
        last = history[-1].result if history else None
        retrying = last is not None and not last.ok and last.error == _TRANSIENT
        call_id = f"{reference.call_id}-retry{len(history)}" if retrying else reference.call_id
        call = ToolCall(call_id=call_id, name=reference.name, arguments=dict(reference.arguments))
        return AgentAction(kind="tool_call", tool_call=call, message=None)


class NoisyOracleAgent:
    """Oracle that, with probability ``p_error``, corrupts a step into an irrecoverable call.

    The draw for step ``i`` of task ``t`` is ``random.Random(f"{seed}|{episode}|{t}|{i}")``,
    where ``episode`` is the runner's per-episode seed (see ``begin_episode``), so trials are
    independent and still reproducible. A corrupted step is an order for a
    quantity far beyond the budget. The env records it as a budget violation, and violations
    persist, so the episode can never succeed afterwards.
    """

    def __init__(self, p_error: float, seed: int) -> None:
        """Create the agent.

        Args:
            p_error: Per-step probability of corrupting the reference call, in [0, 1].
            seed: Seed for the corruption draws.

        Raises:
            ValueError: If ``p_error`` is outside [0, 1].
        """
        if not 0.0 <= p_error <= 1.0:
            raise ValueError(f"p_error must be in [0, 1], got {p_error}")
        self.p_error = p_error
        self.seed = seed
        self._episode_seed: int | None = None
        self._oracle = OracleAgent()

    def begin_episode(self, seed: int) -> None:
        """Mix the runner's per-episode seed into the corruption draws.

        Without this every trial of a task would replay the same corruption pattern, and
        trials would not be independent samples.

        Args:
            seed: Episode seed supplied by the runner.
        """
        self._episode_seed = seed

    def act(self, task: Task, history: list[Step], tools: list[ToolSpec]) -> AgentAction:
        """Return the next action, corrupted with probability ``p_error``.

        Args:
            task: Task being solved.
            history: Steps taken so far.
            tools: Tools available.

        Returns:
            The next action. Final actions are never corrupted.
        """
        action = self._oracle.act(task, history, tools)
        if action.kind != "tool_call":
            return action
        index = len(history)
        rng = random.Random(f"{self.seed}|{self._episode_seed}|{task.id}|{index}")
        if rng.random() >= self.p_error:
            return action
        return AgentAction(
            kind="tool_call",
            tool_call=ToolCall(
                call_id=f"noise-{index}",
                name="place_order",
                arguments={"compound_id": _bust_target(task), "grams": _BUST_GRAMS},
            ),
            message=None,
        )


def _bust_target(task: Task) -> str:
    """Return the compound id used for a corrupted order: the first with a positive price."""
    compounds = task.initial_state["compounds"]
    for compound in compounds:
        if float(compound["price_per_g"]) > 0:
            return str(compound["id"])
    return str(compounds[0]["id"])


class ReplayAgent:
    """Replays recorded actions from a JSONL file of ``Trajectory`` records, one per line."""

    def __init__(self, path: Path) -> None:
        """Load the recorded actions.

        When a task appears in several lines, the first line for that task is used.

        Args:
            path: JSONL file with one serialised ``Trajectory`` per line.

        Raises:
            ValueError: If a line is not a valid trajectory.
        """
        self._actions: dict[str, list[AgentAction]] = {}
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                trajectory = Trajectory.model_validate(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: not a valid Trajectory: {exc}") from exc
            self._actions.setdefault(trajectory.task_id, [step.action for step in trajectory.steps])

    def act(self, task: Task, history: list[Step], tools: list[ToolSpec]) -> AgentAction:
        """Return the recorded action for the current step.

        Args:
            task: Task being solved.
            history: Steps taken so far.
            tools: Tools available (unused).

        Returns:
            The recorded action, or a final action once the recording is exhausted.

        Raises:
            ValueError: If no trajectory was recorded for ``task.id``.
        """
        del tools
        actions = self._actions.get(task.id)
        if actions is None:
            raise ValueError(f"no recorded trajectory for task {task.id!r}")
        index = len(history)
        if index < len(actions):
            return actions[index]
        return _final("replay exhausted")
