"""Deterministic agents: oracle, noisy oracle and replay from JSONL."""

from pathlib import Path

from passk.schema import AgentAction, Step, Task, ToolSpec


class OracleAgent:
    """Replays the task's ``reference_solution`` and then returns a final action."""

    def act(self, task: Task, history: list[Step], tools: list[ToolSpec]) -> AgentAction:
        """Return the next reference call, or a final action when the solution is done.

        Args:
            task: Task being solved.
            history: Steps taken so far.
            tools: Tools available (unused by the oracle).

        Returns:
            The next action.
        """
        raise NotImplementedError("OracleAgent.act: scheduled for work package P-B")


class NoisyOracleAgent:
    """Oracle that, with probability ``p_error``, corrupts a step into an irrecoverable call."""

    def __init__(self, p_error: float, seed: int) -> None:
        """Create the agent.

        Args:
            p_error: Per-step probability of corrupting the reference call.
            seed: Seed for the corruption draws.
        """
        raise NotImplementedError("NoisyOracleAgent: scheduled for work package P-B")

    def act(self, task: Task, history: list[Step], tools: list[ToolSpec]) -> AgentAction:
        """Return the next action, corrupted with probability ``p_error``.

        Args:
            task: Task being solved.
            history: Steps taken so far.
            tools: Tools available.

        Returns:
            The next action.
        """
        raise NotImplementedError("NoisyOracleAgent.act: scheduled for work package P-B")


class ReplayAgent:
    """Replays recorded actions from a JSONL file, one line per step."""

    def __init__(self, path: Path) -> None:
        """Load the recorded actions.

        Args:
            path: JSONL file with one serialised ``AgentAction`` per line.
        """
        raise NotImplementedError("ReplayAgent: scheduled for work package P-B")

    def act(self, task: Task, history: list[Step], tools: list[ToolSpec]) -> AgentAction:
        """Return the recorded action for the current step.

        Args:
            task: Task being solved.
            history: Steps taken so far.
            tools: Tools available.

        Returns:
            The recorded action.
        """
        raise NotImplementedError("ReplayAgent.act: scheduled for work package P-D")
