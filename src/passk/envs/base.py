"""Shared environment machinery: handle minting, task loading and the base class."""

from abc import ABC, abstractmethod
from typing import Any

from passk.schema import Task, ToolCall, ToolResult, ToolSpec


def mint_handle(task_id: str, seed: int, counter: int) -> str:
    """Mint an opaque, server-side handle for one episode.

    Args:
        task_id: Task the episode runs.
        seed: Episode seed.
        counter: Monotonic counter kept by the environment instance.

    Returns:
        A string handle that is unique within the environment instance.
    """
    raise NotImplementedError("mint_handle: scheduled for work package P-B")


def load_tasks(tasks_glob: str) -> list[Task]:
    """Load and validate every task YAML matching ``tasks_glob``.

    Args:
        tasks_glob: Glob pattern, for example ``src/passk/tasks/labstock/*.yaml``.

    Returns:
        Tasks sorted by id.

    Raises:
        ValueError: If a file does not validate against ``Task``.
    """
    raise NotImplementedError("load_tasks: scheduled for work package P-B")


class BaseEnv(ABC):
    """Abstract environment. Concrete classes implement the ``Env`` protocol shape."""

    name: str = "base"

    @abstractmethod
    def reset(self, task: Task, seed: int) -> str:
        """Start an episode and return its handle.

        Args:
            task: Task to run.
            seed: Episode seed.

        Returns:
            Handle identifying the episode state.
        """
        raise NotImplementedError

    @abstractmethod
    def tools(self) -> list[ToolSpec]:
        """Return the tools this environment exposes."""
        raise NotImplementedError

    @abstractmethod
    def call(self, handle: str, call: ToolCall) -> ToolResult:
        """Execute a tool call for the episode identified by ``handle``."""
        raise NotImplementedError

    @abstractmethod
    def state(self, handle: str) -> dict[str, Any]:
        """Return the current state of the episode identified by ``handle``."""
        raise NotImplementedError

    @abstractmethod
    def violations(self, handle: str) -> list[str]:
        """Return policy violations recorded so far for the episode."""
        raise NotImplementedError
