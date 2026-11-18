"""Shared environment machinery: handle minting, task loading and the base class."""

from __future__ import annotations

import glob
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from passk.goals import validate_assertion
from passk.schema import Task, ToolCall, ToolResult, ToolSpec

_TASKS_ROOT = Path(__file__).resolve().parent.parent / "tasks"


def mint_handle(task_id: str, seed: int, counter: int) -> str:
    """Mint an opaque, server-side handle for one episode.

    Args:
        task_id: Task the episode runs.
        seed: Episode seed.
        counter: Monotonic counter kept by the environment instance.

    Returns:
        A string handle that is unique within the environment instance.
    """
    digest = hashlib.sha256(f"{task_id}|{seed}|{counter}".encode()).hexdigest()[:16]
    return f"ep{counter:06d}-{digest}"


def tasks_dir(env: str = "labstock") -> Path:
    """Return the directory that ships the task YAML files for ``env``.

    Args:
        env: Environment name, for example ``labstock``.

    Returns:
        Absolute path to ``passk/tasks/<env>``.
    """
    return _TASKS_ROOT / env


def load_tasks(tasks_glob: str) -> list[Task]:
    """Load and validate every task YAML matching ``tasks_glob``.

    Each file holds either one task mapping or a list of task mappings. Every task is
    validated against ``schema.Task``, its assertion paths are parsed, its goal must be
    non-empty and task ids must be unique across all files.

    Args:
        tasks_glob: Glob pattern, for example ``src/passk/tasks/labstock/*.yaml``.

    Returns:
        Tasks sorted by id.

    Raises:
        ValueError: If no file matches, a file is not valid YAML, a task does not validate,
            an assertion path is malformed, the goal is empty, or an id is duplicated.
    """
    paths = sorted(Path(p) for p in glob.glob(tasks_glob, recursive=True))
    paths = [p for p in paths if p.is_file()]
    if not paths:
        raise ValueError(f"no task files match {tasks_glob!r}")
    tasks: list[Task] = []
    first_seen: dict[str, Path] = {}
    for path in paths:
        for raw in _read_documents(path):
            task = _validate_task(raw, path)
            if task.id in first_seen:
                previous = first_seen[task.id]
                raise ValueError(
                    f"{path}: duplicate task id {task.id!r} (first defined in {previous})"
                )
            first_seen[task.id] = path
            tasks.append(task)
    if not tasks:
        raise ValueError(f"no tasks found in {tasks_glob!r}")
    return sorted(tasks, key=lambda t: t.id)


def _read_documents(path: Path) -> list[Any]:
    """Parse a task file into a list of raw task mappings.

    Args:
        path: YAML file to read.

    Returns:
        One entry per task mapping found in the file.

    Raises:
        ValueError: If the file is not valid YAML or has an unexpected top-level shape.
    """
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: not UTF-8 text: {exc}") from exc
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"{path}: top level must be a task mapping or a list of them")


def _validate_task(raw: Any, path: Path) -> Task:
    """Validate one raw task mapping and its assertions.

    Args:
        raw: Parsed YAML object for one task.
        path: File the task came from, used in error messages.

    Returns:
        The validated ``Task``.

    Raises:
        ValueError: On any validation failure, with the file and task id in the message.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: each task must be a mapping, got {type(raw).__name__}")
    label = str(raw.get("id", "<no id>"))
    try:
        task = Task.model_validate(raw)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors()
        )
        raise ValueError(f"{path}: task {label!r} does not validate: {details}") from exc
    if not task.goal:
        raise ValueError(f"{path}: task {task.id!r} has an empty goal")
    for assertion in [*task.goal, *task.forbidden]:
        try:
            validate_assertion(assertion)
        except ValueError as exc:
            raise ValueError(f"{path}: task {task.id!r}: {exc}") from exc
    call_ids = [call.call_id for call in task.reference_solution]
    if len(set(call_ids)) != len(call_ids):
        raise ValueError(f"{path}: task {task.id!r} has duplicate reference call_ids")
    return task


class BaseEnv(ABC):
    """Abstract environment. Concrete classes implement the ``Env`` protocol shape."""

    name: str = "base"

    def __init__(self) -> None:
        """Create the environment with an empty handle counter."""
        self._counter = 0

    def _new_handle(self, task: Task, seed: int) -> str:
        """Mint a handle that is unique within this environment instance.

        Args:
            task: Task the episode runs.
            seed: Episode seed.

        Returns:
            A new handle string.
        """
        self._counter += 1
        return mint_handle(task.id, seed, self._counter)

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
