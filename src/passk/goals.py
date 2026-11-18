"""Tiny assertion DSL over final environment state.

Paths are dotted. A segment is a name, optionally followed by a ``[key=value]`` filter that
keeps the list items whose ``key`` renders to ``value``. For example
``orders[compound_id=C07].grams`` is the ``grams`` of every order for compound ``C07``.
A list met by a segment without a filter is expanded to its dict items, so
``orders.grams`` yields the grams of every order. A path that ends on a list returns the
list itself. Filter values cannot contain ``.``, ``[`` or ``]``.

Operators (``Assertion.op``):

- ``equals``: at least one value matches and every matched value equals ``value`` (numbers
  compare within 1e-6). Repeated identical records, such as two approvals, still pass.
- ``contains``: some matched value is a list containing ``value``, a string containing it as
  a substring, or a dict with that key.
- ``not_contains``: the negation of ``contains``. True when the path is absent.
- ``count_eq``: the number of matched values equals ``value`` (a non-negative int).
- ``absent``: nothing matches.

Goal and forbidden semantics: every goal assertion must hold; a forbidden assertion must
NOT hold. Write forbidden assertions as the wrong outcome, for example a duplicate order.

All functions here are pure.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from passk.schema import Assertion, Task

_SEGMENT = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)(?:\[([A-Za-z_][A-Za-z0-9_]*)=([^\[\].]+)\])?$")
_OPS: frozenset[str] = frozenset({"equals", "contains", "not_contains", "count_eq", "absent"})
_NUMBER_TOLERANCE = 1e-6


@dataclass(frozen=True)
class PathSegment:
    """One parsed segment of a dotted path.

    Attributes:
        name: Dict key to read.
        key: Filter key, or None when the segment has no filter.
        value: Filter value that ``key`` must render to, or None.
    """

    name: str
    key: str | None = None
    value: str | None = None


def parse_path(path: str) -> list[PathSegment]:
    """Parse a dotted path with optional list filters.

    Args:
        path: Path such as ``orders[compound_id=C07].grams``.

    Returns:
        The parsed segments, in order.

    Raises:
        ValueError: If the path is empty or any segment is malformed.
    """
    if not path:
        raise ValueError("empty assertion path")
    segments: list[PathSegment] = []
    for raw in path.split("."):
        match = _SEGMENT.match(raw)
        if match is None:
            raise ValueError(f"invalid path segment {raw!r} in {path!r}")
        segments.append(PathSegment(match.group(1), match.group(2), match.group(3)))
    return segments


def resolve_path(state: dict[str, Any], path: str) -> list[Any]:
    """Return every value addressed by ``path`` in ``state``.

    Args:
        state: Environment state.
        path: Dotted path with optional ``[key=value]`` filters on lists.

    Returns:
        The matched values, in traversal order. Empty if nothing matches.

    Raises:
        ValueError: If ``path`` is malformed.
    """
    nodes: list[Any] = [state]
    for segment in parse_path(path):
        next_nodes: list[Any] = []
        for node in nodes:
            candidates: list[Any] = (
                [item for item in node if isinstance(item, dict)]
                if isinstance(node, list)
                else [node]
            )
            for candidate in candidates:
                if not isinstance(candidate, dict) or segment.name not in candidate:
                    continue
                value = candidate[segment.name]
                if segment.key is None:
                    next_nodes.append(value)
                elif isinstance(value, list):
                    next_nodes.extend(
                        item
                        for item in value
                        if isinstance(item, dict)
                        and segment.key in item
                        and str(item[segment.key]) == segment.value
                    )
        nodes = next_nodes
    return nodes


def validate_assertion(assertion: Assertion) -> None:
    """Check that an assertion is well formed.

    Args:
        assertion: Assertion to check.

    Raises:
        ValueError: If the path is malformed, the operator is unknown, or the value does not
            fit the operator (``count_eq`` needs a non-negative int; ``contains`` and
            ``not_contains`` need a value).
    """
    parse_path(assertion.path)
    if assertion.op not in _OPS:
        raise ValueError(f"unknown assertion op {assertion.op!r}")
    if assertion.op == "count_eq":
        value = assertion.value
        if not _is_int(value) or value < 0:
            raise ValueError(f"count_eq on {assertion.path!r} needs a non-negative int value")
    if assertion.op in {"contains", "not_contains"} and assertion.value is None:
        raise ValueError(f"{assertion.op} on {assertion.path!r} needs a value")


def evaluate(assertion: Assertion, state: dict[str, Any]) -> bool:
    """Evaluate one assertion against ``state``.

    Args:
        assertion: Assertion to evaluate.
        state: Environment state.

    Returns:
        True if the assertion holds.

    Raises:
        ValueError: If the assertion is malformed (see ``validate_assertion``).
    """
    validate_assertion(assertion)
    matches = resolve_path(state, assertion.path)
    if assertion.op == "absent":
        return not matches
    if assertion.op == "equals":
        return bool(matches) and all(_same(match, assertion.value) for match in matches)
    if assertion.op == "count_eq":
        return len(matches) == assertion.value
    contained = any(_contains(match, assertion.value) for match in matches)
    if assertion.op == "contains":
        return contained
    return not contained


check_assertion = evaluate
"""Alias of ``evaluate``, kept for callers written against the stub name."""


def describe(assertion: Assertion) -> str:
    """Return a short human-readable description of an assertion.

    Args:
        assertion: Assertion to describe.

    Returns:
        A string such as ``equals orders[compound_id=C07].grams 25``.
    """
    if assertion.op == "absent":
        return f"absent {assertion.path}"
    return f"{assertion.op} {assertion.path} {assertion.value!r}"


def check_assertions(state: dict[str, Any], assertions: Sequence[Assertion]) -> list[str]:
    """Evaluate several assertions and collect a message for each one that fails.

    Args:
        state: Environment state.
        assertions: Assertions to evaluate. Each is read as a goal (must hold).

    Returns:
        Human-readable failure messages, empty if every assertion holds.
    """
    return [
        f"unmet: {describe(assertion)}"
        for assertion in assertions
        if not evaluate(assertion, state)
    ]


def explain_goal(task: Task, state: dict[str, Any], violations: Sequence[str]) -> list[str]:
    """List every reason a task is not solved in ``state``.

    Args:
        task: Task whose goal and forbidden assertions are checked.
        state: Final environment state.
        violations: Policy violations recorded during the episode.

    Returns:
        Messages for each violation, unmet goal and triggered forbidden assertion. Empty
        when the task is solved.
    """
    reasons = [f"violation: {message}" for message in violations]
    reasons.extend(check_assertions(state, task.goal))
    reasons.extend(
        f"forbidden: {describe(assertion)}"
        for assertion in task.forbidden
        if evaluate(assertion, state)
    )
    return reasons


def check_goal(task: Task, state: dict[str, Any], violations: Sequence[str]) -> bool:
    """Decide whether an episode solved ``task``.

    Success needs no policy violations, every goal assertion to hold and every forbidden
    assertion to be false.

    Args:
        task: Task that was attempted.
        state: Final environment state.
        violations: Policy violations recorded during the episode.

    Returns:
        True if the task is solved.
    """
    if violations:
        return False
    if not all(evaluate(assertion, state) for assertion in task.goal):
        return False
    return not any(evaluate(assertion, state) for assertion in task.forbidden)


def _is_int(value: Any) -> bool:
    """Return True for ints that are not bools."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """Return True for finite ints and floats that are not bools."""
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _same(actual: Any, expected: Any) -> bool:
    """Compare two values, treating bools as distinct from numbers.

    Args:
        actual: Value found in the state.
        expected: Value written in the assertion.

    Returns:
        True when the values match.
    """
    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual == expected
    if _is_number(actual) and _is_number(expected):
        return abs(actual - expected) <= _NUMBER_TOLERANCE
    return bool(actual == expected)


def _contains(container: Any, item: Any) -> bool:
    """Return True if ``item`` is inside ``container`` (list, string or dict keys).

    Args:
        container: Matched value.
        item: Value the assertion looks for.

    Returns:
        True when the container holds the item.
    """
    if isinstance(container, list):
        return any(_same(element, item) for element in container)
    if isinstance(container, dict):
        return isinstance(item, str) and item in container
    if isinstance(container, str):
        return isinstance(item, str) and item in container
    return False
