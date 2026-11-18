"""Tiny assertion DSL over final environment state.

Paths are dotted, with optional list filters, for example ``orders[compound_id=C07].grams``.
All functions here are pure.
"""

from collections.abc import Sequence
from typing import Any

from passk.schema import Assertion


def resolve_path(state: dict[str, Any], path: str) -> list[Any]:
    """Return every value addressed by ``path`` in ``state``.

    Args:
        state: Environment state.
        path: Dotted path with optional ``[key=value]`` filters on lists.

    Returns:
        The matched values, in traversal order. Empty if nothing matches.
    """
    raise NotImplementedError("resolve_path: scheduled for work package P-B")


def check_assertion(state: dict[str, Any], assertion: Assertion) -> bool:
    """Evaluate one assertion against ``state``.

    Args:
        state: Environment state.
        assertion: Assertion to evaluate (``equals``, ``contains``, ``not_contains``,
            ``count_eq``, ``absent``).

    Returns:
        True if the assertion holds.
    """
    raise NotImplementedError("check_assertion: scheduled for work package P-B")


def check_assertions(state: dict[str, Any], assertions: Sequence[Assertion]) -> list[str]:
    """Evaluate several assertions and collect a message for each one that fails.

    Args:
        state: Environment state.
        assertions: Assertions to evaluate.

    Returns:
        Human-readable failure messages, empty if every assertion holds.
    """
    raise NotImplementedError("check_assertions: scheduled for work package P-B")
