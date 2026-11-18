"""LabStock: a deterministic lab-inventory environment with policies.

Compounds have an id, name, stock in grams, a hazard class (``none``, ``flammable``,
``toxic`` or ``controlled``) and a price per gram. Policies: a controlled compound needs
an approval before it can be ordered, and the budget cannot be exceeded. An optional
seeded transient failure (``flaky_p``) makes some tool calls fail.
"""

from typing import Any

from passk.envs.base import BaseEnv
from passk.schema import Task, ToolCall, ToolResult, ToolSpec

TOOL_NAMES: tuple[str, ...] = (
    "search_compounds",
    "get_compound",
    "place_order",
    "request_approval",
    "cancel_order",
    "get_budget",
)


class LabStockEnv(BaseEnv):
    """Lab-inventory environment implementing the ``Env`` protocol."""

    name = "labstock"

    def __init__(self, flaky_p: float = 0.0) -> None:
        """Create the environment.

        Args:
            flaky_p: Probability that a tool call fails transiently (seeded per episode).
        """
        raise NotImplementedError("LabStockEnv: scheduled for work package P-B")

    def reset(self, task: Task, seed: int) -> str:
        """Start an episode for ``task`` seeded with ``seed``; return its handle."""
        raise NotImplementedError("LabStockEnv.reset: scheduled for work package P-B")

    def tools(self) -> list[ToolSpec]:
        """Return the six LabStock tool specs."""
        raise NotImplementedError("LabStockEnv.tools: scheduled for work package P-B")

    def call(self, handle: str, call: ToolCall) -> ToolResult:
        """Execute one LabStock tool call for the episode identified by ``handle``."""
        raise NotImplementedError("LabStockEnv.call: scheduled for work package P-B")

    def state(self, handle: str) -> dict[str, Any]:
        """Return compounds, orders, approvals and budget for the episode."""
        raise NotImplementedError("LabStockEnv.state: scheduled for work package P-B")

    def violations(self, handle: str) -> list[str]:
        """Return policy violations (for example an unapproved controlled order)."""
        raise NotImplementedError("LabStockEnv.violations: scheduled for work package P-B")
