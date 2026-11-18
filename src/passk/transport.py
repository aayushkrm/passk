"""Transports that carry tool calls from the runner to an environment.

``InProcessTransport`` is the default for tests and demos. ``MCPStdioTransport`` talks
to ``passk serve-tools`` over MCP v2 stdio.
"""

from collections.abc import Sequence
from typing import Any

from passk.schema import Env, Task, ToolCall, ToolResult, ToolSpec


class InProcessTransport:
    """``ToolTransport`` that calls an ``Env`` object in the same process."""

    def __init__(self, env: Env) -> None:
        """Wrap an environment.

        Args:
            env: Environment to call directly.
        """
        raise NotImplementedError("InProcessTransport: scheduled for work package P-C")

    def reset(self, task: Task, seed: int) -> str:
        """Start an episode and return its handle."""
        raise NotImplementedError("InProcessTransport.reset: scheduled for work package P-C")

    def tools(self) -> list[ToolSpec]:
        """Return the environment's tool specs."""
        raise NotImplementedError("InProcessTransport.tools: scheduled for work package P-C")

    def call(self, handle: str, call: ToolCall) -> ToolResult:
        """Execute a tool call for the episode identified by ``handle``."""
        raise NotImplementedError("InProcessTransport.call: scheduled for work package P-C")

    def state(self, handle: str) -> dict[str, Any]:
        """Return the episode state identified by ``handle``."""
        raise NotImplementedError("InProcessTransport.state: scheduled for work package P-C")

    def violations(self, handle: str) -> list[str]:
        """Return policy violations for the episode identified by ``handle``."""
        raise NotImplementedError("InProcessTransport.violations: scheduled for work package P-C")


class MCPStdioTransport:
    """``ToolTransport`` that speaks MCP v2 over stdio to ``passk serve-tools``."""

    def __init__(self, env_name: str = "labstock", command: Sequence[str] | None = None) -> None:
        """Configure the transport. The subprocess starts on first use.

        Args:
            env_name: Environment served by the MCP server.
            command: Command line that starts the server. Defaults to
                ``passk serve-tools <env_name>``.
        """
        raise NotImplementedError("MCPStdioTransport: scheduled for work package P-C")

    def close(self) -> None:
        """Stop the server subprocess and release the client."""
        raise NotImplementedError("MCPStdioTransport.close: scheduled for work package P-C")

    def reset(self, task: Task, seed: int) -> str:
        """Start an episode on the server and return its server-minted handle."""
        raise NotImplementedError("MCPStdioTransport.reset: scheduled for work package P-C")

    def tools(self) -> list[ToolSpec]:
        """List the tools served by the MCP server."""
        raise NotImplementedError("MCPStdioTransport.tools: scheduled for work package P-C")

    def call(self, handle: str, call: ToolCall) -> ToolResult:
        """Call a tool over MCP, passing ``handle`` as an explicit argument."""
        raise NotImplementedError("MCPStdioTransport.call: scheduled for work package P-C")

    def state(self, handle: str) -> dict[str, Any]:
        """Fetch the episode state from the server."""
        raise NotImplementedError("MCPStdioTransport.state: scheduled for work package P-C")

    def violations(self, handle: str) -> list[str]:
        """Fetch policy violations from the server."""
        raise NotImplementedError("MCPStdioTransport.violations: scheduled for work package P-C")
