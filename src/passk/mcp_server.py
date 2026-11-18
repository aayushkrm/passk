"""MCP v2 server that exposes environment tools over stdio.

Uses the MCP v2 SDK API (``from mcp.server import MCPServer``, ``@mcp.tool()``), not
``FastMCP``. Per-episode state is held under a server-minted ``env_handle`` that the
client passes as a tool argument. The 2026-07-28 spec removed sessions.
"""

from mcp.server import MCPServer


def build_server(env_name: str = "labstock") -> MCPServer:
    """Create an ``MCPServer`` whose tools wrap the named environment.

    Args:
        env_name: Environment to serve. Only ``labstock`` is supported in v0.1.

    Returns:
        A configured server. Not started.
    """
    raise NotImplementedError("build_server: scheduled for work package P-C")


def serve_stdio(env_name: str = "labstock") -> None:
    """Run the server on stdio until the client disconnects.

    Args:
        env_name: Environment to serve.
    """
    raise NotImplementedError("serve_stdio: scheduled for work package P-C")
