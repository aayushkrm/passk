"""The MCP server subprocess gets a minimal environment, not the parent's full environment."""

from __future__ import annotations

import sys
from typing import Any

import pytest

pytest.importorskip("mcp")

from passk.transport import CHILD_ENV_KEYS, MCPStdioTransport, _payload, child_environment  # noqa: E402

SECRET_NAME = "OPENROUTER_API_KEY"
SECRET_VALUE = "sk-or-test-SECRET-0123456789"

# A tiny MCP server that reports the names of its own environment variables.
ENV_DUMP_SERVER = """
import os
from mcp.server import MCPServer
mcp = MCPServer("envdump")

@mcp.tool()
def dump_env() -> dict:
    return {"keys": sorted(os.environ)}

mcp.run("stdio")
"""


def test_child_environment_copies_only_the_whitelist() -> None:
    parent = {key: f"value-{key}" for key in CHILD_ENV_KEYS}
    parent[SECRET_NAME] = SECRET_VALUE
    parent["AWS_SECRET_ACCESS_KEY"] = "x"
    child = child_environment(parent)
    assert set(child) == set(CHILD_ENV_KEYS)
    assert SECRET_VALUE not in child.values()


def test_child_environment_is_empty_of_absent_keys() -> None:
    assert child_environment({"HOME": "/h"}) == {"HOME": "/h"}
    assert child_environment({}) == {}


def test_mcp_child_does_not_see_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SECRET_NAME, SECRET_VALUE)
    transport = MCPStdioTransport(command=[sys.executable, "-c", ENV_DUMP_SERVER])
    try:
        result = transport._submit(lambda client: client.call_tool("dump_env", {}))
    finally:
        transport.close()
    payload: Any = _payload(result)
    keys = set(payload["keys"])
    assert SECRET_NAME not in keys
    assert "PATH" in keys
    # Expected names: our whitelist, the MCP SDK's own default set, and what the interpreter
    # and the OS add at startup. None of them come from the parent's secrets.
    from mcp.client.stdio import get_default_environment

    allowed = (
        set(CHILD_ENV_KEYS)
        | set(get_default_environment())
        | {
            "LC_CTYPE",
            "__CF_USER_TEXT_ENCODING",
        }
    )
    assert keys <= allowed, sorted(keys - allowed)
