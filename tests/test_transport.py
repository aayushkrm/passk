"""Tests for ``passk.transport``.

The in-process tests run without ``mcp``. The MCP tests spawn a real server subprocess
(``run_fake_server`` from ``test_mcp_server``) and skip if ``mcp`` cannot be imported.
"""

import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from test_mcp_server import FakeEnv, make_tasks

from passk.schema import ToolCall
from passk.transport import (
    InProcessTransport,
    MCPStdioTransport,
    TransportClosedError,
    TransportError,
    TransportTimeoutError,
    _strip_handle,
)

TESTS_DIR = Path(__file__).resolve().parent
HANDLE_NAME = "env_handle"

# A scripted sequence with success, failure, a policy violation and an unknown item.
# Every call is valid for the schema, so both transports must agree on each result.
SCRIPT: list[tuple[str, dict[str, Any]]] = [
    ("add_stock", {"item": "acid", "units": 2}),
    ("add_stock", {"item": "base", "units": 4, "note": "restock"}),
    ("take_stock", {"item": "acid", "units": 7}),
    ("take_stock", {"item": "acid", "units": 4}),
    ("add_stock", {"item": "base", "units": 10}),
    ("take_stock", {"item": "base", "units": 6}),
    ("take_stock", {"item": "base", "units": 9}),
    ("get_stock", {"item": "acid"}),
    ("get_stock", {"item": "zinc"}),
    ("noop", {}),
]


def fake_server_command() -> list[str]:
    """Return the command that starts ``FakeEnv`` behind the MCP stdio server."""
    script = (
        f"import sys; sys.path.insert(0, {str(TESTS_DIR)!r}); "
        "from test_mcp_server import run_fake_server; run_fake_server()"
    )
    return [sys.executable, "-c", script]


def run_script(transport: Any) -> dict[str, Any]:
    """Run ``SCRIPT`` on a fresh episode and collect everything the runner would see."""
    task = make_tasks()[0]
    handle = transport.reset(task, seed=7)
    results = []
    for index, (name, arguments) in enumerate(SCRIPT):
        call = ToolCall(call_id=f"call-{index}", name=name, arguments=arguments)
        results.append(transport.call(handle, call).model_dump(mode="json"))
    return {
        "results": results,
        "state": transport.state(handle),
        "violations": transport.violations(handle),
    }


def check_isolation(transport: Any) -> None:
    """Two episodes on one transport keep separate stock and violations."""
    task_a, task_b = make_tasks()
    handle_a = transport.reset(task_a, seed=1)
    handle_b = transport.reset(task_b, seed=2)
    transport.call(
        handle_a, ToolCall(call_id="a1", name="add_stock", arguments={"item": "acid", "units": 4})
    )
    transport.call(
        handle_a, ToolCall(call_id="a2", name="take_stock", arguments={"item": "acid", "units": 6})
    )
    assert transport.state(handle_a) == {"task_id": "t-restock", "stock": {"acid": 1}}
    assert transport.state(handle_b) == {"task_id": "t-empty", "stock": {}}
    assert transport.violations(handle_a) == ["take_stock of 6 acid exceeds limit 5"]
    assert transport.violations(handle_b) == []


@pytest.fixture(scope="module")
def mcp_transport() -> Iterator[MCPStdioTransport]:
    """One MCP stdio transport (one server subprocess) shared by the module's tests."""
    pytest.importorskip("mcp")
    transport = MCPStdioTransport(command=fake_server_command(), timeout_s=20.0)
    yield transport
    transport.close()


# In-process behaviour (no mcp needed).


def test_in_process_tools_are_the_env_tools() -> None:
    """InProcessTransport reports exactly the Env's tool specs."""
    env = FakeEnv()
    assert InProcessTransport(env).tools() == env.tools()


def test_in_process_env_exception_becomes_failed_result() -> None:
    """An environment exception is returned as ok=False, the same as the MCP server does."""
    transport = InProcessTransport(FakeEnv())
    result = transport.call(
        "no-such-handle", ToolCall(call_id="x", name="get_stock", arguments={"item": "acid"})
    )
    assert result.ok is False
    assert result.call_id == "x"
    assert result.error is not None and result.error.startswith("KeyError")


def test_in_process_unknown_tool_is_failed_result() -> None:
    """An unknown tool name is a failed ToolResult, not an exception."""
    transport = InProcessTransport(FakeEnv())
    handle = transport.reset(make_tasks()[0], seed=0)
    result = transport.call(handle, ToolCall(call_id="y", name="bogus", arguments={}))
    assert result.ok is False
    assert result.error == "unknown tool bogus"


def test_in_process_isolation() -> None:
    """Two episodes on the in-process transport do not share state."""
    check_isolation(InProcessTransport(FakeEnv()))


def test_strip_handle_removes_only_the_handle() -> None:
    """The fallback path drops env_handle, the wire title and an emptied required list."""
    wire = {
        "type": "object",
        "title": "xArguments",
        "properties": {HANDLE_NAME: {"type": "string"}, "units": {"type": "integer"}},
        "required": [HANDLE_NAME, "units"],
    }
    assert _strip_handle(wire) == {
        "type": "object",
        "properties": {"units": {"type": "integer"}},
        "required": ["units"],
    }
    only_handle = {
        "type": "object",
        "properties": {HANDLE_NAME: {"type": "string"}},
        "required": [HANDLE_NAME],
    }
    assert _strip_handle(only_handle) == {"type": "object", "properties": {}}


# MCP stdio behaviour (spawns a server subprocess).


def test_in_process_and_mcp_stdio_give_identical_results(mcp_transport: MCPStdioTransport) -> None:
    """The same scripted calls produce the same ToolResults, state and violations."""
    in_process = run_script(InProcessTransport(FakeEnv()))
    over_mcp = run_script(mcp_transport)
    assert over_mcp == in_process
    assert in_process["results"][2]["ok"] is False
    assert in_process["results"][5]["ok"] is True
    assert in_process["violations"] == ["take_stock of 6 base exceeds limit 5"]


def test_mcp_tools_match_env_tools(mcp_transport: MCPStdioTransport) -> None:
    """list_tools over MCP gives the Env's ToolSpecs, without the control tools."""
    assert mcp_transport.tools() == FakeEnv().tools()


def test_mcp_isolation(mcp_transport: MCPStdioTransport) -> None:
    """Two episodes on the MCP transport do not share state."""
    check_isolation(mcp_transport)


def test_mcp_unknown_handle_raises_transport_error(mcp_transport: MCPStdioTransport) -> None:
    """A state request for an unknown handle raises TransportError over MCP."""
    with pytest.raises(TransportError):
        mcp_transport.state("no-such-handle")


def test_mcp_unknown_task_raises_transport_error(mcp_transport: MCPStdioTransport) -> None:
    """Resetting a task the server does not know raises TransportError."""
    from passk.schema import Task

    stranger = Task(
        id="not-served",
        tier=1,
        instruction="x",
        initial_state={},
        goal=[],
        forbidden=[],
        reference_solution=[],
    )
    with pytest.raises(TransportError, match="unknown task_id"):
        mcp_transport.reset(stranger, seed=0)


def test_mcp_request_timeout_raises() -> None:
    """A call that outlives timeout_s raises TransportTimeoutError and the transport closes."""
    pytest.importorskip("mcp")
    transport = MCPStdioTransport(command=fake_server_command(), timeout_s=0.5)
    try:
        handle = transport.reset(make_tasks()[0], seed=0)
        with pytest.raises(TransportTimeoutError):
            transport.call(
                handle,
                ToolCall(call_id="slow", name="pause", arguments={"seconds": 3}),
            )
    finally:
        transport.close()


def test_mcp_startup_timeout_raises() -> None:
    """A server that never completes the handshake raises TransportTimeoutError on time."""
    pytest.importorskip("mcp")
    command = [sys.executable, "-c", "import time; time.sleep(60)"]
    transport = MCPStdioTransport(command=command, startup_timeout_s=1.0)
    started = time.monotonic()
    with pytest.raises(TransportTimeoutError):
        transport.tools()
    assert time.monotonic() - started < 15


def test_mcp_server_that_exits_raises_transport_error() -> None:
    """A server command that exits at once raises TransportError rather than hanging."""
    pytest.importorskip("mcp")
    transport = MCPStdioTransport(
        command=[sys.executable, "-c", "raise SystemExit(3)"], startup_timeout_s=20.0
    )
    try:
        with pytest.raises(TransportError):
            transport.tools()
    finally:
        transport.close()


def test_closed_transport_raises_and_close_is_idempotent() -> None:
    """After close(), calls raise TransportClosedError and a second close() does nothing."""
    pytest.importorskip("mcp")
    transport = MCPStdioTransport(command=fake_server_command())
    transport.close()
    transport.close()
    with pytest.raises(TransportClosedError):
        transport.tools()


def test_context_manager_closes_the_server() -> None:
    """Leaving a with-block closes the transport so later calls raise TransportClosedError."""
    pytest.importorskip("mcp")
    with MCPStdioTransport(command=fake_server_command(), timeout_s=20.0) as transport:
        assert transport.tools() == FakeEnv().tools()
    with pytest.raises(TransportClosedError):
        transport.state("anything")


def test_invalid_configuration_is_rejected() -> None:
    """An empty command or a non-positive timeout is refused at construction."""
    with pytest.raises(ValueError, match="command must not be empty"):
        MCPStdioTransport(command=[])
    with pytest.raises(ValueError, match="timeouts must be positive"):
        MCPStdioTransport(timeout_s=0)
