"""Tests for ``passk.mcp_server``: server tools checked through an in-process MCP client.

``FakeEnv`` is a tiny environment with the ``Env`` protocol shape. The transport tests import
it too, and the stdio integration test runs it in a subprocess via ``run_fake_server``.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402
from mcp.server import MCPServer  # noqa: E402

from passk.mcp_server import (  # noqa: E402
    CONTROL_TOOLS,
    HANDLE_ARG,
    build_server,
    serve_stdio,
)
from passk.schema import Task, ToolCall, ToolResult, ToolSpec  # noqa: E402

TESTS_DIR = Path(__file__).resolve().parent
STOCK_LIMIT = 5
T = TypeVar("T")

ENV_TOOL_NAMES = ["add_stock", "take_stock", "get_stock", "noop", "pause"]


def make_tasks() -> list[Task]:
    """Return the two fake tasks the tests start episodes for."""
    return [
        Task(
            id="t-restock",
            tier=1,
            instruction="Restock acid and take some.",
            initial_state={"stock": {"acid": 3}},
            goal=[],
            forbidden=[],
            reference_solution=[],
        ),
        Task(
            id="t-empty",
            tier=2,
            instruction="Start from an empty shelf.",
            initial_state={"stock": {}},
            goal=[],
            forbidden=[],
            reference_solution=[],
        ),
    ]


class FakeEnv:
    """Minimal environment: per-episode stock counts and one policy rule.

    Taking more than ``STOCK_LIMIT`` units in one call records a violation. ``pause``
    sleeps, which lets the timeout tests stall the server.
    """

    def __init__(self) -> None:
        """Create an environment with no episodes."""
        self._episodes: dict[str, dict[str, Any]] = {}
        self._minted = 0

    def reset(self, task: Task, seed: int) -> str:
        """Start an episode for ``task`` and return a fresh handle."""
        self._minted += 1
        handle = f"fake-{self._minted:04d}"
        self._episodes[handle] = {
            "task_id": task.id,
            "seed": seed,
            "stock": dict(task.initial_state.get("stock", {})),
            "violations": [],
        }
        return handle

    def tools(self) -> list[ToolSpec]:
        """Return the fake tool specs."""
        return [
            ToolSpec(
                name="add_stock",
                description="Add units of an item to the shelf.",
                parameters={
                    "type": "object",
                    "properties": {
                        "item": {"type": "string", "description": "Item name"},
                        "units": {"type": "integer", "minimum": 1},
                        "note": {"type": "string"},
                    },
                    "required": ["item", "units"],
                },
            ),
            ToolSpec(
                name="take_stock",
                description="Take units of an item off the shelf.",
                parameters={
                    "type": "object",
                    "properties": {
                        "item": {"type": "string"},
                        "units": {"type": "integer"},
                    },
                    "required": ["item", "units"],
                },
            ),
            ToolSpec(
                name="get_stock",
                description="Read the units of an item on the shelf.",
                parameters={
                    "type": "object",
                    "properties": {"item": {"type": "string"}},
                    "required": ["item"],
                },
            ),
            ToolSpec(
                name="noop",
                description="Do nothing and answer ok.",
                parameters={"type": "object", "properties": {}, "required": []},
            ),
            ToolSpec(
                name="pause",
                description="Sleep for a number of seconds, then answer ok.",
                parameters={
                    "type": "object",
                    "properties": {"seconds": {"type": "number"}},
                    "required": ["seconds"],
                },
            ),
        ]

    def call(self, handle: str, call: ToolCall) -> ToolResult:
        """Execute one fake tool call for the episode ``handle``."""
        episode = self._episodes[handle]
        args = call.arguments
        if call.name == "add_stock":
            item, units = str(args["item"]), int(args["units"])
            episode["stock"][item] = episode["stock"].get(item, 0) + units
            content: Any = {
                "item": item,
                "stock": episode["stock"][item],
                "note": args.get("note"),
            }
        elif call.name == "take_stock":
            item, units = str(args["item"]), int(args["units"])
            have = episode["stock"].get(item, 0)
            if units > have:
                return ToolResult(
                    call_id=call.call_id,
                    ok=False,
                    content=None,
                    error=f"insufficient {item}: have {have}, want {units}",
                )
            if units > STOCK_LIMIT:
                episode["violations"].append(
                    f"take_stock of {units} {item} exceeds limit {STOCK_LIMIT}"
                )
            episode["stock"][item] = have - units
            content = {"item": item, "stock": episode["stock"][item]}
        elif call.name == "get_stock":
            item = str(args["item"])
            content = {"item": item, "stock": episode["stock"].get(item, 0)}
        elif call.name == "noop":
            content = "ok"
        elif call.name == "pause":
            time.sleep(float(args["seconds"]))
            content = "ok"
        else:
            return ToolResult(
                call_id=call.call_id, ok=False, content=None, error=f"unknown tool {call.name}"
            )
        return ToolResult(call_id=call.call_id, ok=True, content=content)

    def state(self, handle: str) -> dict[str, Any]:
        """Return a copy of the episode state."""
        episode = self._episodes[handle]
        return {"task_id": episode["task_id"], "stock": dict(episode["stock"])}

    def violations(self, handle: str) -> list[str]:
        """Return the violations recorded for the episode."""
        return list(self._episodes[handle]["violations"])


def run_fake_server() -> None:
    """Serve ``FakeEnv`` on stdio. Used as the subprocess entry point in tests."""
    serve_stdio(FakeEnv(), make_tasks(), server_name="passk-fake")


def run_client(server: MCPServer, fn: Callable[[Client], Awaitable[T]]) -> T:
    """Connect an in-process MCP client to ``server``, run ``fn`` and return its value."""

    async def go() -> T:
        async with Client(server) as client:
            return await fn(client)

    return asyncio.run(go())


def structured(result: Any) -> dict[str, Any]:
    """Return the structured payload of a tool result, asserting that it exists."""
    assert result.structured_content is not None
    payload: dict[str, Any] = result.structured_content
    return payload


def test_build_server_lists_env_and_control_tools() -> None:
    """Every env tool and each control tool is listed; env tools require the handle."""
    env = FakeEnv()
    server = build_server(env, make_tasks())

    async def fn(client: Client) -> Any:
        return await client.list_tools()

    listing = run_client(server, fn)
    names = [tool.name for tool in listing.tools]
    assert sorted(names) == sorted(ENV_TOOL_NAMES + list(CONTROL_TOOLS))
    by_name = {tool.name: tool for tool in listing.tools}
    add_schema = by_name["add_stock"].input_schema
    assert HANDLE_ARG in add_schema["properties"]
    assert HANDLE_ARG in add_schema["required"]
    assert by_name["add_stock"].meta == {"passk": {"parameters": env.tools()[0].parameters}}


def test_env_tool_round_trip_through_handle() -> None:
    """A tool call reaches the episode named by its env_handle and returns its result."""
    server = build_server(FakeEnv(), make_tasks())

    async def fn(client: Client) -> tuple[dict[str, Any], dict[str, Any]]:
        reset = structured(
            await client.call_tool("episode_reset", {"task_id": "t-restock", "seed": 1})
        )
        added = await client.call_tool(
            "add_stock", {HANDLE_ARG: reset[HANDLE_ARG], "item": "acid", "units": 2}
        )
        return reset, structured(added)

    reset, added = run_client(server, fn)
    assert reset[HANDLE_ARG].startswith("fake-")
    assert added["ok"] is True
    assert added["content"] == {"item": "acid", "stock": 5, "note": None}


def test_control_tools_expose_state_and_violations() -> None:
    """episode_state and episode_violations report on the handle they are given."""
    server = build_server(FakeEnv(), make_tasks())

    async def fn(client: Client) -> tuple[dict[str, Any], dict[str, Any]]:
        handle = structured(
            await client.call_tool("episode_reset", {"task_id": "t-restock", "seed": 2})
        )[HANDLE_ARG]
        await client.call_tool("take_stock", {HANDLE_ARG: handle, "item": "acid", "units": 3})
        state = structured(await client.call_tool("episode_state", {HANDLE_ARG: handle}))
        violations = structured(await client.call_tool("episode_violations", {HANDLE_ARG: handle}))
        return state, violations

    state, violations = run_client(server, fn)
    assert state == {"task_id": "t-restock", "stock": {"acid": 0}}
    assert violations == {"violations": []}


def test_policy_violation_is_recorded() -> None:
    """Taking more than the stock limit in one call is recorded as a violation."""
    server = build_server(FakeEnv(), make_tasks())

    async def fn(client: Client) -> dict[str, Any]:
        handle = structured(
            await client.call_tool("episode_reset", {"task_id": "t-restock", "seed": 3})
        )[HANDLE_ARG]
        await client.call_tool("add_stock", {HANDLE_ARG: handle, "item": "acid", "units": 9})
        await client.call_tool("take_stock", {HANDLE_ARG: handle, "item": "acid", "units": 6})
        return structured(await client.call_tool("episode_violations", {HANDLE_ARG: handle}))

    payload = run_client(server, fn)
    assert payload == {"violations": ["take_stock of 6 acid exceeds limit 5"]}


def test_unknown_task_id_is_a_tool_error() -> None:
    """episode_reset rejects a task id the server was not built with."""
    server = build_server(FakeEnv(), make_tasks())

    async def fn(client: Client) -> Any:
        return await client.call_tool("episode_reset", {"task_id": "nope", "seed": 0})

    result = run_client(server, fn)
    assert result.is_error is True


def test_env_exception_becomes_failed_result() -> None:
    """An exception from the environment is returned as ok=False, not as a protocol error."""
    server = build_server(FakeEnv(), make_tasks())

    async def fn(client: Client) -> Any:
        return await client.call_tool("get_stock", {HANDLE_ARG: "no-such-handle", "item": "acid"})

    result = run_client(server, fn)
    assert result.is_error is False
    payload = structured(result)
    assert payload["ok"] is False
    assert payload["error"].startswith("KeyError")


def test_optional_argument_omitted_is_not_passed() -> None:
    """An optional argument the client leaves out is absent from the env call."""
    server = build_server(FakeEnv(), make_tasks())

    async def fn(client: Client) -> dict[str, Any]:
        handle = structured(
            await client.call_tool("episode_reset", {"task_id": "t-empty", "seed": 4})
        )[HANDLE_ARG]
        result = await client.call_tool(
            "add_stock", {HANDLE_ARG: handle, "item": "base", "units": 1}
        )
        return structured(result)

    payload = run_client(server, fn)
    assert payload["content"]["note"] is None


def test_build_server_rejects_duplicate_task_ids() -> None:
    """Two tasks with the same id cannot both be served."""
    tasks = make_tasks() + make_tasks()
    with pytest.raises(ValueError, match="duplicate task id"):
        build_server(FakeEnv(), tasks)


class _ClashingEnv(FakeEnv):
    """FakeEnv whose tool list reuses a control tool name."""

    def tools(self) -> list[ToolSpec]:
        """Return the base tools plus one named like a control tool."""
        return [*super().tools(), ToolSpec(name="episode_reset", description="x", parameters={})]


def test_build_server_rejects_control_name_clash() -> None:
    """An env tool may not take a control tool's name."""
    with pytest.raises(ValueError, match="collide with control tools"):
        build_server(_ClashingEnv(), make_tasks())


class _HandleArgEnv(FakeEnv):
    """FakeEnv whose tool declares the reserved handle argument."""

    def tools(self) -> list[ToolSpec]:
        """Return a single tool that declares env_handle as an argument."""
        return [
            ToolSpec(
                name="bad",
                description="x",
                parameters={"type": "object", "properties": {HANDLE_ARG: {"type": "string"}}},
            )
        ]


def test_build_server_rejects_reserved_handle_argument() -> None:
    """A tool argument may not reuse the name env_handle."""
    with pytest.raises(ValueError, match="must not declare"):
        build_server(_HandleArgEnv(), make_tasks())


class _BadNameEnv(FakeEnv):
    """FakeEnv whose tool argument is not a usable Python identifier."""

    def tools(self) -> list[ToolSpec]:
        """Return a tool with a hyphenated argument name."""
        return [
            ToolSpec(
                name="bad",
                description="x",
                parameters={"type": "object", "properties": {"unit-count": {"type": "integer"}}},
            )
        ]


def test_build_server_rejects_unusable_argument_name() -> None:
    """An argument name that is not a Python identifier is rejected with a clear error."""
    with pytest.raises(ValueError, match="unusable argument name"):
        build_server(_BadNameEnv(), make_tasks())
