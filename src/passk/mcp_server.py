"""MCP v2 server that exposes environment tools over stdio.

Uses the MCP v2 SDK API (``from mcp.server import MCPServer``, ``@mcp.tool()``), not
``FastMCP``. The 2026-07-28 spec removed sessions, so per-episode state lives in the
environment under a server-minted ``env_handle`` that the client passes as an explicit
tool argument.

Every ``Env.tools()`` spec becomes an MCP tool with ``env_handle`` added as a required
argument. Three control tools manage episodes: ``episode_reset``, ``episode_state`` and
``episode_violations``. The original ``ToolSpec.parameters`` travel in the tool's
``meta`` so a client can recover them exactly.

Run as ``python -m passk.mcp_server labstock`` to serve the LabStock environment.
"""

import inspect
import keyword
import sys
import threading
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import WithJsonSchema

from passk.schema import Env, Task, ToolCall, ToolResult, ToolSpec

HANDLE_ARG = "env_handle"
RESET_TOOL = "episode_reset"
STATE_TOOL = "episode_state"
VIOLATIONS_TOOL = "episode_violations"
CONTROL_TOOLS: tuple[str, ...] = (RESET_TOOL, STATE_TOOL, VIOLATIONS_TOOL)
META_KEY = "passk"
DEFAULT_TASKS_GLOB = str(Path(__file__).parent / "tasks" / "labstock" / "*.yaml")

_HANDLE_SCHEMA: dict[str, Any] = {
    "type": "string",
    "description": "Handle returned by episode_reset that identifies one episode.",
}


def build_server(env: Env, tasks: Sequence[Task], server_name: str = "passk") -> MCPServer:
    """Create an ``MCPServer`` whose tools wrap ``env`` and the given tasks.

    Args:
        env: Environment to serve. Any object implementing the ``Env`` protocol.
        tasks: Tasks that ``episode_reset`` may start, looked up by ``id``.
        server_name: Name reported to MCP clients.

    Returns:
        A configured server. Not started.

    Raises:
        ValueError: If task ids repeat, a tool name repeats or collides with a control
            tool, or a tool argument name is not a usable Python identifier or is
            ``env_handle``.
    """
    task_by_id: dict[str, Task] = {}
    for task in tasks:
        if task.id in task_by_id:
            raise ValueError(f"duplicate task id: {task.id!r}")
        task_by_id[task.id] = task

    specs = env.tools()
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError("environment declares duplicate tool names")
    clashes = sorted(set(names) & set(CONTROL_TOOLS))
    if clashes:
        raise ValueError(f"environment tool names collide with control tools: {clashes}")

    mcp = MCPServer(server_name)
    # Environments hold per-episode state in plain dicts; serialise access to them.
    lock = threading.Lock()

    for spec in specs:
        _register_env_tool(mcp, env, lock, spec)

    @mcp.tool(
        name=RESET_TOOL,
        description="Start an episode for a task and return its env_handle.",
    )
    def episode_reset(task_id: str, seed: int) -> dict[str, str]:
        """Start an episode for ``task_id`` seeded with ``seed``."""
        task = task_by_id.get(task_id)
        if task is None:
            raise ToolError(f"unknown task_id: {task_id!r}")
        with lock:
            handle = env.reset(task, seed)
        return {HANDLE_ARG: handle}

    @mcp.tool(
        name=STATE_TOOL,
        description="Return the current state of the episode identified by env_handle.",
    )
    def episode_state(env_handle: str) -> dict[str, Any]:
        """Return the state of one episode."""
        with lock:
            return env.state(env_handle)

    @mcp.tool(
        name=VIOLATIONS_TOOL,
        description="Return policy violations recorded so far for env_handle.",
    )
    def episode_violations(env_handle: str) -> dict[str, list[str]]:
        """Return the policy violations of one episode."""
        with lock:
            return {"violations": env.violations(env_handle)}

    return mcp


def _register_env_tool(mcp: MCPServer, env: Env, lock: threading.Lock, spec: ToolSpec) -> None:
    """Register one ``Env`` tool on ``mcp`` with a signature built from its JSON Schema.

    Args:
        mcp: Server to register on.
        env: Environment that executes the call.
        lock: Lock serialising access to ``env``.
        spec: Tool description whose parameters become the MCP argument schema.

    Raises:
        ValueError: If an argument name cannot be used as a Python parameter name.
    """
    properties: dict[str, Any] = dict(spec.parameters.get("properties", {}))
    required = set(spec.parameters.get("required", []))
    if HANDLE_ARG in properties:
        raise ValueError(f"tool {spec.name!r} must not declare the argument {HANDLE_ARG!r}")

    parameters = [
        inspect.Parameter(
            HANDLE_ARG,
            inspect.Parameter.KEYWORD_ONLY,
            annotation=Annotated[str, WithJsonSchema(_HANDLE_SCHEMA)],
        )
    ]
    for arg_name, arg_schema in properties.items():
        if not arg_name.isidentifier() or arg_name.startswith("_") or _is_keyword(arg_name):
            raise ValueError(f"tool {spec.name!r} has unusable argument name {arg_name!r}")
        parameters.append(
            inspect.Parameter(
                arg_name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=Annotated[Any, WithJsonSchema(arg_schema)],
                default=inspect.Parameter.empty if arg_name in required else None,
            )
        )

    def dispatch(**kwargs: Any) -> dict[str, Any]:
        handle = str(kwargs.pop(HANDLE_ARG))
        # Optional arguments the client omitted (sent as None) are left out of the call.
        arguments = {k: v for k, v in kwargs.items() if v is not None or k in required}
        call = ToolCall(call_id=uuid.uuid4().hex, name=spec.name, arguments=arguments)
        try:
            with lock:
                result = env.call(handle, call)
        except Exception as exc:
            result = ToolResult(
                call_id=call.call_id,
                ok=False,
                content=None,
                error=f"{type(exc).__name__}: {exc}",
            )
        return result.model_dump(mode="json")

    dispatch.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        parameters, return_annotation=dict[str, Any]
    )
    mcp.add_tool(
        dispatch,
        name=spec.name,
        description=spec.description,
        meta={META_KEY: {"parameters": spec.parameters}},
    )


def _is_keyword(name: str) -> bool:
    """Return True if ``name`` is a reserved Python keyword."""
    return keyword.iskeyword(name)


def serve_stdio(env: Env, tasks: Sequence[Task], server_name: str = "passk") -> None:
    """Run a server for ``env`` on stdio until the client disconnects.

    Args:
        env: Environment to serve.
        tasks: Tasks that ``episode_reset`` may start.
        server_name: Name reported to MCP clients.
    """
    build_server(env, tasks, server_name).run("stdio")


def serve_labstock_stdio(tasks_glob: str = DEFAULT_TASKS_GLOB) -> None:
    """Load the LabStock tasks and run the LabStock server on stdio.

    This is the entry point for ``passk serve-tools labstock``.

    Args:
        tasks_glob: Glob of task YAML files to serve.
    """
    from passk.envs.base import load_tasks
    from passk.envs.labstock import LabStockEnv

    tasks = load_tasks(tasks_glob)
    serve_stdio(LabStockEnv(), tasks, server_name="passk-labstock")


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point used by ``python -m passk.mcp_server``.

    Args:
        argv: Arguments after the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code: 0 after a clean run, 2 on a usage error.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["labstock"]:
        sys.stderr.write("usage: python -m passk.mcp_server labstock\n")
        return 2
    serve_labstock_stdio()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
