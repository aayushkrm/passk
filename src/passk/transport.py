"""Transports that carry tool calls from the runner to an environment.

``InProcessTransport`` calls an ``Env`` object directly and is the default for tests and
demos. ``MCPStdioTransport`` talks to ``passk.mcp_server`` over MCP v2 stdio. It exposes the
synchronous ``ToolTransport`` protocol by running the asyncio MCP client on a dedicated
background thread.

Both transports return the same ``ToolResult`` for the same calls. Two differences are
documented on ``MCPStdioTransport``: results are JSON-normalised, and protocol-level
failures raise ``TransportError`` instead of the environment's own exception type.
"""

import asyncio
import concurrent.futures
import os
import sys
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from passk.mcp_server import (
    CONTROL_TOOLS,
    HANDLE_ARG,
    META_KEY,
    RESET_TOOL,
    STATE_TOOL,
    VIOLATIONS_TOOL,
)
from passk.schema import Env, Task, ToolCall, ToolResult, ToolSpec

if TYPE_CHECKING:
    from mcp import Client
    from mcp.types import CallToolResult

T = TypeVar("T")

_SHUTDOWN_TIMEOUT_S = 10.0


CHILD_ENV_KEYS: tuple[str, ...] = ("PATH", "PYTHONPATH", "HOME", "LANG", "VIRTUAL_ENV")
"""Variables copied from the parent into the MCP server process. Nothing else is passed."""


def child_environment(parent: Mapping[str, str] | None = None) -> dict[str, str]:
    """Return the minimal environment for the MCP server subprocess.

    Args:
        parent: Environment to copy from. Defaults to ``os.environ``.

    Returns:
        Only the keys in ``CHILD_ENV_KEYS`` that the parent defines.
    """
    source = os.environ if parent is None else parent
    return {key: source[key] for key in CHILD_ENV_KEYS if key in source}


class TransportError(RuntimeError):
    """A tool call could not be completed because of the transport or the server."""


class TransportClosedError(TransportError):
    """The transport was closed, or the server connection is gone."""


class TransportTimeoutError(TransportError):
    """The server did not answer, or did not start, within the configured timeout."""


def _error_text(exc: BaseException) -> str:
    """Format an exception the same way on both transports."""
    return f"{type(exc).__name__}: {exc}"


class InProcessTransport:
    """``ToolTransport`` that calls an ``Env`` object in the same process."""

    def __init__(self, env: Env) -> None:
        """Wrap an environment.

        Args:
            env: Environment to call directly.
        """
        self._env = env

    def reset(self, task: Task, seed: int) -> str:
        """Start an episode and return its handle.

        Args:
            task: Task to run.
            seed: Episode seed.

        Returns:
            The handle minted by the environment.
        """
        return self._env.reset(task, seed)

    def tools(self) -> list[ToolSpec]:
        """Return the environment's tool specs.

        Returns:
            The specs exactly as ``Env.tools()`` reports them.
        """
        return self._env.tools()

    def call(self, handle: str, call: ToolCall) -> ToolResult:
        """Execute a tool call for the episode identified by ``handle``.

        An exception raised by the environment becomes ``ToolResult(ok=False)``, matching
        the MCP server, so the runner sees the same outcome on both transports.

        Args:
            handle: Episode handle from ``reset``.
            call: Tool call to execute.

        Returns:
            The environment's result, or a failed result if the environment raised.
        """
        try:
            return self._env.call(handle, call)
        except Exception as exc:
            return ToolResult(call_id=call.call_id, ok=False, content=None, error=_error_text(exc))

    def state(self, handle: str) -> dict[str, Any]:
        """Return the episode state identified by ``handle``.

        Args:
            handle: Episode handle.

        Returns:
            The environment's state dict.
        """
        return self._env.state(handle)

    def violations(self, handle: str) -> list[str]:
        """Return policy violations for the episode identified by ``handle``.

        Args:
            handle: Episode handle.

        Returns:
            Violation messages recorded so far.
        """
        return self._env.violations(handle)


class MCPStdioTransport:
    """``ToolTransport`` that speaks MCP v2 over stdio to ``passk.mcp_server``.

    The server process starts on first use. A background thread owns the asyncio event
    loop and the MCP client, so the public methods block like the in-process ones.
    Use ``close()`` or a ``with`` block to stop the server process.

    Differences from ``InProcessTransport``:

    * Results are JSON-normalised (tuples become lists, for example).
    * A protocol or server failure in ``reset``, ``state`` or ``violations`` raises
      ``TransportError``. In ``call``, a server-side failure becomes ``ToolResult(ok=False)``.
    * Argument-schema violations are rejected by the MCP layer and reported as failed results.
    """

    def __init__(
        self,
        env_name: str = "labstock",
        command: Sequence[str] | None = None,
        *,
        timeout_s: float = 30.0,
        startup_timeout_s: float = 60.0,
    ) -> None:
        """Configure the transport. The subprocess starts on first use.

        Args:
            env_name: Environment served by the MCP server. Used only for the default command.
            command: Command line that starts the server. Defaults to
                ``[sys.executable, "-m", "passk.mcp_server", env_name]``.
            timeout_s: Seconds to wait for any single request before raising
                ``TransportTimeoutError``.
            startup_timeout_s: Seconds to wait for the server to start and complete the
                MCP handshake.

        Raises:
            ValueError: If the command is empty or a timeout is not positive.
        """
        resolved = (
            tuple(command)
            if command is not None
            else (sys.executable, "-m", "passk.mcp_server", env_name)
        )
        if not resolved:
            raise ValueError("command must not be empty")
        if timeout_s <= 0 or startup_timeout_s <= 0:
            raise ValueError("timeouts must be positive")
        self._command: tuple[str, ...] = resolved
        self._timeout_s = timeout_s
        self._startup_timeout_s = startup_timeout_s
        self._lifecycle = threading.Lock()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._failure: BaseException | None = None
        self._client: Client | None = None
        self._task: asyncio.Task[None] | None = None
        self._tool_cache: list[ToolSpec] | None = None

    def __enter__(self) -> "MCPStdioTransport":
        """Return the transport; the server starts lazily on first use."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the transport on leaving the ``with`` block."""
        self.close()

    def close(self) -> None:
        """Stop the server subprocess and release the client. Safe to call twice."""
        with self._lifecycle:
            if self._closed:
                return
            self._closed = True
            loop, thread = self._loop, self._thread
        if loop is not None and thread is not None and thread.is_alive():
            loop.call_soon_threadsafe(self._cancel_supervisor)
            thread.join(timeout=_SHUTDOWN_TIMEOUT_S)

    def reset(self, task: Task, seed: int) -> str:
        """Start an episode on the server and return its server-minted handle.

        Args:
            task: Task to run. Its ``id`` must exist in the server's task list.
            seed: Episode seed.

        Returns:
            The handle minted by the server.

        Raises:
            TransportError: If the server rejects the reset.
        """
        payload = self._control(RESET_TOOL, {"task_id": task.id, "seed": seed})
        handle = payload.get(HANDLE_ARG)
        if not isinstance(handle, str):
            raise TransportError(f"{RESET_TOOL} returned no {HANDLE_ARG}")
        return handle

    def tools(self) -> list[ToolSpec]:
        """List the environment tools served by the MCP server.

        Control tools are excluded. Each spec's parameters are the ``Env`` originals,
        recovered from the tool ``meta``. The listing is cached after the first call.

        Returns:
            The tool specs, matching ``Env.tools()`` on the server.

        Raises:
            TransportError: If the server cannot list its tools.
        """
        if self._tool_cache is None:
            listing = self._submit(lambda client: client.list_tools())
            specs: list[ToolSpec] = []
            for tool in listing.tools:
                if tool.name in CONTROL_TOOLS:
                    continue
                meta = tool.meta or {}
                info = meta.get(META_KEY)
                if isinstance(info, dict) and "parameters" in info:
                    parameters = dict(info["parameters"])
                else:
                    parameters = _strip_handle(tool.input_schema)
                specs.append(
                    ToolSpec(
                        name=tool.name, description=tool.description or "", parameters=parameters
                    )
                )
            self._tool_cache = specs
        return list(self._tool_cache)

    def call(self, handle: str, call: ToolCall) -> ToolResult:
        """Call a tool over MCP, passing ``handle`` as an explicit argument.

        Args:
            handle: Episode handle from ``reset``.
            call: Tool call to execute. Its ``call_id`` is restored on the result.

        Returns:
            The environment's result. Server-side failures and argument-schema
            violations become ``ToolResult(ok=False)``.

        Raises:
            TransportError: If the transport itself fails (closed, timed out, or the
                server is unreachable).
        """
        arguments: dict[str, Any] = {**call.arguments, HANDLE_ARG: handle}
        result = self._submit(lambda client: client.call_tool(call.name, arguments))
        if result.is_error:
            return ToolResult(
                call_id=call.call_id,
                ok=False,
                content=None,
                error=_text(result) or "tool call failed",
            )
        payload = _payload(result)
        if not isinstance(payload, dict):
            raise TransportError(f"tool {call.name!r} returned a non-object payload")
        return ToolResult.model_validate(payload).model_copy(update={"call_id": call.call_id})

    def state(self, handle: str) -> dict[str, Any]:
        """Fetch the episode state from the server.

        Args:
            handle: Episode handle.

        Returns:
            The state dict reported by the server.

        Raises:
            TransportError: If the server rejects the request.
        """
        return self._control(STATE_TOOL, {HANDLE_ARG: handle})

    def violations(self, handle: str) -> list[str]:
        """Fetch policy violations from the server.

        Args:
            handle: Episode handle.

        Returns:
            Violation messages recorded so far.

        Raises:
            TransportError: If the server rejects the request or the payload is malformed.
        """
        payload = self._control(VIOLATIONS_TOOL, {HANDLE_ARG: handle})
        violations = payload.get("violations")
        if not isinstance(violations, list):
            raise TransportError(f"{VIOLATIONS_TOOL} returned no violations list")
        return [str(item) for item in violations]

    def _control(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a control tool and return its object payload.

        Args:
            name: Control tool name.
            arguments: Tool arguments.

        Returns:
            The decoded payload object.

        Raises:
            TransportError: If the call fails or the payload is not an object.
        """
        result = self._submit(lambda client: client.call_tool(name, arguments))
        if result.is_error:
            raise TransportError(f"{name} failed: {_text(result)}")
        payload = _payload(result)
        if not isinstance(payload, dict):
            raise TransportError(f"{name} returned a non-object payload")
        return payload

    def _submit(self, op: "Callable[[Client], Awaitable[T]]") -> T:
        """Run ``op`` on the background loop and block until it finishes.

        Args:
            op: Function producing the coroutine to run against the connected client.

        Returns:
            The value the coroutine returns.

        Raises:
            TransportClosedError: If the transport is closed or the connection is gone.
            TransportTimeoutError: If the request exceeds ``timeout_s``.
            TransportError: For any other failure in the request.
        """
        self._start()
        loop = self._loop
        if loop is None:
            raise TransportClosedError("MCP transport is not running")

        async def run() -> T:
            client = self._client
            if client is None:
                raise TransportClosedError("MCP server connection is no longer available")
            return await op(client)

        future = asyncio.run_coroutine_threadsafe(run(), loop)
        try:
            return future.result(timeout=self._timeout_s)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise TransportTimeoutError(f"MCP request timed out after {self._timeout_s}s") from exc
        except TransportError:
            raise
        except Exception as exc:
            raise TransportError(f"MCP request failed: {_error_text(exc)}") from exc

    def _start(self) -> None:
        """Start the background thread and wait until the server is connected.

        Raises:
            TransportClosedError: If the transport was closed.
            TransportTimeoutError: If the server does not become ready in time.
            TransportError: If the server failed to start.
        """
        with self._lifecycle:
            if self._closed:
                raise TransportClosedError("MCP transport is closed")
            if self._thread is None:
                loop = asyncio.new_event_loop()
                self._loop = loop
                self._thread = threading.Thread(
                    target=self._run_loop,
                    args=(loop,),
                    name="passk-mcp-stdio",
                    daemon=True,
                )
                self._thread.start()
            ready = self._ready
        if not ready.wait(self._startup_timeout_s):
            self.close()
            raise TransportTimeoutError(
                f"MCP server did not start within {self._startup_timeout_s}s"
            )
        if self._failure is not None:
            raise TransportError(f"MCP server failed to start: {_error_text(self._failure)}")
        if self._client is None:
            raise TransportClosedError("MCP server connection is no longer available")

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Thread body: run the supervisor coroutine until the connection ends.

        Args:
            loop: Event loop owned by this thread.
        """
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._supervise())
        except asyncio.CancelledError:
            pass
        finally:
            self._client = None
            self._ready.set()
            loop.close()

    async def _supervise(self) -> None:
        """Own the MCP client for its whole life.

        The client context must be entered and exited in the same task, so this coroutine
        holds it open until ``close()`` cancels it.
        """
        if self._closed:
            return
        self._task = asyncio.current_task()
        from mcp import Client, StdioServerParameters

        # The child gets only the whitelisted variables in CHILD_ENV_KEYS. The MCP SDK adds
        # its own small default set (get_default_environment) under them. Secrets in the parent,
        # such as an API key, are not passed to a process that does not need them.
        params = StdioServerParameters(
            command=self._command[0],
            args=list(self._command[1:]),
            env=child_environment(),
        )
        try:
            async with Client(params) as client:
                self._client = client
                self._ready.set()
                await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._failure = exc
        finally:
            self._client = None
            self._ready.set()

    def _cancel_supervisor(self) -> None:
        """Cancel the supervisor task. Runs on the background loop."""
        if self._task is not None:
            self._task.cancel()


def _strip_handle(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove ``env_handle`` from a wire input schema. Used only when ``meta`` is absent.

    Args:
        schema: JSON Schema the server published for the tool.

    Returns:
        A copy without the handle argument and without a generated top-level title.
    """
    out = dict(schema)
    out.pop("title", None)
    properties = dict(out.get("properties", {}))
    properties.pop(HANDLE_ARG, None)
    out["properties"] = properties
    required = [name for name in out.get("required", []) if name != HANDLE_ARG]
    if required:
        out["required"] = required
    else:
        out.pop("required", None)
    return out


def _text(result: "CallToolResult") -> str:
    """Join the text blocks of a tool result.

    Args:
        result: MCP tool result.

    Returns:
        The text content joined by newlines, or an empty string.
    """
    from mcp.types import TextContent

    return "\n".join(block.text for block in result.content if isinstance(block, TextContent))


def _payload(result: "CallToolResult") -> Any:
    """Decode the structured payload of a successful tool result.

    Args:
        result: MCP tool result that is not an error.

    Returns:
        The structured content, or the JSON parsed from the text content.

    Raises:
        TransportError: If the result carries no decodable payload.
    """
    import json

    if result.structured_content is not None:
        return result.structured_content
    try:
        return json.loads(_text(result))
    except json.JSONDecodeError as exc:
        raise TransportError("tool result carried no decodable payload") from exc
