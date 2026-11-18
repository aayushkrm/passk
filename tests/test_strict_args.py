"""Undeclared argument names: lenient (default) and strict, on both transports.

Lenient is the behaviour the E3 runs used over MCP, so it stays the default everywhere. The
two transports agree in both modes. Strict mode rejects undeclared names on both. The error
text differs between the transports (in process it is the labstock message, over MCP it is the
SDK's validation message), so the tests check that both fail and both name the key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import passk
from passk import runner
from passk.envs.base import load_tasks
from passk.envs.labstock import LabStockEnv, validate_env_params
from passk.schema import AgentConfig, RunConfig, Task, ToolCall, ToolResult
from passk.transport import MCPStdioTransport

GLOB = str(Path(passk.__file__).parent / "tasks" / "labstock" / "*.yaml")
UNDECLARED = "arguments"  # not a parameter of get_compound


def _task() -> Task:
    return sorted(load_tasks(GLOB), key=lambda t: t.id)[0]


def _call(arguments: dict[str, object]) -> ToolCall:
    return ToolCall(call_id="c1", name="get_compound", arguments=arguments)  # type: ignore[arg-type]


def _same(result: ToolResult) -> tuple[bool, object, object]:
    return result.ok, result.content, result.error


def _mcp_transport(strict: bool) -> MCPStdioTransport:
    """The transport a run would build: default command (lenient) or with env_params (strict)."""
    config = RunConfig(
        run_id="strict-probe",
        tasks_glob=GLOB,
        agent=AgentConfig(kind="oracle"),
        n_trials=1,
        base_seed=1,
        transport="mcp_stdio",
        env_params={"strict_args": True} if strict else {},
    )
    transport = runner.build_transport(config)
    assert isinstance(transport, MCPStdioTransport)
    return transport


def test_strict_args_is_a_boolean_env_param_and_defaults_to_off() -> None:
    assert validate_env_params({"strict_args": True}) == {"strict_args": True}
    assert validate_env_params({"strict_args": False}) == {"strict_args": False}
    assert validate_env_params({}) == {}  # no key means lenient
    with pytest.raises(ValueError, match="true or false"):
        validate_env_params({"strict_args": "yes"})
    with pytest.raises(ValueError, match="true or false"):
        validate_env_params({"strict_args": 1})


def test_default_environment_drops_undeclared_names_as_the_mcp_server_does() -> None:
    task = _task()
    plain, extra = LabStockEnv(), LabStockEnv()
    result_plain = plain.call(plain.reset(task, 1), _call({"compound_id": "C07"}))
    result_extra = extra.call(extra.reset(task, 1), _call({"compound_id": "C07", UNDECLARED: {}}))
    assert result_extra.ok is True
    assert _same(result_extra) == _same(result_plain)


def test_strict_environment_rejects_undeclared_names() -> None:
    env = LabStockEnv(strict_args=True)
    result = env.call(env.reset(_task(), 1), _call({"compound_id": "C07", UNDECLARED: {}}))
    assert result.ok is False
    assert result.error == f"invalid arguments: unexpected argument {UNDECLARED!r}"


def test_lenient_mode_matches_across_transports_for_a_call_with_an_undeclared_name() -> None:
    task = _task()
    in_process = LabStockEnv()
    expected = in_process.call(
        in_process.reset(task, 1), _call({"compound_id": "C07", UNDECLARED: {}})
    )
    transport = _mcp_transport(strict=False)
    try:
        handle = transport.reset(task, 1)
        over_mcp = transport.call(handle, _call({"compound_id": "C07", UNDECLARED: {}}))
    finally:
        transport.close()
    assert expected.ok is True
    assert _same(over_mcp) == _same(expected)


def test_strict_mode_rejects_undeclared_names_on_both_transports_and_keeps_valid_calls_equal() -> (
    None
):
    task = _task()
    in_process = LabStockEnv(strict_args=True)
    handle = in_process.reset(task, 1)
    rejected_in_process = in_process.call(handle, _call({"compound_id": "C07", UNDECLARED: {}}))
    valid_in_process = in_process.call(handle, _call({"compound_id": "C07"}))

    transport = _mcp_transport(strict=True)
    try:
        mcp_handle = transport.reset(task, 1)
        rejected_over_mcp = transport.call(
            mcp_handle, _call({"compound_id": "C07", UNDECLARED: {}})
        )
        valid_over_mcp = transport.call(mcp_handle, _call({"compound_id": "C07"}))
    finally:
        transport.close()

    assert rejected_in_process.ok is False and rejected_over_mcp.ok is False
    assert UNDECLARED in (rejected_in_process.error or "")
    assert UNDECLARED in (rejected_over_mcp.error or "")
    assert valid_over_mcp.ok is True and valid_in_process.ok is True
    assert _same(valid_over_mcp) == _same(valid_in_process)


def test_strict_mode_does_not_change_the_control_tools() -> None:
    transport = _mcp_transport(strict=True)
    try:
        listed = {spec.name for spec in transport.tools()}
    finally:
        transport.close()
    assert "get_compound" in listed and "episode_reset" not in listed
