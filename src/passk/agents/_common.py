"""Helpers shared by the model agents: prompts, action schema, parsing and error actions.

Private module. The public names are re-exported from ``passk.agents``.

Schema-error policy (decided here, used by the runner):
    A model output that is not a valid action (not JSON, unknown tool, wrong field types)
    does not end the episode. The agent returns a ``tool_call`` action whose tool name is
    the reserved ``SCHEMA_ERROR_TOOL``. The runner does not send it to the environment; it
    records a ``ToolResult`` with ``ok=False`` and ``error="schema_error: <reason>"``. The
    model sees that result on its next turn and may recover. The turn budget still applies,
    and the taxonomy ranks ``schema_error`` above ``max_turns``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from passk.schema import AgentAction, Step, Task, ToolCall, ToolResult, ToolSpec

PromptVariant = Literal["v1", "v2"]

SCHEMA_ERROR_TOOL = "__schema_error__"
"""Reserved tool name that marks an agent output that failed validation."""

NO_TOOL = "none"
"""Value of the ``tool`` field in a ``final`` action."""

SCHEMA_ERROR_PREFIX = "schema_error: "
"""Prefix of ``ToolResult.error`` for a schema-error step."""

MAX_RAW_CHARS = 2000
"""Longest model output kept in a schema-error action (characters)."""

POLICY_RULES: tuple[str, ...] = (
    "A controlled compound needs an approval first: call request_approval for that "
    "compound before place_order orders it.",
    "The budget cannot be exceeded: check get_budget and the price per gram before place_order.",
)
"""Policy section included in prompt variant ``v1`` only."""


@dataclass(frozen=True)
class ModelCallInfo:
    """Accounting for one model call, read by the runner after ``act``.

    Attributes:
        input_tokens: Prompt tokens, or None if the backend did not report them.
        output_tokens: Generated tokens, or None if not reported.
        latency_s: Wall time of the model call alone, in seconds.
        model_reported: Model id string reported by the provider, if any.
    """

    input_tokens: int | None
    output_tokens: int | None
    latency_s: float
    model_reported: str | None = None


def action_json_schema(tool_names: Sequence[str]) -> dict[str, Any]:
    """Build the flat JSON schema that constrains the model's action output.

    Shape: ``{"kind": "tool_call" | "final", "tool": enum(tool_names + "none"),
    "arguments": object, "message": string}``. All four keys are required. For a ``final``
    action the model writes ``"tool": "none"``, ``"arguments": {}`` and the reply in
    ``message``. For a ``tool_call`` it writes the tool name, its arguments and an empty
    ``message``. Kept flat to limit grammar compile time. ``parse_action`` checks the values.

    Args:
        tool_names: Names of the tools the model may call.

    Returns:
        A JSON schema object.

    Raises:
        ValueError: If ``tool_names`` is empty.
    """
    if not tool_names:
        raise ValueError("action_json_schema needs at least one tool name")
    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["tool_call", "final"]},
            "tool": {"type": "string", "enum": [*tool_names, NO_TOOL]},
            "arguments": {"type": "object"},
            "message": {"type": "string"},
        },
        "required": ["kind", "tool", "arguments", "message"],
        "additionalProperties": False,
    }


def _param_names(spec: ToolSpec) -> list[str]:
    """Return the parameter names declared in a tool's JSON schema."""
    properties = spec.parameters.get("properties", {})
    return list(properties) if isinstance(properties, dict) else []


def render_tools(tools: Sequence[ToolSpec]) -> str:
    """Render the tool list for the system prompt, with each full description.

    The two prompt variants differ only in the policy section, so this is the same for v1
    and v2.

    Args:
        tools: Tools available to the agent.

    Returns:
        One line per tool, ``- name(param, ...): description``.
    """
    lines: list[str] = []
    for spec in tools:
        params = ", ".join(_param_names(spec))
        lines.append(f"- {spec.name}({params}): {spec.description.strip()}")
    return "\n".join(lines)


def system_prompt(tools: Sequence[ToolSpec], prompt_variant: PromptVariant) -> str:
    """Build the system message: output format, tool list and (v1 only) the policy section.

    Args:
        tools: Tools available to the agent.
        prompt_variant: ``v1`` adds the policy section; ``v2`` is identical without it.

    Returns:
        The system message text.
    """
    parts = [
        "You operate a lab inventory through tools. Reply with exactly one JSON object "
        "and no other text.",
        'Fields, all four required: "kind" is "tool_call" to call a tool or "final" to '
        'finish. "tool" is the tool name for tool_call, and "none" for final. "arguments" '
        'is a JSON object with the tool arguments, and {} for final. "message" is empty for '
        "tool_call and is your final reply to the user for final.",
        "Tools:",
        render_tools(tools),
    ]
    if prompt_variant == "v1":
        parts.append("Policy:")
        parts.extend(f"- {rule}" for rule in POLICY_RULES)
    return "\n".join(parts)


def _render_assistant(action: AgentAction) -> str:
    """Render a past action as the assistant message that produced it."""
    if action.kind == "final":
        return json.dumps(
            {"kind": "final", "tool": NO_TOOL, "arguments": {}, "message": action.message or ""}
        )
    call = action.tool_call
    if call is None:
        return json.dumps({"kind": "tool_call", "tool": NO_TOOL, "arguments": {}, "message": ""})
    if call.name == SCHEMA_ERROR_TOOL:
        raw = call.arguments.get("raw", "")
        return str(raw)
    return json.dumps(
        {"kind": "tool_call", "tool": call.name, "arguments": call.arguments, "message": ""}
    )


def _render_result(result: ToolResult) -> str:
    """Render a tool result as the user message that follows the call."""
    return json.dumps(
        {"ok": result.ok, "content": result.content, "error": result.error},
        default=str,
        sort_keys=True,
    )


def build_messages(
    task: Task,
    history: Sequence[Step],
    tools: Sequence[ToolSpec],
    prompt_variant: PromptVariant,
) -> list[dict[str, str]]:
    """Render the chat messages for one model call.

    Layout: system (format, tools, policy in v1), user (the task instruction), then for
    each past step an assistant message holding the action JSON followed by a user message
    holding the tool result. Schema-error steps show the model's raw output and the error.

    Args:
        task: Task being solved.
        history: Steps taken so far.
        tools: Tools available to the agent.
        prompt_variant: ``v1`` includes the policy section; ``v2`` is identical without it.

    Returns:
        Chat messages in the OpenAI-style ``{"role", "content"}`` format.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt(tools, prompt_variant)},
        {"role": "user", "content": task.instruction},
    ]
    for step in history:
        messages.append({"role": "assistant", "content": _render_assistant(step.action)})
        if step.result is not None:
            messages.append({"role": "user", "content": _render_result(step.result)})
    return messages


def schema_error_action(call_id: str, raw: str, reason: str) -> AgentAction:
    """Build the recoverable action that stands for an invalid model output.

    Args:
        call_id: Identifier for the synthetic call (unique within the episode).
        raw: The model output that failed validation (truncated to ``MAX_RAW_CHARS``).
        reason: Short human-readable reason.

    Returns:
        A ``tool_call`` action naming ``SCHEMA_ERROR_TOOL``.
    """
    return AgentAction(
        kind="tool_call",
        tool_call=ToolCall(
            call_id=call_id,
            name=SCHEMA_ERROR_TOOL,
            arguments={"raw": raw[:MAX_RAW_CHARS], "reason": reason},
        ),
    )


def schema_error_result(call_id: str, reason: str) -> ToolResult:
    """Build the failed tool result the runner records for a schema-error action.

    Args:
        call_id: Identifier of the synthetic call.
        reason: Short human-readable reason.

    Returns:
        A ``ToolResult`` with ``ok=False`` and ``error`` starting with ``schema_error:``.
    """
    return ToolResult(
        call_id=call_id,
        ok=False,
        content=None,
        error=f"{SCHEMA_ERROR_PREFIX}{reason}",
    )


def _load_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse ``text`` as one JSON object.

    Tries the whole text first, then the span from the first ``{`` to the last ``}``
    (models often wrap JSON in prose or code fences).

    Returns:
        ``(object, "")`` on success, or ``(None, reason)``.
    """
    stripped = text.strip()
    candidates = [stripped]
    start, end = stripped.find("{"), stripped.rfind("}")
    if 0 <= start < end:
        candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed, ""
        return None, "output is not a JSON object"
    return None, "output is not valid JSON"


def parse_action(text: str, tool_names: Sequence[str], call_id: str) -> AgentAction:
    """Validate one model output and convert it to an ``AgentAction``.

    Never raises for bad model output. Invalid output becomes ``schema_error_action``.

    Args:
        text: Raw model output.
        tool_names: Tool names the agent may call.
        call_id: Identifier for the resulting call (unique within the episode).

    Returns:
        A ``final`` action, a ``tool_call`` action, or a schema-error action.
    """
    payload, reason = _load_object(text)
    if payload is None:
        return schema_error_action(call_id, text, reason)
    kind = payload.get("kind")
    if kind == "final":
        message = payload.get("message")
        if message is None:
            message = ""
        if not isinstance(message, str):
            return schema_error_action(call_id, text, "message must be a string")
        return AgentAction(kind="final", message=message)
    if kind == "tool_call":
        tool = payload.get("tool")
        if not isinstance(tool, str) or tool not in tool_names:
            return schema_error_action(call_id, text, f"unknown tool {tool!r}")
        arguments = payload.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            return schema_error_action(call_id, text, "arguments must be a JSON object")
        return AgentAction(
            kind="tool_call",
            tool_call=ToolCall(call_id=call_id, name=tool, arguments=arguments),
        )
    return schema_error_action(call_id, text, "kind must be tool_call or final")


def content_and_usage(response: dict[str, Any]) -> tuple[str, int | None, int | None]:
    """Read the assistant text and token counts from an OpenAI-style chat response.

    Args:
        response: Parsed response object with ``choices`` and optional ``usage``.

    Returns:
        ``(text, prompt_tokens, completion_tokens)``. Text is ``""`` when absent.
    """
    choices = response.get("choices") or []
    text = ""
    if choices and isinstance(choices[0], dict):
        message = choices[0].get("message") or {}
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, str):
            text = content
    usage = response.get("usage") or {}
    prompt = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
    return (
        text,
        prompt if isinstance(prompt, int) else None,
        completion if isinstance(completion, int) else None,
    )
