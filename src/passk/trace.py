"""Span emission as JSONL with OpenTelemetry GenAI semantic-convention attribute names.

No OTel SDK dependency. The names below are the single source of truth. The semconv
version is pinned here; the GenAI conventions are in Development status.

Span record (one JSON object per line in ``spans.jsonl``)::

    {"trace_id", "span_id", "parent_span_id", "name", "kind", "start_offset_s",
     "duration_s", "status": {"code", "description"}, "attributes": {...}}

Times are offsets from the start of the episode, because trajectories hold durations, not
wall-clock timestamps. IDs are derived from (run_id, task_id, trial) with SHA-256, so
re-running the same episode gives the same IDs.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Self

from passk.schema import FailureCategory, Trajectory

SEMCONV_VERSION = "1.45.0"
SEMCONV_STATUS = "Development"

ATTR_OPERATION_NAME = "gen_ai.operation.name"
ATTR_PROVIDER_NAME = "gen_ai.provider.name"
ATTR_REQUEST_MODEL = "gen_ai.request.model"
ATTR_RESPONSE_MODEL = "gen_ai.response.model"
ATTR_AGENT_ID = "gen_ai.agent.id"
ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_TOOL_CALL_ID = "gen_ai.tool.call.id"
ATTR_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
ATTR_ERROR_TYPE = "error.type"

ATTR_PASSK_TASK_ID = "passk.task_id"
ATTR_PASSK_TRIAL = "passk.trial"
ATTR_PASSK_SEED = "passk.seed"
ATTR_PASSK_TURN = "passk.turn"
ATTR_PASSK_SUCCESS = "passk.success"
ATTR_PASSK_FAILURE = "passk.failure"
ATTR_PASSK_TOOL_OK = "passk.tool.ok"

OP_INVOKE_AGENT = "invoke_agent"
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"

SPAN_INVOKE_AGENT = "invoke_agent passk"

KIND_INTERNAL = "INTERNAL"
KIND_CLIENT = "CLIENT"
STATUS_OK = "OK"
STATUS_ERROR = "ERROR"

ERROR_TYPE_SCHEMA = "schema_error"
ERROR_TYPE_TOOL = "tool_error"

SCHEMA_ERROR_PREFIX = "schema_error"


def span_name_chat(model: str) -> str:
    """Return the span name for a model call, ``chat {model}``."""
    return f"{OP_CHAT} {model}"


def span_name_tool(tool: str) -> str:
    """Return the span name for a tool call, ``execute_tool {tool}``."""
    return f"{OP_EXECUTE_TOOL} {tool}"


def _digest(text: str, hex_len: int) -> str:
    """Return the first ``hex_len`` hex characters of the SHA-256 of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:hex_len]


def _span(
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    name: str,
    kind: str,
    start: float,
    duration: float | None,
    status: str,
    description: str | None,
    attributes: dict[str, Any],
) -> dict[str, Any]:
    """Assemble one span record."""
    return {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "kind": kind,
        "start_offset_s": round(start, 6),
        "duration_s": None if duration is None else round(duration, 6),
        "status": {"code": status, "description": description},
        "attributes": attributes,
    }


def build_spans(
    trajectory: Trajectory,
    *,
    request_model: str,
    provider_name: str,
    tool_latencies_s: Sequence[float | None] | None = None,
) -> list[dict[str, Any]]:
    """Build the span tree for one episode.

    Layout: one ``invoke_agent passk`` root (INTERNAL). Under it, one ``chat {model}`` span
    (CLIENT) per step, and one ``execute_tool {tool}`` span (INTERNAL) for each real tool
    call. Schema-error steps get no tool span; their chat span carries
    ``error.type=schema_error``. Message content is never recorded (content capture is
    opt-in in the spec and off here).

    Args:
        trajectory: The episode record.
        request_model: Model name for the ``chat`` span name and ``gen_ai.request.model``.
        provider_name: Value for ``gen_ai.provider.name``.
        tool_latencies_s: Optional tool-call durations, one per step (None if unknown).
            Missing values count as zero for the timeline.

    Returns:
        Span records in parent-before-child order.
    """
    trace_id = _digest(f"{trajectory.run_id}|{trajectory.task_id}|{trajectory.trial}", 32)
    root_id = _digest(f"{trace_id}|root", 16)
    spans: list[dict[str, Any]] = []
    cursor = 0.0
    in_total: int | None = None
    out_total: int | None = None
    for position, step in enumerate(trajectory.steps):
        if step.input_tokens is not None:
            in_total = (in_total or 0) + step.input_tokens
        if step.output_tokens is not None:
            out_total = (out_total or 0) + step.output_tokens

        chat_attrs: dict[str, Any] = {
            ATTR_OPERATION_NAME: OP_CHAT,
            ATTR_PROVIDER_NAME: provider_name,
            ATTR_REQUEST_MODEL: request_model,
            ATTR_PASSK_TURN: step.index,
        }
        if step.input_tokens is not None:
            chat_attrs[ATTR_INPUT_TOKENS] = step.input_tokens
        if step.output_tokens is not None:
            chat_attrs[ATTR_OUTPUT_TOKENS] = step.output_tokens
        result = step.result
        schema_failed = result is not None and (result.error or "").startswith(SCHEMA_ERROR_PREFIX)
        if schema_failed:
            chat_attrs[ATTR_ERROR_TYPE] = ERROR_TYPE_SCHEMA
        chat_id = _digest(f"{trace_id}|{step.index}|chat", 16)
        spans.append(
            _span(
                trace_id=trace_id,
                span_id=chat_id,
                parent_span_id=root_id,
                name=span_name_chat(request_model),
                kind=KIND_CLIENT,
                start=cursor,
                duration=step.latency_s,
                status=STATUS_ERROR if schema_failed else STATUS_OK,
                description=result.error if (schema_failed and result) else None,
                attributes=chat_attrs,
            )
        )
        cursor += step.latency_s

        call = step.action.tool_call
        if step.action.kind != "tool_call" or call is None or schema_failed:
            continue
        tool_s = None
        if tool_latencies_s is not None and position < len(tool_latencies_s):
            tool_s = tool_latencies_s[position]
        tool_ok = bool(result.ok) if result is not None else False
        tool_attrs: dict[str, Any] = {
            ATTR_OPERATION_NAME: OP_EXECUTE_TOOL,
            ATTR_TOOL_NAME: call.name,
            ATTR_TOOL_CALL_ID: call.call_id,
            ATTR_PASSK_TOOL_OK: tool_ok,
            ATTR_PASSK_TURN: step.index,
        }
        if not tool_ok:
            tool_attrs[ATTR_ERROR_TYPE] = ERROR_TYPE_TOOL
        spans.append(
            _span(
                trace_id=trace_id,
                span_id=_digest(f"{trace_id}|{step.index}|tool", 16),
                parent_span_id=root_id,
                name=span_name_tool(call.name),
                kind=KIND_INTERNAL,
                start=cursor,
                duration=tool_s,
                status=STATUS_OK if tool_ok else STATUS_ERROR,
                description=(result.error if result is not None else None),
                attributes=tool_attrs,
            )
        )
        cursor += tool_s or 0.0

    success = trajectory.failure == FailureCategory.SUCCESS
    root_attrs: dict[str, Any] = {
        ATTR_OPERATION_NAME: OP_INVOKE_AGENT,
        ATTR_PROVIDER_NAME: provider_name,
        ATTR_AGENT_ID: trajectory.agent_id,
        ATTR_REQUEST_MODEL: request_model,
        ATTR_PASSK_TASK_ID: trajectory.task_id,
        ATTR_PASSK_TRIAL: trajectory.trial,
        ATTR_PASSK_SEED: trajectory.seed,
        ATTR_PASSK_SUCCESS: success,
        ATTR_PASSK_FAILURE: trajectory.failure.value,
    }
    if in_total is not None:
        root_attrs[ATTR_INPUT_TOKENS] = in_total
    if out_total is not None:
        root_attrs[ATTR_OUTPUT_TOKENS] = out_total
    root = _span(
        trace_id=trace_id,
        span_id=root_id,
        parent_span_id=None,
        name=SPAN_INVOKE_AGENT,
        kind=KIND_INTERNAL,
        start=0.0,
        duration=trajectory.wall_s,
        status=STATUS_OK if success else STATUS_ERROR,
        description=None if success else trajectory.failure.value,
        attributes=root_attrs,
    )
    return [root, *spans]


class SpanWriter:
    """Append-only JSONL writer for spans, atomic on close.

    Lines go to ``<path>.tmp`` while the writer is open. ``close`` renames the file onto
    ``path``, so readers never see a partial file.
    """

    def __init__(self, path: Path) -> None:
        """Open a temporary file beside ``path`` (truncating any previous temporary file).

        Args:
            path: Destination JSONL file, usually ``runs/<run_id>/spans.jsonl``.
        """
        self._path = Path(path)
        self._tmp = self._path.with_name(self._path.name + ".tmp")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._tmp.open("w", encoding="utf-8")

    def write(self, span: dict[str, Any]) -> None:
        """Append one span as a JSON object on its own line.

        Args:
            span: Span record with ``name``, ``kind``, timings and GenAI attributes.
        """
        self._fh.write(json.dumps(span, sort_keys=True) + "\n")

    def close(self) -> None:
        """Flush, close and move the file onto its final path. Safe to call twice."""
        if self._fh.closed:
            return
        self._fh.flush()
        self._fh.close()
        os.replace(self._tmp, self._path)

    def __enter__(self) -> Self:
        """Return the writer for use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the writer on exit from the ``with`` block."""
        self.close()
