"""Span emission as JSONL with OpenTelemetry GenAI semantic-convention attribute names.

No OTel SDK dependency. The names below are the single source of truth. The semconv
version is pinned here; the GenAI conventions are in Development status.
"""

from pathlib import Path
from typing import Any

SEMCONV_VERSION = "1.45.0"
SEMCONV_STATUS = "Development"

ATTR_OPERATION_NAME = "gen_ai.operation.name"
ATTR_PROVIDER_NAME = "gen_ai.provider.name"
ATTR_REQUEST_MODEL = "gen_ai.request.model"
ATTR_TOOL_NAME = "gen_ai.tool.name"
ATTR_TOOL_CALL_ID = "gen_ai.tool.call.id"
ATTR_INPUT_TOKENS = "gen_ai.usage.input_tokens"
ATTR_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"

OP_INVOKE_AGENT = "invoke_agent"
OP_CHAT = "chat"
OP_EXECUTE_TOOL = "execute_tool"

SPAN_INVOKE_AGENT = "invoke_agent passk"


def span_name_chat(model: str) -> str:
    """Return the span name for a model call, ``chat {model}``."""
    return f"{OP_CHAT} {model}"


def span_name_tool(tool: str) -> str:
    """Return the span name for a tool call, ``execute_tool {tool}``."""
    return f"{OP_EXECUTE_TOOL} {tool}"


class SpanWriter:
    """Append-only JSONL writer for spans."""

    def __init__(self, path: Path) -> None:
        """Open ``path`` for writing (truncating any existing file).

        Args:
            path: Destination JSONL file, usually ``runs/<run_id>/spans.jsonl``.
        """
        raise NotImplementedError("SpanWriter: scheduled for work package P-D")

    def write(self, span: dict[str, Any]) -> None:
        """Append one span as a JSON object on its own line.

        Args:
            span: Span record with ``name``, ``kind``, timings and GenAI attributes.
        """
        raise NotImplementedError("SpanWriter.write: scheduled for work package P-D")

    def close(self) -> None:
        """Flush and close the file."""
        raise NotImplementedError("SpanWriter.close: scheduled for work package P-D")
