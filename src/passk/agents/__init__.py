"""Agents that choose actions for an episode.

``scripted`` holds deterministic agents used for tests and calibration. ``llamacpp`` holds the
local model agent and imports ``llama_cpp`` lazily. ``openai_compat`` holds the optional
hosted-model agent (stdlib HTTP only). Shared prompt, schema and parsing helpers live in the
private ``_common`` module.
"""

from passk.agents._common import (
    SCHEMA_ERROR_TOOL,
    ModelCallInfo,
    PromptVariant,
    action_json_schema,
    build_messages,
    parse_action,
    schema_error_action,
    schema_error_result,
    system_prompt,
)
from passk.agents.llamacpp import LlamaCppAgent
from passk.agents.openai_compat import OpenAICompatAgent, ProviderError

__all__ = [
    "SCHEMA_ERROR_TOOL",
    "LlamaCppAgent",
    "ModelCallInfo",
    "OpenAICompatAgent",
    "PromptVariant",
    "ProviderError",
    "action_json_schema",
    "build_messages",
    "parse_action",
    "schema_error_action",
    "schema_error_result",
    "system_prompt",
]
