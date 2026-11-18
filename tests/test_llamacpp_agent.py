"""Tests for the llama-cpp agent. A fake ``Llama`` is injected; no native build is needed."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from passk.agents import SCHEMA_ERROR_TOOL, action_json_schema, build_messages
from passk.agents.llamacpp import LlamaCppAgent
from passk.schema import AgentAction, Step, Task, ToolCall, ToolResult, ToolSpec

TOOLS = [
    ToolSpec(
        name="search_compounds",
        description="Search compounds by name. Returns ids. Use before ordering.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    ),
    ToolSpec(
        name="place_order",
        description="Order grams of a compound. Controlled compounds need approval.",
        parameters={
            "type": "object",
            "properties": {
                "compound_id": {"type": "string"},
                "grams": {"type": "number"},
            },
        },
    ),
]


def make_task() -> Task:
    """Return a minimal task."""
    return Task(
        id="t01",
        tier=1,
        instruction="Order 5 g of compound C07.",
        initial_state={},
        goal=[],
        forbidden=[],
        reference_solution=[],
        max_turns=4,
    )


class FakeLlama:
    """Stands in for ``llama_cpp.Llama``. Returns canned chat completions."""

    instances: list[FakeLlama] = []

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.replies: list[str] = []
        FakeLlama.instances.append(self)

    def create_chat_completion(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        text = self.replies.pop(0)
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 17},
        }


@pytest.fixture(autouse=True)
def _reset_fake() -> None:
    FakeLlama.instances = []


def make_agent(replies: list[str], **kwargs: Any) -> tuple[LlamaCppAgent, FakeLlama]:
    """Build an agent around a fresh fake model that will return ``replies``."""
    holder: dict[str, FakeLlama] = {}

    def factory(**init: Any) -> FakeLlama:
        model = FakeLlama(**init)
        model.replies = list(replies)
        holder["model"] = model
        return model

    agent = LlamaCppAgent(
        Path("/models/qwen2.5-0.5b-instruct-q4_k_m.gguf"), llama_factory=factory, **kwargs
    )
    return agent, holder["model"]


def test_constructor_runs_cpu_only_and_skips_system_info() -> None:
    _, model = make_agent([])
    assert model.init_kwargs["n_gpu_layers"] == 0
    assert model.init_kwargs["verbose"] is False
    assert model.init_kwargs["n_ctx"] == 4096
    assert model.init_kwargs["n_threads"] == 4
    assert model.init_kwargs["model_path"].endswith("qwen2.5-0.5b-instruct-q4_k_m.gguf")


def test_missing_llama_cpp_raises_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    with pytest.raises(ImportError, match="llama-cpp-python"):
        LlamaCppAgent(Path("/models/x.gguf"))


def test_act_returns_tool_call_and_records_tokens() -> None:
    reply = json.dumps(
        {"kind": "tool_call", "tool": "search_compounds", "arguments": {"query": "C07"}}
    )
    agent, _ = make_agent([reply])
    action = agent.act(make_task(), [], TOOLS)
    assert action.kind == "tool_call"
    assert action.tool_call is not None
    assert action.tool_call.name == "search_compounds"
    assert action.tool_call.arguments == {"query": "C07"}
    assert action.tool_call.call_id == "c0"
    assert agent.last_call is not None
    assert agent.last_call.input_tokens == 120
    assert agent.last_call.output_tokens == 17
    assert agent.last_call.latency_s >= 0.0


def test_final_action_parses() -> None:
    agent, _ = make_agent([json.dumps({"kind": "final", "message": "Done."})])
    action = agent.act(make_task(), [], TOOLS)
    assert action.kind == "final"
    assert action.message == "Done."


def test_invalid_json_becomes_schema_error_action_not_exception() -> None:
    agent, _ = make_agent(["this is not json"])
    action = agent.act(make_task(), [], TOOLS)
    assert action.kind == "tool_call"
    assert action.tool_call is not None
    assert action.tool_call.name == SCHEMA_ERROR_TOOL
    assert "not valid JSON" in action.tool_call.arguments["reason"]


def test_unknown_tool_is_schema_error() -> None:
    reply = json.dumps({"kind": "tool_call", "tool": "drop_database", "arguments": {}})
    agent, _ = make_agent([reply])
    action = agent.act(make_task(), [], TOOLS)
    assert action.tool_call is not None
    assert action.tool_call.name == SCHEMA_ERROR_TOOL
    assert "unknown tool" in action.tool_call.arguments["reason"]


def test_response_format_carries_flat_schema_with_tool_enum() -> None:
    agent, model = make_agent([json.dumps({"kind": "final", "message": "ok"})])
    agent.act(make_task(), [], TOOLS)
    call = model.calls[0]
    response_format = call["response_format"]
    assert response_format["type"] == "json_object"
    schema = response_format["schema"]
    assert schema["properties"]["tool"]["enum"] == ["search_compounds", "place_order"]
    assert set(schema["properties"]) == {"kind", "tool", "arguments", "message"}
    assert schema["properties"]["kind"]["enum"] == ["tool_call", "final"]
    assert call["temperature"] == 0.7


def test_per_call_seed_follows_episode_seed_and_history_length() -> None:
    replies = [json.dumps({"kind": "final", "message": "x"})] * 2
    agent, model = make_agent(replies)
    agent.begin_episode(1000)
    agent.act(make_task(), [], TOOLS)
    step = Step(
        index=0,
        action=AgentAction(
            kind="tool_call",
            tool_call=ToolCall(call_id="c0", name="search_compounds", arguments={}),
        ),
        result=ToolResult(call_id="c0", ok=True, content=[]),
        latency_s=0.1,
    )
    agent.act(make_task(), [step], TOOLS)
    assert model.calls[0]["seed"] == 1000
    assert model.calls[1]["seed"] == 1001


def test_policy_section_only_in_v1() -> None:
    v1 = build_messages(make_task(), [], TOOLS, "v1")[0]["content"]
    v2 = build_messages(make_task(), [], TOOLS, "v2")[0]["content"]
    assert "Policy:" in v1
    assert "request_approval" in v1
    assert "budget" in v1
    assert "Policy:" not in v2
    assert "request_approval" not in v2


def test_v2_tool_descriptions_are_terse() -> None:
    v1 = build_messages(make_task(), [], TOOLS, "v1")[0]["content"]
    v2 = build_messages(make_task(), [], TOOLS, "v2")[0]["content"]
    assert "Use before ordering." in v1
    assert "Use before ordering." not in v2
    assert "- search_compounds(query): Search compounds by name" in v2


def test_history_is_rendered_as_assistant_then_tool_result() -> None:
    action = AgentAction(
        kind="tool_call",
        tool_call=ToolCall(call_id="c0", name="search_compounds", arguments={"query": "C07"}),
    )
    result = ToolResult(call_id="c0", ok=True, content=[{"compound_id": "C07"}])
    step = Step(index=0, action=action, result=result, latency_s=0.2)
    messages = build_messages(make_task(), [step], TOOLS, "v1")
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert json.loads(messages[2]["content"])["tool"] == "search_compounds"
    assert json.loads(messages[3]["content"])["ok"] is True


def test_schema_error_step_shows_raw_output_to_model() -> None:
    action = AgentAction(
        kind="tool_call",
        tool_call=ToolCall(
            call_id="c0",
            name=SCHEMA_ERROR_TOOL,
            arguments={"raw": "oops {", "reason": "output is not valid JSON"},
        ),
    )
    result = ToolResult(
        call_id="c0", ok=False, content=None, error="schema_error: output is not valid JSON"
    )
    step = Step(index=0, action=action, result=result, latency_s=0.2)
    messages = build_messages(make_task(), [step], TOOLS, "v2")
    assert messages[2]["content"] == "oops {"
    assert "schema_error" in messages[3]["content"]


def test_action_json_schema_rejects_empty_tool_list() -> None:
    with pytest.raises(ValueError):
        action_json_schema([])


@pytest.mark.model
def test_real_gguf_smoke() -> None:
    """Loads a real GGUF and asks for one action. Needs PASSK_TEST_GGUF and llama-cpp-python."""
    import os

    pytest.importorskip("llama_cpp")
    gguf = os.environ.get("PASSK_TEST_GGUF")
    if not gguf or not Path(gguf).is_file():
        pytest.skip("set PASSK_TEST_GGUF to a local GGUF file")
    agent = LlamaCppAgent(Path(gguf), n_ctx=2048, temperature=0.0, seed=0)
    action = agent.act(make_task(), [], TOOLS)
    assert action.kind in {"tool_call", "final"}
    assert agent.last_call is not None
