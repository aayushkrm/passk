"""Tests for trace.py: GenAI semconv names, span tree shape, deterministic IDs and atomic writes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from passk import trace
from passk.schema import AgentAction, FailureCategory, Step, ToolCall, ToolResult, Trajectory


def tool_step(index: int, name: str, ok: bool = True) -> Step:
    """Return a tool-call step with a result."""
    call = ToolCall(call_id=f"c{index}", name=name, arguments={"x": index})
    result = ToolResult(
        call_id=call.call_id,
        ok=ok,
        content=None if not ok else {"x": index},
        error=None if ok else "unknown compound",
    )
    return Step(
        index=index,
        action=AgentAction(kind="tool_call", tool_call=call),
        result=result,
        latency_s=0.5,
        input_tokens=100,
        output_tokens=8,
    )


def make_trajectory(steps: list[Step], failure: FailureCategory) -> Trajectory:
    """Return a trajectory around ``steps``."""
    return Trajectory(
        run_id="run-x",
        task_id="t07",
        trial=2,
        seed=12345,
        agent_id="llamacpp:qwen",
        steps=steps,
        final_state={},
        success=failure == FailureCategory.SUCCESS,
        violations=[],
        failure=failure,
        wall_s=2.0,
    )


def final_step(index: int) -> Step:
    """Return a final step."""
    return Step(
        index=index,
        action=AgentAction(kind="final", message="done"),
        result=None,
        latency_s=0.25,
        input_tokens=120,
        output_tokens=5,
    )


def test_semconv_version_is_pinned_and_status_is_development() -> None:
    assert trace.SEMCONV_VERSION == "1.45.0"
    assert trace.SEMCONV_STATUS == "Development"


def test_span_name_helpers_match_spec() -> None:
    assert trace.SPAN_INVOKE_AGENT == "invoke_agent passk"
    assert trace.span_name_chat("qwen2.5-0.5b") == "chat qwen2.5-0.5b"
    assert trace.span_name_tool("place_order") == "execute_tool place_order"


def test_attribute_names_are_the_genai_names() -> None:
    assert trace.ATTR_OPERATION_NAME == "gen_ai.operation.name"
    assert trace.ATTR_PROVIDER_NAME == "gen_ai.provider.name"
    assert trace.ATTR_REQUEST_MODEL == "gen_ai.request.model"
    assert trace.ATTR_TOOL_NAME == "gen_ai.tool.name"
    assert trace.ATTR_TOOL_CALL_ID == "gen_ai.tool.call.id"
    assert trace.ATTR_INPUT_TOKENS == "gen_ai.usage.input_tokens"
    assert trace.ATTR_OUTPUT_TOKENS == "gen_ai.usage.output_tokens"


def test_span_tree_for_tool_then_final_episode() -> None:
    traj = make_trajectory(
        [tool_step(0, "search_compounds"), final_step(1)], FailureCategory.SUCCESS
    )
    spans = trace.build_spans(traj, request_model="qwen", provider_name="llama_cpp")
    names = [s["name"] for s in spans]
    assert names == [
        "invoke_agent passk",
        "chat qwen",
        "execute_tool search_compounds",
        "chat qwen",
    ]
    root, chat0, tool, chat1 = spans
    assert root["parent_span_id"] is None
    assert {s["parent_span_id"] for s in spans[1:]} == {root["span_id"]}
    assert len({s["trace_id"] for s in spans}) == 1
    assert len({s["span_id"] for s in spans}) == len(spans)
    assert chat0["kind"] == "CLIENT"
    assert tool["kind"] == "INTERNAL"
    assert chat1["kind"] == "CLIENT"
    assert tool["attributes"]["gen_ai.tool.name"] == "search_compounds"
    assert tool["attributes"]["gen_ai.tool.call.id"] == "c0"
    assert tool["attributes"]["gen_ai.operation.name"] == "execute_tool"
    assert chat0["attributes"]["gen_ai.operation.name"] == "chat"
    assert chat0["attributes"]["gen_ai.provider.name"] == "llama_cpp"
    assert chat0["attributes"]["gen_ai.request.model"] == "qwen"
    assert chat0["attributes"]["gen_ai.usage.input_tokens"] == 100
    assert chat0["attributes"]["gen_ai.usage.output_tokens"] == 8


def test_root_span_carries_agent_identity_and_totals() -> None:
    traj = make_trajectory(
        [tool_step(0, "search_compounds"), final_step(1)], FailureCategory.SUCCESS
    )
    root = trace.build_spans(traj, request_model="qwen", provider_name="llama_cpp")[0]
    attrs = root["attributes"]
    assert attrs["gen_ai.operation.name"] == "invoke_agent"
    assert attrs["gen_ai.agent.id"] == "llamacpp:qwen"
    assert attrs["gen_ai.usage.input_tokens"] == 220
    assert attrs["gen_ai.usage.output_tokens"] == 13
    assert attrs["passk.task_id"] == "t07"
    assert attrs["passk.trial"] == 2
    assert attrs["passk.seed"] == 12345
    assert attrs["passk.success"] is True
    assert attrs["passk.failure"] == "success"
    assert root["status"]["code"] == "OK"
    assert root["duration_s"] == pytest.approx(2.0)


def test_timeline_offsets_follow_latencies() -> None:
    traj = make_trajectory(
        [tool_step(0, "search_compounds"), final_step(1)], FailureCategory.SUCCESS
    )
    spans = trace.build_spans(
        traj, request_model="qwen", provider_name="p", tool_latencies_s=[0.1, None]
    )
    chat0, tool, chat1 = spans[1], spans[2], spans[3]
    assert chat0["start_offset_s"] == 0.0
    assert chat0["duration_s"] == pytest.approx(0.5)
    assert tool["start_offset_s"] == pytest.approx(0.5)
    assert tool["duration_s"] == pytest.approx(0.1)
    assert chat1["start_offset_s"] == pytest.approx(0.6)


def test_schema_error_step_has_no_tool_span_and_marks_chat_span() -> None:
    call = ToolCall(
        call_id="c0",
        name="__schema_error__",
        arguments={"raw": "oops", "reason": "output is not valid JSON"},
    )
    bad = Step(
        index=0,
        action=AgentAction(kind="tool_call", tool_call=call),
        result=ToolResult(
            call_id="c0", ok=False, content=None, error="schema_error: output is not valid JSON"
        ),
        latency_s=0.3,
    )
    traj = make_trajectory([bad, final_step(1)], FailureCategory.SCHEMA_ERROR)
    spans = trace.build_spans(traj, request_model="qwen", provider_name="p")
    assert all(not s["name"].startswith("execute_tool") for s in spans)
    chat0 = spans[1]
    assert chat0["attributes"]["error.type"] == "schema_error"
    assert chat0["status"]["code"] == "ERROR"
    assert spans[0]["status"]["code"] == "ERROR"
    assert spans[0]["status"]["description"] == "schema_error"


def test_failed_tool_call_is_marked_as_tool_error() -> None:
    traj = make_trajectory(
        [tool_step(0, "get_compound", ok=False), final_step(1)],
        FailureCategory.FALSE_SUCCESS_CLAIM,
    )
    spans = trace.build_spans(traj, request_model="qwen", provider_name="p")
    tool = next(s for s in spans if s["name"].startswith("execute_tool"))
    assert tool["attributes"]["passk.tool.ok"] is False
    assert tool["attributes"]["error.type"] == "tool_error"
    assert tool["status"]["code"] == "ERROR"


def test_missing_token_counts_are_omitted_not_zero() -> None:
    step = Step(
        index=0,
        action=AgentAction(kind="final", message="hi"),
        result=None,
        latency_s=0.1,
    )
    traj = make_trajectory([step], FailureCategory.WRONG_STATE)
    spans = trace.build_spans(traj, request_model="m", provider_name="p")
    assert "gen_ai.usage.input_tokens" not in spans[1]["attributes"]
    assert "gen_ai.usage.input_tokens" not in spans[0]["attributes"]


def test_ids_are_deterministic_and_well_formed() -> None:
    traj = make_trajectory(
        [tool_step(0, "search_compounds"), final_step(1)], FailureCategory.SUCCESS
    )
    first = trace.build_spans(traj, request_model="qwen", provider_name="p")
    second = trace.build_spans(traj, request_model="qwen", provider_name="p")
    assert first == second
    for span in first:
        assert len(span["trace_id"]) == 32
        assert len(span["span_id"]) == 16
        int(span["trace_id"], 16)
        int(span["span_id"], 16)


def test_different_trials_get_different_trace_ids() -> None:
    a = make_trajectory([final_step(0)], FailureCategory.SUCCESS)
    b = a.model_copy(update={"trial": 3})
    ta = trace.build_spans(a, request_model="m", provider_name="p")[0]["trace_id"]
    tb = trace.build_spans(b, request_model="m", provider_name="p")[0]["trace_id"]
    assert ta != tb


def test_spans_serialise_to_json() -> None:
    traj = make_trajectory(
        [tool_step(0, "search_compounds"), final_step(1)], FailureCategory.SUCCESS
    )
    spans = trace.build_spans(traj, request_model="qwen", provider_name="p")
    text = json.dumps(spans)
    assert "invoke_agent passk" in text


def test_span_writer_is_atomic_and_idempotent_close(tmp_path: Path) -> None:
    final = tmp_path / "spans.jsonl"
    writer = trace.SpanWriter(final)
    writer.write({"name": "a", "attributes": {}})
    assert not final.exists()  # visible only after close
    writer.write({"name": "b", "attributes": {"k": 1}})
    writer.close()
    writer.close()  # second call is a no-op
    assert final.exists()
    assert not (tmp_path / "spans.jsonl.tmp").exists()
    lines = [json.loads(x) for x in final.read_text().splitlines()]
    assert [line["name"] for line in lines] == ["a", "b"]


def test_span_writer_context_manager_closes_on_exit(tmp_path: Path) -> None:
    final = tmp_path / "s.jsonl"
    span: dict[str, Any] = {"name": "chat m"}
    with trace.SpanWriter(final) as writer:
        writer.write(span)
    assert json.loads(final.read_text()) == span
