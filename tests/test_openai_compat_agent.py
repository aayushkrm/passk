"""Tests for the OpenAI-compatible agent with an injected fake ``post``. No network.

The fake API key below is a test value. Every test checks that it never appears in an
exception string, a repr, the agent metadata or a file written by the runner.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from passk.agents import ProviderError
from passk.agents.openai_compat import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    HttpResponse,
    OpenAICompatAgent,
    urllib_post,
)
from passk.schema import Task, ToolSpec

TEST_KEY = "sk-or-test-SECRET-0123456789"
MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"
TOOLS = [
    ToolSpec(
        name="get_budget",
        description="Return the remaining budget.",
        parameters={"type": "object", "properties": {}},
    )
]


@pytest.fixture(autouse=True)
def _set_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DEFAULT_API_KEY_ENV, TEST_KEY)


def make_task() -> Task:
    """Return a minimal task."""
    return Task(
        id="t02",
        tier=2,
        instruction="Report the remaining budget.",
        initial_state={},
        goal=[],
        forbidden=[],
        reference_solution=[],
        max_turns=3,
    )


def ok_body(content: str, model: str = MODEL) -> bytes:
    """Return a successful chat completion body."""
    return json.dumps(
        {
            "model": model,
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 90, "completion_tokens": 12},
        }
    ).encode()


class FakePost:
    """Records calls and returns scripted responses or raises scripted exceptions."""

    def __init__(self, script: list[HttpResponse | BaseException]) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self, url: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> HttpResponse:
        self.calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class Clock:
    """Fake monotonic clock whose sleep advances it."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_agent(post: FakePost, clock: Clock | None = None, **kwargs: Any) -> OpenAICompatAgent:
    """Build an agent wired to the fake post and fake clock."""
    clock = clock or Clock()
    params: dict[str, Any] = {
        "model": MODEL,
        "min_interval_s": 0.0,
        "backoff_base_s": 2.0,
        "backoff_max_s": 60.0,
        "post": post,
        "sleep": clock.sleep,
        "clock": clock,
    }
    params.update(kwargs)
    return OpenAICompatAgent(**params)


def assert_no_key(*texts: str) -> None:
    """Fail if the test key appears in any of ``texts``."""
    for text in texts:
        assert TEST_KEY not in text


def test_defaults_point_at_openrouter_and_key_env() -> None:
    agent = make_agent(FakePost([]))
    assert agent.base_url == DEFAULT_BASE_URL == "https://openrouter.ai/api/v1"
    assert agent.api_key_env == "OPENROUTER_API_KEY"
    assert agent.agent_id == f"openai_compat:{MODEL}"


def test_missing_key_raises_without_printing_any_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(DEFAULT_API_KEY_ENV)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY") as info:
        make_agent(FakePost([]))
    assert TEST_KEY not in str(info.value)


def test_success_parses_action_and_records_reported_model() -> None:
    post = FakePost(
        [HttpResponse(200, {}, ok_body(json.dumps({"kind": "final", "message": "Budget 40."})))]
    )
    agent = make_agent(post)
    action = agent.act(make_task(), [], TOOLS)
    assert action.kind == "final"
    assert action.message == "Budget 40."
    assert agent.reported_models == [MODEL]
    assert agent.metadata["model_reported"] == [MODEL]
    assert agent.last_call is not None
    assert agent.last_call.input_tokens == 90
    assert agent.last_call.output_tokens == 12
    assert agent.last_call.model_reported == MODEL


def test_request_carries_bearer_key_and_json_mode_body() -> None:
    post = FakePost([HttpResponse(200, {}, ok_body(json.dumps({"kind": "final", "message": "x"})))])
    agent = make_agent(post, temperature=0.3, max_tokens=99)
    agent.begin_episode(777)
    agent.act(make_task(), [], TOOLS)
    call = post.calls[0]
    assert call["url"] == f"{DEFAULT_BASE_URL}/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {TEST_KEY}"
    body = json.loads(call["body"])
    assert body["model"] == MODEL
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 99
    assert body["seed"] == 777


def test_429_then_success_retries_with_exponential_backoff() -> None:
    post = FakePost(
        [
            HttpResponse(429, {}, b'{"error":{"message":"rate limited"}}'),
            HttpResponse(429, {}, b""),
            HttpResponse(200, {}, ok_body(json.dumps({"kind": "final", "message": "ok"}))),
        ]
    )
    clock = Clock()
    agent = make_agent(post, clock=clock)
    action = agent.act(make_task(), [], TOOLS)
    assert action.kind == "final"
    assert len(post.calls) == 3
    # attempt 0 -> 2 s, attempt 1 -> 4 s
    assert clock.sleeps == [2.0, 4.0]


def test_retry_after_header_takes_precedence_over_backoff() -> None:
    post = FakePost(
        [
            HttpResponse(429, {"retry-after": "7"}, b""),
            HttpResponse(200, {}, ok_body(json.dumps({"kind": "final", "message": "ok"}))),
        ]
    )
    clock = Clock()
    make_agent(post, clock=clock).act(make_task(), [], TOOLS)
    assert clock.sleeps == [7.0]


def test_unparseable_retry_after_falls_back_to_backoff() -> None:
    post = FakePost(
        [
            HttpResponse(503, {"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, b""),
            HttpResponse(200, {}, ok_body(json.dumps({"kind": "final", "message": "ok"}))),
        ]
    )
    clock = Clock()
    make_agent(post, clock=clock).act(make_task(), [], TOOLS)
    assert clock.sleeps == [2.0]


def test_5xx_exhausts_retries_and_raises_provider_error() -> None:
    post = FakePost([HttpResponse(502, {}, b"") for _ in range(3)])
    clock = Clock()
    agent = make_agent(post, clock=clock, max_retries=2)
    with pytest.raises(ProviderError, match="HTTP 502") as info:
        agent.act(make_task(), [], TOOLS)
    assert len(post.calls) == 3
    assert clock.sleeps == [2.0, 4.0]
    assert info.value.__cause__ is None
    assert_no_key(str(info.value), repr(info.value))


def test_non_retryable_401_raises_once_and_scrubs_echoed_key() -> None:
    echoed = json.dumps({"error": {"message": f"invalid key {TEST_KEY}"}}).encode()
    post = FakePost([HttpResponse(401, {}, echoed)])
    clock = Clock()
    agent = make_agent(post, clock=clock)
    with pytest.raises(ProviderError) as info:
        agent.act(make_task(), [], TOOLS)
    assert len(post.calls) == 1
    assert clock.sleeps == []
    assert "HTTP 401" in str(info.value)
    assert "[REDACTED]" in str(info.value)
    assert_no_key(str(info.value), repr(info.value))


def test_network_error_is_retried_then_succeeds() -> None:
    post = FakePost(
        [
            urllib.error.URLError(f"connection reset for {TEST_KEY}"),
            HttpResponse(200, {}, ok_body(json.dumps({"kind": "final", "message": "ok"}))),
        ]
    )
    clock = Clock()
    action = make_agent(post, clock=clock).act(make_task(), [], TOOLS)
    assert action.kind == "final"
    assert clock.sleeps == [2.0]


def test_network_error_exhaustion_message_has_no_key() -> None:
    post = FakePost([TimeoutError(f"timed out {TEST_KEY}") for _ in range(2)])
    agent = make_agent(post, max_retries=1)
    with pytest.raises(ProviderError) as info:
        agent.act(make_task(), [], TOOLS)
    assert info.value.__cause__ is None
    assert "TimeoutError" in str(info.value)
    assert_no_key(str(info.value), repr(info.value))


def test_200_with_error_object_raises_provider_error() -> None:
    body = json.dumps({"error": {"message": "model overloaded"}}).encode()
    post = FakePost([HttpResponse(200, {}, body)])
    with pytest.raises(ProviderError, match="model overloaded"):
        make_agent(post).act(make_task(), [], TOOLS)


def test_non_json_success_body_raises_provider_error() -> None:
    post = FakePost([HttpResponse(200, {}, b"<html>gateway</html>")])
    with pytest.raises(ProviderError, match="not JSON"):
        make_agent(post).act(make_task(), [], TOOLS)


def test_model_text_that_is_not_an_action_becomes_schema_error() -> None:
    post = FakePost([HttpResponse(200, {}, ok_body("I think we should search."))])
    action = make_agent(post).act(make_task(), [], TOOLS)
    assert action.tool_call is not None
    assert action.tool_call.name == "__schema_error__"


def test_fenced_json_in_prose_is_still_parsed() -> None:
    text = 'Here you go:\n```json\n{"kind": "tool_call", "tool": "get_budget"}\n```'
    post = FakePost([HttpResponse(200, {}, ok_body(text))])
    action = make_agent(post).act(make_task(), [], TOOLS)
    assert action.kind == "tool_call"
    assert action.tool_call is not None
    assert action.tool_call.name == "get_budget"


def test_pacing_waits_for_min_interval_between_requests() -> None:
    final = HttpResponse(200, {}, ok_body(json.dumps({"kind": "final", "message": "ok"})))
    post = FakePost([final, final])
    clock = Clock()
    agent = make_agent(post, clock=clock, min_interval_s=3.0)
    agent.act(make_task(), [], TOOLS)
    clock.now += 1.0  # one second of model time already passed
    agent.act(make_task(), [], TOOLS)
    assert clock.sleeps == [2.0]


def test_no_pacing_sleep_when_interval_already_elapsed() -> None:
    final = HttpResponse(200, {}, ok_body(json.dumps({"kind": "final", "message": "ok"})))
    post = FakePost([final, final])
    clock = Clock()
    agent = make_agent(post, clock=clock, min_interval_s=3.0)
    agent.act(make_task(), [], TOOLS)
    clock.now += 10.0
    agent.act(make_task(), [], TOOLS)
    assert clock.sleeps == []


def test_repr_and_metadata_never_contain_the_key() -> None:
    agent = make_agent(FakePost([]))
    assert_no_key(repr(agent), json.dumps(agent.metadata, default=str), str(agent.metadata))


def test_urllib_post_maps_connection_failure_to_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_args: object, **_kwargs: object) -> None:
        raise urllib.error.URLError("refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(OSError):
        urllib_post("https://example.invalid/x", {}, b"{}", 1.0)


def test_urllib_post_returns_http_error_status(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    def http_error(url: str, *_args: object, **_kwargs: object) -> None:
        raise urllib.error.HTTPError(
            url,
            429,
            "Too Many",
            {"Retry-After": "5"},
            io.BytesIO(b'{"error":1}'),  # type: ignore[arg-type]
        )

    monkeypatch.setattr("urllib.request.urlopen", http_error)
    response = urllib_post("https://example.invalid/x", {}, b"{}", 1.0)
    assert response.status == 429
    assert response.headers["retry-after"] == "5"
    assert response.body == b'{"error":1}'


def test_key_never_reaches_runner_outputs(tmp_path: Path) -> None:
    """End to end: a run that hits 429 then succeeds leaves no trace of the key on disk."""
    from passk.runner import RunDeps, run
    from passk.schema import AgentConfig, FailureCategory, RunConfig, ToolResult

    responses = [
        HttpResponse(429, {}, f'{{"error":{{"message":"slow down {TEST_KEY}"}}}}'.encode()),
        HttpResponse(200, {}, ok_body(json.dumps({"kind": "final", "message": "done"}))),
    ]
    post = FakePost(responses)
    clock = Clock()
    agent = make_agent(post, clock=clock)

    class OneStepTransport:
        def reset(self, task: Task, seed: int) -> str:
            return "h"

        def tools(self) -> list[ToolSpec]:
            return TOOLS

        def call(self, handle: str, call: Any) -> ToolResult:
            raise AssertionError("no tool call expected")

        def state(self, handle: str) -> dict[str, Any]:
            return {}

        def violations(self, handle: str) -> list[str]:
            return []

    def classify(steps: Any, **_kw: Any) -> FailureCategory:
        return FailureCategory.SUCCESS if steps else FailureCategory.MAX_TURNS

    config = RunConfig(
        run_id="keycheck",
        tasks_glob="unused",
        agent=AgentConfig(kind="openai_compat", params={"model": MODEL}),
        n_trials=1,
        base_seed=1,
        transport="inprocess",
    )
    deps = RunDeps(
        tasks=[make_task()],
        make_agent=lambda _cfg: agent,
        make_transport=lambda _cfg: OneStepTransport(),  # type: ignore[arg-type,return-value]
        check_goal=lambda _task, _state, _violations: True,
        classify=classify,
    )
    report = run(config, tmp_path, deps=deps)
    assert report.failure_totals == {"success": 1}
    run_dir = tmp_path / "keycheck"
    written = "".join(p.read_text(encoding="utf-8") for p in run_dir.iterdir())
    assert TEST_KEY not in written
    assert clock.sleeps  # the 429 was retried
