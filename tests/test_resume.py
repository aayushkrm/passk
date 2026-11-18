"""Incremental persistence, crash-resume, config-hash refusal and the provider-quota stop."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import passk
from passk import runner
from passk.agents.openai_compat import (
    HttpResponse,
    OpenAICompatAgent,
    ProviderError,
    reset_time,
)
from passk.envs.base import load_tasks
from passk.runner import ConfigMismatchError, RunDeps, RunIncomplete, run
from passk.schema import AgentConfig, RunConfig

TASKS_GLOB = str(Path(passk.__file__).parent / "tasks" / "labstock" / "*.yaml")
MODEL = "test/model"
NOW = 1_800_000_000.0  # fixed epoch seconds for the hosted tests
TEST_KEY = "sk-or-test-SECRET-0123456789"


class Crash(RuntimeError):
    """Stands in for a kill: not a provider error, so the run must not swallow it."""


class CrashAfter:
    """Transport wrapper that dies when the episode after ``after`` finished episodes starts."""

    def __init__(self, inner: Any, after: int) -> None:
        self._inner = inner
        self._after = after
        self._resets = 0

    def reset(self, task: Any, seed: int) -> str:
        if self._resets >= self._after:
            raise Crash("simulated kill")
        self._resets += 1
        return self._inner.reset(task, seed)  # type: ignore[no-any-return]

    def tools(self) -> Any:
        return self._inner.tools()

    def call(self, handle: str, call: Any) -> Any:
        return self._inner.call(handle, call)

    def state(self, handle: str) -> Any:
        return self._inner.state(handle)

    def violations(self, handle: str) -> Any:
        return self._inner.violations(handle)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()


@pytest.fixture
def fixed_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every timing reads 0, so episodes are byte-for-byte reproducible."""
    monkeypatch.setattr(runner.time, "perf_counter", lambda: 0.0)


def _tasks(count: int = 6) -> list[Any]:
    return sorted(load_tasks(TASKS_GLOB), key=lambda t: t.id)[:count]


def _config(run_id: str, n_trials: int = 3) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        tasks_glob=TASKS_GLOB,
        agent=AgentConfig(kind="noisy_oracle", params={"p_error": 0.3, "seed": 4}),
        n_trials=n_trials,
        base_seed=11,
        transport="inprocess",
    )


def _crashing_deps(tasks: list[Any], after: int) -> RunDeps:
    def make_transport(cfg: RunConfig) -> Any:
        return CrashAfter(runner.build_transport(cfg), after)

    return RunDeps(tasks=tasks, make_transport=make_transport)


def test_crash_then_resume_gives_byte_identical_trajectories_and_spans(
    tmp_path: Path, fixed_clock: None
) -> None:
    tasks = _tasks()
    # The same run id in two out roots: trajectories embed the run id, so bytes are comparable.
    clean = run(_config("same"), tmp_path / "clean", deps=RunDeps(tasks=tasks))

    with pytest.raises(Crash):
        run(_config("same"), tmp_path / "crash", deps=_crashing_deps(tasks, after=7))
    killed_dir = tmp_path / "crash" / "same"
    assert len((killed_dir / "trajectories.jsonl").read_text().splitlines()) == 7
    state = json.loads((killed_dir / "state.json").read_text())
    assert state["status"] == "running" and state["episodes_done"] == 7

    resumed = run(_config("same"), tmp_path / "crash", deps=RunDeps(tasks=tasks))
    assert resumed.complete is True
    assert resumed.tasks == clean.tasks
    assert resumed.summary == clean.summary
    clean_dir = tmp_path / "clean" / "same"
    for name in ("trajectories.jsonl", "spans.jsonl"):
        assert (killed_dir / name).read_bytes() == (clean_dir / name).read_bytes(), name
    assert json.loads((killed_dir / "state.json").read_text())["status"] == "complete"


def test_resume_drops_a_torn_last_line_and_still_matches(tmp_path: Path, fixed_clock: None) -> None:
    tasks = _tasks(4)
    run(_config("same"), tmp_path / "clean", deps=RunDeps(tasks=tasks))
    with pytest.raises(Crash):
        run(_config("same"), tmp_path / "torn", deps=_crashing_deps(tasks, after=5))
    torn = tmp_path / "torn" / "same" / "trajectories.jsonl"
    with torn.open("a", encoding="utf-8") as handle:
        handle.write('{"run_id": "same", "task_id": "t1_0')  # interrupted mid-write, no newline
    run(_config("same"), tmp_path / "torn", deps=RunDeps(tasks=tasks))
    assert torn.read_bytes() == (tmp_path / "clean" / "same" / "trajectories.jsonl").read_bytes()


def test_a_different_configuration_is_refused_and_nothing_changes(
    tmp_path: Path, fixed_clock: None
) -> None:
    tasks = _tasks(3)
    with pytest.raises(Crash):
        run(_config("hashed"), tmp_path / "out", deps=_crashing_deps(tasks, after=2))
    run_dir = tmp_path / "out" / "hashed"
    before = {p.name: p.read_bytes() for p in run_dir.iterdir()}
    with pytest.raises(ConfigMismatchError, match="different configuration"):
        run(_config("hashed", n_trials=4), tmp_path / "out", deps=RunDeps(tasks=tasks))
    with pytest.raises(ConfigMismatchError):
        run(_config("hashed"), tmp_path / "out", deps=RunDeps(tasks=_tasks(2)))
    assert {p.name: p.read_bytes() for p in run_dir.iterdir()} == before


def test_resume_flag_controls_an_existing_run(tmp_path: Path, fixed_clock: None) -> None:
    tasks = _tasks(2)
    with pytest.raises(Crash):
        run(_config("flag"), tmp_path / "out", deps=_crashing_deps(tasks, after=1))
    with pytest.raises(FileExistsError, match="--resume"):
        run(_config("flag"), tmp_path / "out", deps=RunDeps(tasks=tasks), resume=False)
    with pytest.raises(ValueError, match="nothing to resume"):
        run(_config("fresh"), tmp_path / "out", deps=RunDeps(tasks=tasks), resume=True)
    run(_config("flag"), tmp_path / "out", deps=RunDeps(tasks=tasks), resume=True)


def test_a_complete_run_is_not_overwritten(tmp_path: Path, fixed_clock: None) -> None:
    tasks = _tasks(2)
    run(_config("done"), tmp_path / "out", deps=RunDeps(tasks=tasks))
    with pytest.raises(FileExistsError, match="complete run"):
        run(_config("done"), tmp_path / "out", deps=RunDeps(tasks=tasks))


def test_reset_time_reads_epoch_ms_epoch_s_relative_and_retry_after() -> None:
    reset_ms = str(int((NOW + 3600) * 1000))
    assert reset_time({"X-RateLimit-Reset": reset_ms}, NOW) == (
        datetime.fromtimestamp(NOW + 3600, tz=UTC).isoformat(timespec="seconds"),
        3600.0,
    )
    assert reset_time({"x-ratelimit-reset": str(int(NOW + 90))}, NOW)[1] == 90.0
    assert reset_time({"retry-after": "12"}, NOW)[1] == 12.0
    assert reset_time({}, NOW) == (None, None)
    assert reset_time({"retry-after": "soon"}, NOW) == (None, None)


def _ok(content: str) -> HttpResponse:
    body = json.dumps(
        {
            "model": MODEL,
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 8},
        }
    ).encode()
    return HttpResponse(200, {}, body)


FINAL = json.dumps({"kind": "final", "tool": "none", "arguments": {}, "message": "done"})


class ScriptedPost:
    """Fake HTTP post: returns the scripted responses in order, then the default response."""

    def __init__(self, script: list[HttpResponse], default: HttpResponse | None = None) -> None:
        self._script = list(script)
        self._default = default
        self.calls = 0

    def __call__(self, url: str, headers: dict[str, str], body: bytes, timeout: float) -> Any:
        self.calls += 1
        if self._script:
            return self._script.pop(0)
        assert self._default is not None, "no response scripted"
        return self._default


def _hosted_deps(tasks: list[Any], post: ScriptedPost) -> RunDeps:
    def make_agent(_cfg: AgentConfig) -> OpenAICompatAgent:
        return OpenAICompatAgent(
            model=MODEL,
            post=post,
            sleep=lambda _s: None,
            min_interval_s=0.0,
            wall_clock=lambda: NOW,
        )

    return RunDeps(tasks=tasks, make_agent=make_agent)


def _hosted_config(run_id: str) -> RunConfig:
    return RunConfig(
        run_id=run_id,
        tasks_glob=TASKS_GLOB,
        agent=AgentConfig(kind="openai_compat", params={"model": MODEL}),
        n_trials=2,
        base_seed=3,
        transport="inprocess",
    )


def test_provider_quota_stops_cleanly_keeps_episodes_and_writes_a_partial_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_clock: None
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", TEST_KEY)
    tasks = _tasks(2)
    reset_epoch = NOW + 6 * 3600
    quota = HttpResponse(
        429,
        {
            "X-RateLimit-Limit": "50",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(reset_epoch * 1000)),
        },
        b'{"error": {"message": "daily limit reached"}}',
    )
    post = ScriptedPost([_ok(FINAL), _ok(FINAL)], default=quota)

    with pytest.raises(RunIncomplete) as info:
        run(_hosted_config("quota"), tmp_path / "out", deps=_hosted_deps(tasks, post))

    exc = info.value
    expected_reset = datetime.fromtimestamp(reset_epoch, tz=UTC).isoformat(timespec="seconds")
    assert exc.reset_at == expected_reset
    assert "quota" in exc.reason
    assert exc.report.complete is False
    assert sum(s.n for s in exc.report.tasks) == 2  # the two finished episodes
    assert post.calls == 3  # the third episode made one request, then the quota stop
    run_dir = tmp_path / "out" / "quota"
    report = json.loads((run_dir / "report.json").read_text())
    assert report["complete"] is False
    assert report["reset_at"] == expected_reset
    assert "quota" in report["stop_reason"]
    assert len((run_dir / "trajectories.jsonl").read_text().splitlines()) == 2
    state = json.loads((run_dir / "state.json").read_text())
    assert state["status"] == "partial" and state["episodes_done"] == 2


def test_quota_partial_run_resumes_after_the_limit_resets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_clock: None
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", TEST_KEY)
    tasks = _tasks(2)
    quota = HttpResponse(429, {"X-RateLimit-Remaining": "0"}, b"{}")
    with pytest.raises(RunIncomplete):
        run(
            _hosted_config("later"),
            tmp_path / "out",
            deps=_hosted_deps(tasks, ScriptedPost([_ok(FINAL)], default=quota)),
        )
    before = (tmp_path / "out" / "later" / "trajectories.jsonl").read_bytes()
    report = run(
        _hosted_config("later"),
        tmp_path / "out",
        deps=_hosted_deps(tasks, ScriptedPost([], default=_ok(FINAL))),
    )
    assert report.complete is True
    after = (tmp_path / "out" / "later" / "trajectories.jsonl").read_bytes()
    assert after.startswith(before) and after != before
    assert sum(s.n for s in report.tasks) == 4


def test_a_burst_429_is_retried_not_treated_as_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fixed_clock: None
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", TEST_KEY)
    tasks = _tasks(1)
    burst = HttpResponse(429, {"Retry-After": "2"}, b"{}")
    post = ScriptedPost([burst, _ok(FINAL)], default=_ok(FINAL))
    report = run(
        _hosted_config("burst").model_copy(update={"n_trials": 1}),
        tmp_path / "out",
        deps=_hosted_deps(tasks, post),
    )
    assert report.complete is True
    assert post.calls == 2


def test_provider_error_carries_the_reset_time(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", TEST_KEY)
    agent = OpenAICompatAgent(
        model=MODEL,
        post=ScriptedPost([], default=HttpResponse(429, {"X-RateLimit-Remaining": "0"}, b"{}")),
        sleep=lambda _s: None,
        wall_clock=lambda: NOW,
    )
    with pytest.raises(ProviderError) as info:
        agent._request(b"{}")
    assert info.value.quota is True
    assert info.value.reset_at is None  # no reset header in this response


def test_resume_refuses_a_run_that_was_started_by_another_task_set(
    tmp_path: Path, fixed_clock: None
) -> None:
    tasks = _tasks(3)
    with pytest.raises(Crash):
        run(_config("tasks"), tmp_path / "out", deps=_crashing_deps(tasks, after=1))
    assert os.path.exists(tmp_path / "out" / "tasks" / "state.json")
    with pytest.raises(ConfigMismatchError):
        run(_config("tasks"), tmp_path / "out", deps=RunDeps(tasks=tasks[:1]))
