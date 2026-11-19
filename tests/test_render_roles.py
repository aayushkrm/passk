"""Renderer pieces for the E6, E7b and Qwen3 roles: names, caveats, recovery and comparisons."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from passk.schema import (
    AgentAction,
    FailureCategory,
    Step,
    ToolCall,
    ToolResult,
    Trajectory,
)

REPO = Path(__file__).resolve().parents[1]


def _renderer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "render_readme", REPO / "scripts" / "render_readme.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RR = _renderer()


def _step(index: int, name: str, *, ok: bool, error: str | None) -> Step:
    call = ToolCall(call_id=f"c{index}", name=name, arguments={})
    result = ToolResult(call_id=f"c{index}", ok=ok, content=None, error=error)
    return Step(
        index=index,
        action=AgentAction(kind="tool_call", tool_call=call),
        result=result,
        latency_s=0.0,
    )


def _traj(trial: int, steps: list[Step], success: bool) -> Trajectory:
    return Trajectory(
        run_id="e6-test",
        task_id="t1_01",
        trial=trial,
        seed=trial,
        agent_id="x",
        steps=steps,
        final_state={},
        success=success,
        violations=[],
        failure=FailureCategory.SUCCESS if success else FailureCategory.WRONG_STATE,
        wall_s=0.0,
    )


def test_quantization_is_read_from_the_file_name() -> None:
    assert RR._quantization("Qwen3-1.7B-Q8_0.gguf") == "Q8_0"
    assert RR._quantization("qwen2.5-1.5b-instruct-q4_k_m.gguf") == "Q4_K_M"
    assert RR._quantization(None) == "not stated"


def test_roles_come_from_the_run_id() -> None:
    assert RR._role("e3-qwen3-1.7b-q8-t0.7").startswith("E3 optional arm, Qwen3-1.7B")
    assert RR._role("e6-qwen2.5-1.5b-flaky0.2").startswith("E6 descriptive robustness run")
    assert RR._role("e7b-nemotron-3-ultra-stratified-t0.7").startswith("E7b descriptive hosted run")
    assert RR._role("e7-nemotron-3-ultra-t0.7") == "E7 hosted run"
    assert RR._role("something-else") == "something-else"


def test_recovery_counts_episodes_with_a_transient_that_still_succeeded() -> None:
    transient = _step(0, "get_budget", ok=False, error="transient")
    retry_ok = _step(1, "get_budget", ok=True, error=None)
    plain = _step(0, "get_budget", ok=True, error=None)
    trajectories = [
        _traj(0, [transient, retry_ok], success=True),
        _traj(1, [transient, _step(1, "get_budget", ok=False, error="transient")], success=False),
        _traj(2, [plain], success=True),
    ]
    rec = RR._recovery(trajectories)
    assert rec is not None
    assert rec["episodes_with_transient"] == 2
    assert rec["recovered"] == 1
    assert rec["transient_results"] == 3
    assert RR._recovery(None) is None


def _record(
    run_id: str, *, trials: int, pass1: float, pass_k: dict[str, float], repo: str
) -> dict[str, Any]:
    interval = {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    return {
        "run_id": run_id,
        "role": RR._role(run_id),
        "trials_per_task": trials,
        "pass_at_k": {"1": {**interval, "mean": pass1}},
        "pass_hat_k": {k: {**interval, "mean": v} for k, v in pass_k.items()},
        "tiers": [
            {
                "successes": 1,
                "episodes": 2,
                "tier": 1,
                "tasks": 1,
                "rate": 0.5,
                "ci_low": 0.0,
                "ci_high": 1.0,
            }
        ],
        "agent_meta": {"repo_id": repo, "model_file": "x-Q4_K_M.gguf"},
        "tokens": {},
        "complete": True,
    }


def test_e6_caveat_states_the_trial_counts_and_no_significance_claim() -> None:
    baseline = _record(
        "e3-qwen2.5-1.5b-t0.7", trials=5, pass1=0.4, pass_k={"3": 0.1, "5": 0.0}, repo="r"
    )
    e6 = {
        **_record("e6-qwen2.5-1.5b-flaky0.2", trials=3, pass1=0.4, pass_k={"3": 0.1}, repo="r"),
        "flaky_p": 0.2,
        "agent_meta": {},
    }
    text = " ".join(RR._role_caveat(e6, baseline))
    assert "3 trials per task" in text and "5 trials per task" in text
    assert "No significance claim is made" in text


def test_comparison_tables_appear_only_when_both_runs_exist() -> None:
    baseline = _record(
        "e3-qwen2.5-1.5b-t0.7", trials=5, pass1=0.4, pass_k={"3": 0.1, "5": 0.0}, repo="r"
    )
    e6 = _record("e6-qwen2.5-1.5b-flaky0.2", trials=3, pass1=0.4, pass_k={"3": 0.1}, repo="r")
    text = "\n".join(RR._comparison_lines([baseline, e6]))
    assert "Descriptive comparison: E6 against the E3 1.5B baseline" in text
    assert "Not a significance test" in text
    assert "Descriptive contrast" not in text  # no Qwen3 run in this set
    assert RR._comparison_lines([e6]) == []  # no baseline, no comparison
