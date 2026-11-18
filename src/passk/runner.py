"""Run tasks x trials for one configuration and write the run directory.

Output layout: ``runs/<run_id>/{trajectories.jsonl, spans.jsonl, report.json, env.json}``.
Every file is written after all episodes finish, and each one is replaced atomically, so a
crash leaves no half-written report. A run whose directory already holds files is refused.

Assumed signatures of other work packages (tests inject fakes through ``RunDeps``):
    passk.envs.base.load_tasks(tasks_glob: str) -> list[Task]
    passk.envs.labstock.LabStockEnv()                       (BaseEnv subclass)
    passk.transport.InProcessTransport(env)
    passk.transport.MCPStdioTransport(env_name: str = ...)
    passk.goals.check_goal(task, final_state, violations) -> bool
        (no violations, goals hold, no forbidden assertion holds)
    passk.taxonomy.classify(steps, *, violations, goal_met, claimed_done, max_turns)
        -> FailureCategory
    passk.metrics.pass_at_k(n, c, k), pass_hat_k(n, c, k), bootstrap_ci(values, seed=...)
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import statistics
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from passk import metrics, taxonomy, trace
from passk.agents._common import SCHEMA_ERROR_TOOL, schema_error_result
from passk.agents.llamacpp import LlamaCppAgent
from passk.agents.openai_compat import OpenAICompatAgent
from passk.schema import (
    Agent,
    AgentAction,
    AgentConfig,
    FailureCategory,
    RunConfig,
    RunReport,
    Step,
    Task,
    TaskStats,
    ToolResult,
    ToolSpec,
    ToolTransport,
    Trajectory,
)

SEED_MODULUS = 2**31 - 1
"""Episode seeds are non-negative 31-bit integers (fits llama.cpp and JSON APIs)."""

RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")

SCRIPTED_CLASSES: dict[str, str] = {
    "oracle": "OracleAgent",
    "noisy_oracle": "NoisyOracleAgent",
    "replay": "ReplayAgent",
}
"""Agent kinds served by ``passk.agents.scripted`` and the class each one names."""

REPO_ROOT = Path(__file__).resolve().parents[2]
ENVINFO_PATH = REPO_ROOT / "scripts" / "envinfo.py"

GoalChecker = Callable[[Task, dict[str, Any], list[str]], bool]
Classifier = Callable[..., FailureCategory]


@dataclass
class RunDeps:
    """Optional overrides for the collaborators of ``run``. Unset fields use the real modules.

    Attributes:
        tasks: Tasks to run instead of ``load_tasks(config.tasks_glob)``.
        make_transport: Builds the tool transport for a run.
        make_agent: Builds the agent from its ``AgentConfig``.
        check_goal: Returns True when the task is solved (no violations, goals hold,
            no forbidden assertion holds).
        classify: Assigns the primary ``FailureCategory`` for an episode.
    """

    tasks: Sequence[Task] | None = None
    make_transport: Callable[[RunConfig], ToolTransport] | None = None
    make_agent: Callable[[AgentConfig], Agent] | None = None
    check_goal: GoalChecker | None = None
    classify: Classifier | None = None


def episode_seed(base_seed: int, task_id: str, trial: int) -> int:
    """Derive the seed for one episode as a stable hash of (base_seed, task_id, trial).

    Uses SHA-256, not Python's randomised ``hash``, so the value is the same in every process.

    Args:
        base_seed: Base seed from the run config.
        task_id: Task identifier.
        trial: Trial index.

    Returns:
        A non-negative integer below 2**31 - 1.
    """
    digest = hashlib.sha256(f"{base_seed}|{task_id}|{trial}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % SEED_MODULUS


def build_agent(config: AgentConfig) -> Agent:
    """Construct the agent named by ``config``.

    ``llamacpp`` and ``openai_compat`` are built here. ``oracle``, ``noisy_oracle`` and
    ``replay`` are looked up lazily in ``passk.agents.scripted``.

    Args:
        config: Agent kind and parameters. ``model_path`` may be given as a string.

    Returns:
        An object satisfying the ``Agent`` protocol.

    Raises:
        ValueError: If the kind is unknown.
    """
    params = dict(config.params)
    if config.kind == "llamacpp":
        if "model_path" in params:
            params["model_path"] = Path(params["model_path"])
        return LlamaCppAgent(**params)
    if config.kind == "openai_compat":
        return OpenAICompatAgent(**params)
    if config.kind in SCRIPTED_CLASSES:
        scripted = importlib.import_module("passk.agents.scripted")
        cls = getattr(scripted, SCRIPTED_CLASSES[config.kind])
        return cls(**params)
    raise ValueError(f"unknown agent kind: {config.kind!r}")


def build_transport(config: RunConfig) -> ToolTransport:
    """Construct the transport named by ``config.transport``.

    Args:
        config: Run configuration.

    Returns:
        ``InProcessTransport`` around a fresh environment, or an ``MCPStdioTransport``.

    Raises:
        ValueError: If the environment name is unknown.
    """
    if config.env != "labstock":
        raise ValueError(f"unknown environment: {config.env!r}")
    if config.transport == "inprocess":
        from passk.envs.labstock import LabStockEnv
        from passk.transport import InProcessTransport

        return InProcessTransport(LabStockEnv())
    from passk.transport import MCPStdioTransport

    return MCPStdioTransport(env_name=config.env)


def _default_check_goal(task: Task, final_state: dict[str, Any], violations: list[str]) -> bool:
    """Call ``passk.goals.check_goal`` (imported lazily)."""
    from passk.goals import check_goal

    return bool(check_goal(task, final_state, violations))


def _load_envinfo() -> ModuleType | None:
    """Import ``scripts/envinfo.py`` by path, or return None when it is not present."""
    if not ENVINFO_PATH.is_file():
        return None
    spec = importlib.util.spec_from_file_location("passk_envinfo", ENVINFO_PATH)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def environment_record() -> dict[str, Any]:
    """Return the ``env.json`` record: the ``scripts/envinfo.py`` record when available.

    Falls back to a minimal stdlib record (Python and OS only, no host identifiers) when the
    script is absent, for example in an installed wheel.

    Returns:
        A JSON-serialisable dict with no hostname, username or paths.
    """
    module = _load_envinfo()
    if module is not None:
        record: dict[str, Any] = module.collect(REPO_ROOT)
        return record
    return {
        "envinfo": "scripts/envinfo.py not found; minimal record",
        "python": {"version": platform.python_version()},
        "os": {"os": platform.system(), "arch": platform.machine()},
    }


def _model_name(agent: Any, config: AgentConfig) -> str:
    """Return the requested model name for spans, falling back to the agent kind."""
    name = getattr(agent, "model_id", None)
    return name if isinstance(name, str) and name else config.kind


def _agent_id(agent: Any, config: AgentConfig) -> str:
    """Return the agent identifier written into trajectories."""
    value = getattr(agent, "agent_id", None)
    return value if isinstance(value, str) and value else config.kind


def _provider(agent: Any) -> str:
    """Return the ``gen_ai.provider.name`` value for an agent."""
    value = getattr(agent, "provider_name", None)
    return value if isinstance(value, str) and value else "passk"


def _run_episode(
    config: RunConfig,
    task: Task,
    trial: int,
    agent: Any,
    transport: ToolTransport,
    agent_id: str,
    goal_checker: GoalChecker,
    classifier: Classifier,
) -> tuple[Trajectory, list[float | None]]:
    """Run one episode and return its trajectory and per-step tool latencies.

    Turn budget: ``task.max_turns`` agent decisions at most. A schema-error action is recorded
    as a failed tool result and is not sent to the environment.
    """
    seed = episode_seed(config.base_seed, task.id, trial)
    begin = getattr(agent, "begin_episode", None)
    if callable(begin):
        begin(seed)
    handle = transport.reset(task, seed)
    tools: list[ToolSpec] = transport.tools()
    steps: list[Step] = []
    tool_latencies: list[float | None] = []
    claimed_done = False
    started = time.perf_counter()
    for index in range(task.max_turns):
        if hasattr(agent, "last_call"):
            agent.last_call = None
        t_act = time.perf_counter()
        action: AgentAction = agent.act(task, list(steps), tools)
        act_s = time.perf_counter() - t_act
        info = getattr(agent, "last_call", None)
        in_tokens = getattr(info, "input_tokens", None)
        out_tokens = getattr(info, "output_tokens", None)
        if action.kind == "final":
            steps.append(
                Step(
                    index=index,
                    action=action,
                    result=None,
                    latency_s=act_s,
                    input_tokens=in_tokens,
                    output_tokens=out_tokens,
                )
            )
            tool_latencies.append(None)
            claimed_done = True
            break
        call = action.tool_call
        if call is None:
            raise ValueError("agent returned kind='tool_call' without a tool_call")
        tool_s: float | None = None
        result: ToolResult
        if call.name == SCHEMA_ERROR_TOOL:
            result = schema_error_result(call.call_id, str(call.arguments.get("reason", "")))
        else:
            t_tool = time.perf_counter()
            result = transport.call(handle, call)
            tool_s = time.perf_counter() - t_tool
        steps.append(
            Step(
                index=index,
                action=action,
                result=result,
                latency_s=act_s,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
            )
        )
        tool_latencies.append(tool_s)
    final_state = transport.state(handle)
    violations = list(transport.violations(handle))
    goal_met = bool(goal_checker(task, final_state, violations))
    failure = classifier(
        steps,
        violations=violations,
        goal_met=goal_met,
        claimed_done=claimed_done,
        max_turns=task.max_turns,
    )
    trajectory = Trajectory(
        run_id=config.run_id,
        task_id=task.id,
        trial=trial,
        seed=seed,
        agent_id=agent_id,
        steps=steps,
        final_state=final_state,
        success=failure == FailureCategory.SUCCESS,
        violations=violations,
        failure=failure,
        wall_s=time.perf_counter() - started,
    )
    return trajectory, tool_latencies


def _median_or_none(values: Sequence[float]) -> float | None:
    """Return the median of ``values``, or None when there are none."""
    return float(statistics.median(values)) if values else None


def _aggregate(
    config: RunConfig,
    trajectories: Sequence[Trajectory],
    tasks: Sequence[Task],
) -> tuple[list[TaskStats], dict[str, dict[int, dict[str, float]]], dict[str, int], dict[str, Any]]:
    """Aggregate episodes into per-task stats, summary intervals, failure and token totals.

    Only k values with ``1 <= k <= n_trials`` are reported; larger ones are skipped.
    """
    by_task: dict[str, list[Trajectory]] = {task.id: [] for task in tasks}
    for traj in trajectories:
        by_task[traj.task_id].append(traj)
    tier_of = {task.id: task.tier for task in tasks}

    stats: list[TaskStats] = []
    for task in tasks:
        trajs = by_task[task.id]
        n = len(trajs)
        c = sum(1 for t in trajs if t.success)
        ks = [k for k in config.k_values if 1 <= k <= n]
        stats.append(
            TaskStats(
                task_id=task.id,
                tier=tier_of[task.id],
                n=n,
                c=c,
                pass_at_k={k: metrics.pass_at_k(n, c, k) for k in ks},
                pass_hat_k={k: metrics.pass_hat_k(n, c, k) for k in ks},
                failures=dict(sorted(Counter(t.failure.value for t in trajs).items())),
            )
        )

    summary: dict[str, dict[int, dict[str, float]]] = {}
    for metric in ("pass_at_k", "pass_hat_k"):
        summary[metric] = {}
        for k in sorted({k for s in stats for k in getattr(s, metric)}):
            values = [getattr(s, metric)[k] for s in stats if k in getattr(s, metric)]
            mean, low, high = metrics.bootstrap_ci(values, seed=config.base_seed)
            summary[metric][k] = {"mean": mean, "ci_low": low, "ci_high": high}

    failure_totals = dict(sorted(Counter(t.failure.value for t in trajectories).items()))

    steps = [s for t in trajectories for s in t.steps]
    in_values = [s.input_tokens for s in steps if s.input_tokens is not None]
    out_values = [s.output_tokens for s in steps if s.output_tokens is not None]
    per_in = [sum(s.input_tokens or 0 for s in t.steps) for t in trajectories]
    per_out = [sum(s.output_tokens or 0 for s in t.steps) for t in trajectories]
    tokens: dict[str, Any] = {
        "model_calls": sum(
            1 for s in steps if s.input_tokens is not None or s.output_tokens is not None
        ),
        "input_total": sum(in_values),
        "output_total": sum(out_values),
        "input_median_per_episode": _median_or_none(per_in),
        "output_median_per_episode": _median_or_none(per_out),
    }
    return stats, summary, failure_totals, tokens


def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to a temporary sibling and rename it onto ``path``."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run(
    config: RunConfig,
    out_root: Path = Path("runs"),
    *,
    deps: RunDeps | None = None,
) -> RunReport:
    """Run every task matching ``config.tasks_glob`` for ``config.n_trials`` trials.

    Tasks are sorted by id. Episodes run in order: task, then trial. Enforces
    ``Task.max_turns``, writes the run directory and returns the report.

    Provider errors (``ProviderError``) propagate and abort the run without writing anything,
    because an outage is not a model failure.

    Args:
        config: Run configuration.
        out_root: Directory under which ``<run_id>/`` is created.
        deps: Optional collaborator overrides (used by tests).

    Returns:
        The aggregated ``RunReport`` (also written to ``report.json``).

    Raises:
        ValueError: On an invalid run id, ``n_trials < 1`` or no matching tasks.
        FileExistsError: If the run directory already holds files.
    """
    deps = deps or RunDeps()
    if config.n_trials < 1:
        raise ValueError("n_trials must be at least 1")
    if not RUN_ID_PATTERN.fullmatch(config.run_id):
        raise ValueError(f"run_id {config.run_id!r} must be a plain name (letters, digits, ._-)")
    run_dir = Path(out_root) / config.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{run_dir} already holds files; choose a new run_id")

    if deps.tasks is not None:
        tasks = sorted(deps.tasks, key=lambda t: t.id)
    else:
        from passk.envs.base import load_tasks

        tasks = load_tasks(config.tasks_glob)
    if not tasks:
        raise ValueError(f"no tasks match {config.tasks_glob!r}")

    make_agent = deps.make_agent or build_agent
    make_transport = deps.make_transport or build_transport
    goal_checker = deps.check_goal or _default_check_goal
    classifier = deps.classify or taxonomy.classify

    started = time.perf_counter()
    agent = make_agent(config.agent)
    agent_id = _agent_id(agent, config.agent)
    model = _model_name(agent, config.agent)
    provider = _provider(agent)
    transport = make_transport(config)
    trajectories: list[Trajectory] = []
    span_records: list[dict[str, Any]] = []
    try:
        for task in tasks:
            for trial in range(config.n_trials):
                traj, tool_latencies = _run_episode(
                    config, task, trial, agent, transport, agent_id, goal_checker, classifier
                )
                trajectories.append(traj)
                span_records.extend(
                    trace.build_spans(
                        traj,
                        request_model=model,
                        provider_name=provider,
                        tool_latencies_s=tool_latencies,
                    )
                )
    finally:
        close = getattr(transport, "close", None)
        if callable(close):
            close()

    stats, summary, failure_totals, tokens = _aggregate(config, trajectories, tasks)
    env_info = environment_record()
    agent_meta = getattr(agent, "metadata", None)
    if isinstance(agent_meta, dict):
        env_info["agent"] = agent_meta

    report = RunReport(
        run_id=config.run_id,
        config=config,
        env_info=env_info,
        tasks=stats,
        summary=summary,
        failure_totals=failure_totals,
        tokens=tokens,
        wall_s=time.perf_counter() - started,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        run_dir / "trajectories.jsonl",
        "".join(t.model_dump_json() + "\n" for t in trajectories),
    )
    with trace.SpanWriter(run_dir / "spans.jsonl") as writer:
        for span in span_records:
            writer.write(span)
    _atomic_write_text(run_dir / "env.json", json.dumps(env_info, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(run_dir / "report.json", report.model_dump_json(indent=2) + "\n")
    return report
