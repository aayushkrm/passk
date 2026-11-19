#!/usr/bin/env python3
"""Render the results block of README.md from the committed results/ directory.

Every number in the README results block and in ``results/headline.json`` comes from the
JSON files under ``results/``. Nothing is typed by hand.

Inputs:
    results/**/report.json          run reports written by ``passk run``
    results/**/trajectories.jsonl   trajectories (optional; used for token and time medians)
    results/calibration/validate.json  E1 and E2 output of ``passk validate``
    results/gates/*.txt             verbatim ``passk gate`` output (E5 and others)

Outputs:
    results/headline.json           the numbers the hub and the README quote
    README.md                       the block between the results markers

Modes:
    (default)     write results/headline.json and the README block
    --headline    write results/headline.json only
    --check       exit 1 if either output would change; write nothing

The copy in this repository is deliberately standalone: it imports only ``passk`` (this
repo's package) and the standard library.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
README = REPO / "README.md"
HEADLINE = RESULTS / "headline.json"
CALIBRATION = RESULTS / "calibration" / "validate.json"
GATES_DIR = RESULTS / "gates"
START = "<!-- results:start -->"
END = "<!-- results:end -->"
SYNTHETIC = "SYNTHETIC"
NO_RESULTS = "No committed model results yet. The block fills in when `results/` holds a run."

sys.path.insert(0, str(REPO / "src"))

from passk import metrics  # noqa: E402
from passk.diff import is_synthetic, load_report, load_trajectories  # noqa: E402
from passk.gate import gate, load_policy  # noqa: E402
from passk.schema import RunReport, Trajectory  # noqa: E402
from passk.validate import wilson_interval  # noqa: E402

CONFIGS = REPO / "configs"
GATE_POLICY = CONFIGS / "gate.yaml"
TOOL = "passk"
CONTRACT_SCHEMA_VERSION = "1"
MAX_ITEMS = 6
TAGLINE = (
    "pass^k for tool-using agents: an offline, CI-gated reliability harness that measures "
    "whether an agent succeeds every time, not just once."
)
QUICKSTART = "uv tool install git+https://github.com/aayushkrm/passk && passk demo"
LINKS: dict[str, str | None] = {
    "repo": "https://github.com/aayushkrm/passk",
    "site": "https://aayushkrm.github.io/passk/",
    "release": None,
}


def _rel(path: Path) -> str:
    """Repo-relative path for links. A file outside the repository is named by its base name."""
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.name


def _interval(item: dict[str, Any] | None) -> str:
    if not item:
        return "n/a"
    return f"{item['mean']:.3f} [{item['ci_low']:.3f}, {item['ci_high']:.3f}]"


def _k_values(report: RunReport) -> list[int]:
    return sorted(set(report.config.k_values))


def _summary_at(report: RunReport, metric: str, k: int) -> dict[str, float] | None:
    value = report.summary.get(metric, {}).get(k)
    return value if isinstance(value, dict) else None


def _failure_shares(report: RunReport) -> dict[str, float]:
    """Each failure category as a share of all episodes (successes count in the denominator)."""
    episodes = sum(stats.n for stats in report.tasks)
    if episodes == 0:
        return {}
    return {name: count / episodes for name, count in sorted(report.failure_totals.items())}


def _tier_failure_shares(report: RunReport) -> dict[str, dict[str, float]]:
    """Failure shares by tier, computed from the per-task failure counts."""
    by_tier: dict[int, dict[str, int]] = {}
    for stats in report.tasks:
        bucket = by_tier.setdefault(stats.tier, {})
        for name, count in stats.failures.items():
            bucket[name] = bucket.get(name, 0) + count
    shares: dict[str, dict[str, float]] = {}
    for tier, counts in sorted(by_tier.items()):
        total = sum(counts.values())
        shares[f"tier{tier}"] = (
            {name: count / total for name, count in sorted(counts.items())} if total else {}
        )
    return shares


def _episode_costs(
    report: RunReport, trajectories: list[Trajectory] | None
) -> dict[str, float | None]:
    """Median tokens per episode and median wall seconds per episode, when available."""
    if not trajectories:
        return {"median_wall_s": None, "median_output_tokens": None}
    walls = [t.wall_s for t in trajectories]
    outputs = [sum(s.output_tokens or 0 for s in t.steps) for t in trajectories]
    return {
        "median_wall_s": float(statistics.median(walls)),
        "median_output_tokens": float(statistics.median(outputs)),
    }


def _agent_label(report: RunReport) -> str:
    agent = report.env_info.get("agent")
    if isinstance(agent, dict):
        if agent.get("kind") == "llamacpp":
            return f"{agent.get('repo_id') or agent.get('model')} ({agent.get('model_file')})"
        if agent.get("kind") == "openai_compat":
            reported = ", ".join(agent.get("model_reported") or []) or "n/a"
            return f"{agent.get('model_requested')} (reported: {reported})"
    return report.config.agent.kind


def _tier_rows(report: RunReport) -> list[dict[str, Any]]:
    """Pooled success counts per tier, with a Wilson 95% interval on the success rate."""
    rows: list[dict[str, Any]] = []
    for tier in sorted({task.tier for task in report.tasks}):
        group = [task for task in report.tasks if task.tier == tier]
        episodes = sum(task.n for task in group)
        successes = sum(task.c for task in group)
        low, high = wilson_interval(successes, episodes)
        rows.append(
            {
                "tier": tier,
                "tasks": len(group),
                "successes": successes,
                "episodes": episodes,
                "rate": successes / episodes if episodes else None,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return rows


def _identical_action_tasks(
    trajectories: list[Trajectory] | None, n_trials: int
) -> dict[str, int] | None:
    """Tasks whose trials all took the same action sequence (tool names and arguments).

    Returns None when there are no trajectories or only one trial per task.
    """
    if not trajectories or n_trials < 2:
        return None
    by_task: dict[str, set[tuple[Any, ...]]] = {}
    for traj in trajectories:
        sequence = tuple(
            (
                step.action.kind,
                step.action.tool_call.name if step.action.tool_call else None,
                json.dumps(step.action.tool_call.arguments, sort_keys=True)
                if step.action.tool_call
                else "",
            )
            for step in traj.steps
        )
        by_task.setdefault(traj.task_id, set()).add(sequence)
    identical = sum(1 for sequences in by_task.values() if len(sequences) == 1)
    return {"identical": identical, "tasks": len(by_task)}


def _run_record(report: RunReport, path: Path) -> dict[str, Any]:
    trajectories = load_trajectories(path.parent)
    env_path = path.parent / "env.json"
    gap = _gap(report)
    pass_at: dict[str, Any] = {}
    pass_hat: dict[str, Any] = {}
    for k in _k_values(report):
        at = _summary_at(report, "pass_at_k", k)
        hat = _summary_at(report, "pass_hat_k", k)
        if at:
            pass_at[str(k)] = at
        if hat:
            pass_hat[str(k)] = hat
    n_episodes = sum(stats.n for stats in report.tasks)
    return {
        "run_id": report.run_id,
        "label": SYNTHETIC if is_synthetic(report.env_info) else "model",
        "complete": report.complete,
        "stop_reason": report.stop_reason,
        "reset_at": report.reset_at,
        "episodes_total": report.episodes_total
        if report.episodes_total is not None
        else len(report.tasks) * report.config.n_trials,
        "agent": _agent_label(report),
        "agent_kind": report.config.agent.kind,
        "transport": report.config.transport,
        "temperature": report.env_info.get("agent", {}).get("temperature")
        if isinstance(report.env_info.get("agent"), dict)
        else None,
        "prompt_variant": report.env_info.get("agent", {}).get("prompt_variant")
        if isinstance(report.env_info.get("agent"), dict)
        else None,
        "tasks": len(report.tasks),
        "trials_per_task": report.config.n_trials,
        "episodes": n_episodes,
        "base_seed": report.config.base_seed,
        "pass_at_k": pass_at,
        "pass_hat_k": pass_hat,
        "failure_shares": _failure_shares(report),
        "tier_failure_shares": _tier_failure_shares(report),
        "tiers": _tier_rows(report),
        "gap": {"mean": gap[0], "ci_low": gap[1], "ci_high": gap[2]} if gap else None,
        "identical_action_tasks": _identical_action_tasks(trajectories, report.config.n_trials),
        "tokens": {
            "input_total": report.tokens.get("input_total"),
            "output_total": report.tokens.get("output_total"),
            "model_calls": report.tokens.get("model_calls"),
        },
        "costs": _episode_costs(report, trajectories),
        "env_json": _rel(env_path) if env_path.is_file() else None,
        "report_json": _rel(path),
    }


def _load_calibration() -> dict[str, Any] | None:
    if not CALIBRATION.is_file():
        return None
    data: dict[str, Any] = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    e1 = data["e1"]
    e2 = data["e2"]
    return {
        "label": data.get("label", SYNTHETIC),
        "e1": {
            "n_trials": e1["n_trials"],
            "levels": [
                {
                    "p": level["p"],
                    "max_abs_error": level["max_abs_error"],
                    "max_abs_z": level["max_abs_z"],
                }
                for level in e1["levels"]
            ],
            "max_abs_error": e1["max_abs_error"],
            "max_abs_z": e1["max_abs_z"],
        },
        "e2": {
            "pairs": e2["pairs"],
            "trials_per_task": e2.get("trials_per_task"),
            "policy": e2["policy"],
            "aa": e2["aa"],
            "ab": e2["ab"],
        },
        "env_json": _rel(CALIBRATION.parent / "env.json")
        if (CALIBRATION.parent / "env.json").is_file()
        else None,
        "validate_json": _rel(CALIBRATION),
    }


GATE_LABELS: dict[str, str] = {
    "e5_v1_vs_v2.txt": (
        "Pre-registered E5 comparison: baseline E3 1.5B (prompt v1), candidate E5 (prompt v2)."
    ),
    "e5_reversed_v2_baseline_v1_candidate.txt": (
        "POST-HOC demonstration, NOT pre-registered: the same two E5 runs with the roles "
        "swapped (baseline prompt v2, candidate prompt v1). It shows the gate firing on a "
        "drop; it is not a second E5 result."
    ),
}


def _load_gates() -> list[dict[str, str]]:
    """Gate outputs from results/gates, verbatim, each with a label from GATE_LABELS.

    A gate file without a label is refused, so that no gate output is shown without saying what
    comparison it is.
    """
    if not GATES_DIR.is_dir():
        return []
    gates: list[dict[str, str]] = []
    for path in sorted(GATES_DIR.glob("*.txt"), key=lambda p: ("reversed" in p.name, p.name)):
        if path.name not in GATE_LABELS:
            raise ValueError(f"{_rel(path)} has no label in GATE_LABELS")
        gates.append(
            {
                "file": _rel(path),
                "label": GATE_LABELS[path.name],
                "text": path.read_text(encoding="utf-8").rstrip(),
            }
        )
    return gates


def collect_results() -> dict[str, Any]:
    """Load every committed result once. Nothing here is typed by hand.

    Returns:
        ``reports`` (path and RunReport per run, sorted by path), ``runs`` (the README records),
        ``calibration`` and ``gates``.
    """
    reports: list[tuple[Path, RunReport]] = [
        (path, load_report(path)) for path in sorted(RESULTS.rglob("report.json"))
    ]
    runs = sorted(
        (_run_record(report, path) for path, report in reports), key=lambda r: r["run_id"]
    )
    return {
        "reports": reports,
        "runs": runs,
        "calibration": _load_calibration(),
        "gates": _load_gates(),
    }


def _fmt_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _calibration_block(cal: dict[str, Any]) -> list[str]:
    e1, e2 = cal["e1"], cal["e2"]
    lines = [
        f"**{cal['label']}: estimator and gate calibration** (scripted NoisyOracle agents; "
        "this checks the tool, not a model).",
        "",
        f"E1 (n={e1['n_trials']} trials per task, k=1..5). Max absolute error of pass^k against "
        "the analytic value, and the largest z-score.",
        "",
        "| per-step error p | max abs error | max abs z |",
        "|---|---|---|",
    ]
    for level in e1["levels"]:
        lines.append(
            f"| {level['p']:.2f} | {level['max_abs_error']:.4f} | {level['max_abs_z']:.2f} |"
        )
    aa, ab = e2["aa"], e2["ab"]
    lines += [
        "",
        f"E2 gate (k={e2['policy']['k']}, max_drop={e2['policy']['max_drop']}, "
        f"{e2['trials_per_task']} trials per task, {e2['pairs']} pairs each). Wilson 95% "
        "intervals.",
        "",
        "| comparison | rate | count | 95% CI |",
        "|---|---|---|---|",
        f"| false alarm, A/A (p 0.10 vs 0.10) | {_fmt_pct(aa['rate'])} | "
        f"{aa['false_alarms']}/{e2['pairs']} | {_fmt_pct(aa['ci95'][0])} to "
        f"{_fmt_pct(aa['ci95'][1])} |",
        f"| power, A/B (p 0.05 vs 0.15) | {_fmt_pct(ab['power'])} | "
        f"{ab['detections']}/{e2['pairs']} | {_fmt_pct(ab['ci95'][0])} to "
        f"{_fmt_pct(ab['ci95'][1])} |",
    ]
    if cal.get("env_json"):
        lines += ["", f"Environment record: [`{cal['env_json']}`]({cal['env_json']})."]
    return lines


def _run_block(run: dict[str, Any]) -> list[str]:
    head = f"**{run['label']}: `{run['run_id']}`**"
    if not run["complete"]:
        head = f"**{run['label']} PARTIAL: `{run['run_id']}`**"
    costs = run["costs"]
    lines = [
        head,
        "",
    ]
    if not run["complete"]:
        reset = f" The provider limit resets at {run['reset_at']} (UTC)." if run["reset_at"] else ""
        lines += [
            f"> **PARTIAL RUN.** Stopped before every episode finished ({run['stop_reason']}). "
            f"{run['episodes']} of {run['episodes_total']} episodes are included. This is not a "
            f"full result.{reset}",
            "",
        ]
    lines += [
        f"Agent: {run['agent']}. Transport: {run['transport']}. "
        f"Temperature: {run['temperature']}. Prompt: {run['prompt_variant']}. "
        f"{run['tasks']} tasks x {run['trials_per_task']} trials = {run['episodes']} episodes. "
        f"Base seed {run['base_seed']}.",
        "",
        "| k | pass@k (mean, 95% CI) | pass^k (mean, 95% CI) |",
        "|---|---|---|",
    ]
    for k in sorted(set(run["pass_at_k"]) | set(run["pass_hat_k"]), key=int):
        lines.append(
            f"| {k} | {_interval(run['pass_at_k'].get(k))} | "
            f"{_interval(run['pass_hat_k'].get(k))} |"
        )
    if run["failure_shares"]:
        shares = ", ".join(f"{name} {_fmt_pct(v)}" for name, v in run["failure_shares"].items())
        lines += ["", f"Failure categories, share of all episodes: {shares}."]
    if run["gap"]:
        gap = run["gap"]
        lines += [
            "",
            "Gap pass@1 - pass^5: "
            f"{gap['mean']:.3f} (95% CI [{gap['ci_low']:.3f}, {gap['ci_high']:.3f}]), "
            "over tasks.",
        ]
    if run["tiers"]:
        lines += [
            "",
            "| tier | tasks | successes / episodes | success rate (Wilson 95% CI) |",
            "|---|---|---|---|",
        ]
        for tier in run["tiers"]:
            rate = _fmt_pct(tier["rate"]) if tier["rate"] is not None else "n/a"
            lines.append(
                f"| {tier['tier']} | {tier['tasks']} | {tier['successes']} / {tier['episodes']} "
                f"| {rate} [{_fmt_pct(tier['ci_low'])}, {_fmt_pct(tier['ci_high'])}] |"
            )
    if run["identical_action_tasks"]:
        same = run["identical_action_tasks"]
        lines += [
            "",
            f"Tasks whose trials all took the same action sequence: {same['identical']} of "
            f"{same['tasks']}.",
        ]
    if costs["median_wall_s"] is not None:
        tokens = run["tokens"]
        lines += [
            "",
            f"Median per episode: {costs['median_wall_s']:.1f} s wall, "
            f"{costs['median_output_tokens']:.0f} output tokens. Totals: "
            f"{tokens['input_total']} input and {tokens['output_total']} output tokens over "
            f"{tokens['model_calls']} model calls. Timing was measured on a shared laptop, not a "
            "benchmark (see docs/deviations.md, D29).",
        ]
    links = [f"[report]({run['report_json']})"]
    if run["env_json"]:
        links.append(f"[env]({run['env_json']})")
    lines += ["", " | ".join(links)]
    return lines


def _summary_mean(report: RunReport, metric: str, k: int) -> float | None:
    value = report.summary.get(metric, {}).get(k)
    return float(value["mean"]) if isinstance(value, dict) else None


def _episodes(report: RunReport) -> int:
    return sum(stats.n for stats in report.tasks)


def _short_model(report: RunReport) -> str:
    agent = report.env_info.get("agent")
    if isinstance(agent, dict) and agent.get("repo_id"):
        name = str(agent["repo_id"]).split("/")[-1]
        return name.removesuffix("-GGUF")
    return report.config.agent.kind


def _leader(report: RunReport) -> str:
    """The failure category with the largest share of episodes (success is not a failure)."""
    totals = {name: count for name, count in report.failure_totals.items() if name != "success"}
    episodes = _episodes(report)
    if not totals or episodes == 0:
        return "no failures recorded"
    name, count = max(sorted(totals.items()), key=lambda kv: kv[1])
    return f"{name} {_fmt_pct(count / episodes)} of episodes"


def _gap(report: RunReport) -> tuple[float, float, float] | None:
    """Mean of the per-task pass@1 minus pass^5, with its bootstrap 95% interval over tasks."""
    values = [
        task.pass_at_k[1] - task.pass_hat_k[5]
        for task in report.tasks
        if 1 in task.pass_at_k and 5 in task.pass_hat_k
    ]
    if not values:
        return None
    mean, low, high = metrics.bootstrap_ci(values, seed=0)
    return mean, low, high


def _item(
    item_id: str,
    label: str,
    value: str,
    detail: str,
    *,
    n: int | None,
    ci: str | None,
    synthetic: bool,
    featured: bool,
    source: str,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "label": label,
        "value": value,
        "detail": detail,
        "n": n,
        "ci": ci,
        "synthetic": synthetic,
        "featured": featured,
        "source": source,
    }


def _gap_item(
    item_id: str, label: str, path: Path, report: RunReport, *, featured: bool
) -> dict[str, Any] | None:
    gap = _gap(report)
    p1 = _summary_mean(report, "pass_at_k", 1)
    p5 = _summary_mean(report, "pass_hat_k", 5)
    if gap is None or p1 is None or p5 is None:
        return None
    mean, low, high = gap
    prefix = "" if report.complete else "PARTIAL: "
    detail = (
        f"{_episodes(report)} episodes, {len(report.tasks)} tasks, {_short_model(report)}. "
        f"Failure leader: {_leader(report)}."
    )
    if not report.complete:
        detail += f" Stopped early: {report.stop_reason}."
    return _item(
        item_id,
        label,
        f"{prefix}pass@1 {p1:.3f} vs pass^5 {p5:.3f}, gap {mean:.3f}",
        detail,
        n=_episodes(report),
        ci=f"95% CI on the gap [{low:.3f}, {high:.3f}]",
        synthetic=is_synthetic(report.env_info),
        featured=featured,
        source=_rel(path),
    )


def _find(reports: list[tuple[Path, RunReport]], prefix: str) -> tuple[Path, RunReport] | None:
    for path, report in reports:
        if report.run_id.startswith(prefix) and "smoke" not in report.run_id:
            return path, report
    return None


def _e5_item(reports: list[tuple[Path, RunReport]]) -> dict[str, Any] | None:
    baseline = _find(reports, "e3-qwen2.5-1.5b")
    candidate = _find(reports, "e5-qwen2.5-1.5b")
    if baseline is None or candidate is None:
        return None
    path, cand = candidate
    if not baseline[1].complete or not cand.complete:
        return _item(
            "e5-regression",
            "E5 prompt v2 vs v1 (1.5B), regression gate",
            "not evaluated: partial run",
            "The gate refuses incomplete reports. Re-run the missing episodes first.",
            n=_episodes(cand),
            ci=None,
            synthetic=False,
            featured=False,
            source=_rel(path),
        )
    try:
        decision = gate(baseline[1], cand, load_policy(GATE_POLICY))
    except ValueError:
        return None  # different task sets or trial counts: no verdict to report
    return _item(
        "e5-regression",
        "E5 prompt v2 vs v1 (1.5B), regression gate",
        "PASS" if decision.passed else "FAIL",
        " | ".join(decision.reasons),
        n=_episodes(cand),
        ci=(
            f"delta 95% CI [{decision.ci_low:+.3f}, {decision.ci_high:+.3f}] "
            f"(delta {decision.delta:+.3f})"
        ),
        synthetic=False,
        featured=False,
        source=_rel(path),
    )


def _e7_item(reports: list[tuple[Path, RunReport]]) -> dict[str, Any] | None:
    found = _find(reports, "e7-")
    if found is None:
        return None
    path, report = found
    if not report.complete:
        done = report.episodes_done if report.episodes_done is not None else _episodes(report)
        total = report.episodes_total
        progress = f"{done} of {total}" if total is not None else f"{done}"
        reset = f" The provider limit resets at {report.reset_at} (UTC)." if report.reset_at else ""
        return _item(
            "e7-hosted",
            "E7 hosted free-tier model (descriptive)",
            f"PARTIAL: {progress} episodes",
            f"Stopped: {report.stop_reason}.{reset} The full run was not completed on the free "
            "quota; this is not a full result.",
            n=done,
            ci=None,
            synthetic=False,
            featured=False,
            source=_rel(path),
        )
    gap = _gap_item(
        "e7-hosted", "E7 hosted free-tier model (descriptive)", path, report, featured=False
    )
    return gap


def _e1_item(calibration: dict[str, Any] | None) -> dict[str, Any] | None:
    if calibration is None:
        return None
    e1 = calibration["e1"]
    noise = ", ".join(f"{level['p']:.2f}" for level in e1["levels"])
    return _item(
        "e1-estimator",
        "E1 estimator, max absolute error of pass^k",
        f"max abs error {e1['max_abs_error']:.4f}",
        f"Over k = 1..5 and per-step error p in {{{noise}}}, {e1['n_trials']} trials per task. "
        f"Largest |z| {e1['max_abs_z']:.2f}; a value below 3.0 is consistent with the analytic "
        "pass^k. Calibrates the tool, not a model.",
        n=None,
        ci=None,
        synthetic=calibration["label"] == SYNTHETIC,
        featured=False,
        source=_rel(CALIBRATION),
    )


def _e2_item(calibration: dict[str, Any] | None) -> dict[str, Any] | None:
    if calibration is None:
        return None
    e2 = calibration["e2"]
    aa, ab = e2["aa"], e2["ab"]
    return _item(
        "e2-gate",
        "E2 gate calibration: false-alarm rate and power",
        f"false alarm {_fmt_pct(aa['rate'])}, power {_fmt_pct(ab['power'])}",
        f"Gate k={e2['policy']['k']}, max_drop={e2['policy']['max_drop']}; {e2['pairs']} A/A "
        f"and {e2['pairs']} A/B pairs at {e2.get('trials_per_task')} trials per task. Calibrates "
        "the tool, not a model.",
        n=e2["pairs"],
        ci=(
            f"false alarm 95% CI [{_fmt_pct(aa['ci95'][0])}, {_fmt_pct(aa['ci95'][1])}]; "
            f"power 95% CI [{_fmt_pct(ab['ci95'][0])}, {_fmt_pct(ab['ci95'][1])}]"
        ),
        synthetic=calibration["label"] == SYNTHETIC,
        featured=False,
        source=_rel(CALIBRATION),
    )


def _environment_text() -> str:
    """Hardware and platform of the committed calibration, read from its env.json."""
    env_path = CALIBRATION.parent / "env.json"
    if not env_path.is_file():
        return "labstock environment served over MCP stdio; CPU only; hardware not recorded."
    env = json.loads(env_path.read_text(encoding="utf-8"))
    os_info = env.get("os") if isinstance(env.get("os"), dict) else {}
    return (
        f"{env.get('cpu_model', 'unknown CPU')} ({env.get('cores_physical', '?')} physical "
        f"cores), {os_info.get('arch', '?')}, macOS {os_info.get('macos_version', '?')}; CPU "
        "only. labstock environment served over MCP stdio. Results apply to this hardware."
    )


def build_items(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Headline items in priority order, each present only when its source result exists.

    Priority: E3 0.5B and 1.5B gaps (featured), E5 verdict, E7, E1, E2, then the optional
    Qwen3 arm. At most ``MAX_ITEMS`` are kept, so the last candidates are dropped first.
    """
    reports: list[tuple[Path, RunReport]] = results["reports"]
    calibration = results["calibration"]
    candidates: list[dict[str, Any] | None] = []
    for item_id, prefix, label in (
        ("e3-qwen2.5-0.5b", "e3-qwen2.5-0.5b", "E3 Qwen2.5-0.5B-Instruct (Q4_K_M)"),
        ("e3-qwen2.5-1.5b", "e3-qwen2.5-1.5b", "E3 Qwen2.5-1.5B-Instruct (Q4_K_M)"),
    ):
        found = _find(reports, prefix)
        candidates.append(
            _gap_item(item_id, label, found[0], found[1], featured=True) if found else None
        )
    candidates.append(_e5_item(reports))
    e7 = _e7_item(reports)
    candidates.append(e7)
    candidates.append(_e1_item(calibration))
    candidates.append(_e2_item(calibration))
    qwen3 = _find(reports, "e3-qwen3-1.7b")
    candidates.append(
        _gap_item(
            "e3-qwen3-1.7b",
            "E3 Qwen3-1.7B Q8_0 (optional arm)",
            qwen3[0],
            qwen3[1],
            featured=True,
        )
        if qwen3
        else None
    )
    return [item for item in candidates if item is not None][:MAX_ITEMS]


def build_headline(results: dict[str, Any] | None = None) -> dict[str, Any]:
    """The hub's standard contract for ``results/headline.json``, from results JSON only.

    Args:
        results: Output of ``collect_results`` (collected here when omitted).

    Returns:
        A dict with ``schema_version``, ``tool``, ``tagline``, ``quickstart``, ``environment``,
        ``items`` (at most ``MAX_ITEMS``) and ``links``.
    """
    if results is None:
        results = collect_results()
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "tool": TOOL,
        "tagline": TAGLINE,
        "quickstart": QUICKSTART,
        "environment": _environment_text(),
        "items": build_items(results),
        "links": dict(LINKS),
    }


def _headline_list(items: list[dict[str, Any]]) -> list[str]:
    """Bullets for the README, built from the same item strings as headline.json."""
    if not items:
        return []
    lines = ["**Headline** (from `results/headline.json`):", ""]
    for item in items:
        flag = "SYNTHETIC: " if item["synthetic"] else ""
        ci = f" {item['ci']}." if item["ci"] else ""
        lines.append(f"- {flag}**{item['label']}**: {item['value']}.{ci} {item['detail']}")
    return lines + [""]


def _block(results: dict[str, Any], contract: dict[str, Any]) -> str:
    lines: list[str] = _headline_list(contract["items"])
    if results["calibration"]:
        lines += _calibration_block(results["calibration"]) + [""]
    for run in results["runs"]:
        lines += _run_block(run) + [""]
    for gate_file in results["gates"]:
        lines += [
            f"{gate_file['label']} Gate output, [`{gate_file['file']}`]({gate_file['file']}):",
            "",
            "```",
            gate_file["text"],
            "```",
            "",
        ]
    if not results["runs"] and not results["calibration"]:
        lines.append(NO_RESULTS)
    return "\n".join(lines).rstrip() + "\n"


def _replace_block(text: str, body: str) -> str:
    if text.count(START) != 1 or text.count(END) != 1:
        raise ValueError(f"README must contain exactly one {START} and one {END} marker")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    return f"{head}{START}\n{body}{END}{tail}"


def _headline_text(headline: dict[str, Any]) -> str:
    return json.dumps(headline, indent=2, sort_keys=True) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if outputs are stale")
    parser.add_argument("--headline", action="store_true", help="write only headline.json")
    args = parser.parse_args(argv)

    results = collect_results()
    contract = build_headline(results)
    headline_text = _headline_text(contract)
    body = _block(results, contract)
    readme_text = README.read_text(encoding="utf-8")
    new_readme = _replace_block(readme_text, body)

    if args.check:
        stale: list[str] = []
        if not HEADLINE.is_file() or HEADLINE.read_text(encoding="utf-8") != headline_text:
            stale.append(_rel(HEADLINE))
        if new_readme != readme_text:
            stale.append("README.md (results block)")
        if stale:
            print("stale results output, run `make report`: " + ", ".join(stale))
            return 1
        print(
            f"results up to date: {len(results['runs'])} run(s), "
            f"{len(contract['items'])} headline item(s), "
            f"calibration {'present' if results['calibration'] else 'absent'}"
        )
        return 0

    RESULTS.mkdir(parents=True, exist_ok=True)
    HEADLINE.write_text(headline_text, encoding="utf-8")
    if not args.headline:
        README.write_text(new_readme, encoding="utf-8")
    print(f"wrote {_rel(HEADLINE)}" + ("" if args.headline else " and README.md block"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
