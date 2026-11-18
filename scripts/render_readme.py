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

from passk.diff import is_synthetic, load_report, load_trajectories  # noqa: E402
from passk.schema import RunReport, Trajectory  # noqa: E402


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
    total = sum(report.failure_totals.values())
    if total == 0:
        return {}
    return {name: count / total for name, count in sorted(report.failure_totals.items())}


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


def _run_record(report: RunReport, path: Path) -> dict[str, Any]:
    trajectories = load_trajectories(path.parent)
    env_path = path.parent / "env.json"
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


def _load_gates() -> list[dict[str, str]]:
    if not GATES_DIR.is_dir():
        return []
    return [
        {"file": _rel(path), "text": path.read_text(encoding="utf-8").rstrip()}
        for path in sorted(GATES_DIR.glob("*.txt"))
    ]


def build_headline() -> dict[str, Any]:
    """Collect every number the README quotes from the committed results."""
    runs: list[dict[str, Any]] = []
    for path in sorted(RESULTS.rglob("report.json")):
        report = load_report(path)
        runs.append(_run_record(report, path))
    runs.sort(key=lambda r: r["run_id"])
    return {
        "schema": "passk-headline-1",
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
        lines += ["", f"Failure shares (all episodes): {shares}."]
    if costs["median_wall_s"] is not None:
        lines.append(
            f"Median per episode: {costs['median_wall_s']:.1f} s wall, "
            f"{costs['median_output_tokens']:.0f} output tokens."
        )
    links = [f"[report]({run['report_json']})"]
    if run["env_json"]:
        links.append(f"[env]({run['env_json']})")
    lines += ["", " | ".join(links)]
    return lines


def _block(headline: dict[str, Any]) -> str:
    lines: list[str] = []
    if headline["calibration"]:
        lines += _calibration_block(headline["calibration"]) + [""]
    for run in headline["runs"]:
        lines += _run_block(run) + [""]
    for gate in headline["gates"]:
        lines += [f"Gate output, [`{gate['file']}`]({gate['file']}):", "", "```", gate["text"]]
        lines += ["```", ""]
    if not headline["runs"] and not headline["calibration"]:
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

    headline = build_headline()
    headline_text = _headline_text(headline)
    body = _block(headline)
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
            f"results up to date: {len(headline['runs'])} run(s), "
            f"calibration {'present' if headline['calibration'] else 'absent'}"
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
