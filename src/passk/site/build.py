"""Static site and single-run report rendering.

``render_report`` turns one run into Markdown, JSON or a standalone HTML page.
``build_site`` writes ``index.html``, ``llms.txt`` and, for every run, ``runs/<run_id>/``
with ``index.html`` and a copy of ``report.json``. Pages are self-contained: CSS and SVG
are inline and nothing is loaded from the network. A run whose ``env_info`` marks it as
synthetic gets a SYNTHETIC banner.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from passk.diff import is_synthetic, load_report, load_trajectories, mean_metric
from passk.schema import FailureCategory, RunReport, Trajectory
from passk.site.charts import (
    Curve,
    CurvePoint,
    HeatCell,
    HeatRow,
    bars_svg,
    curves_svg,
    heatmap_svg,
)

_HERE = Path(__file__).parent
_TEMPLATES = _HERE / "templates"
_STYLE = _HERE / "static" / "style.css"
SYNTHETIC_BANNER = (
    " scripted or fixture data, not a model result. Do not read it beside model results."
)
_SHORT_LIMIT = 300


@dataclass(frozen=True)
class _Entry:
    report: RunReport
    trajectories: list[Trajectory] | None
    slug: str


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATES)),
        # Every template here is HTML, but the names end in .html.j2, which
        # select_autoescape's extension list would miss. Escape everything by default.
        autoescape=True,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _css() -> str:
    return _STYLE.read_text(encoding="utf-8")


def _slug(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", run_id) or "run"


def _short(value: Any, limit: int = _SHORT_LIMIT) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _pass_rows(report: RunReport) -> list[dict[str, Any]]:
    ci_table = report.summary.get("pass_hat_k", {})
    rows: list[dict[str, Any]] = []
    for k in sorted(set(report.config.k_values)):
        hat = mean_metric(report, "pass_hat_k", k)
        if hat is None:
            continue
        at = mean_metric(report, "pass_at_k", k)
        ci = ci_table.get(k)
        rows.append(
            {
                "k": k,
                "pass_hat_k": hat,
                "pass_at_k": at,
                "ci_low": ci.get("ci_low") if ci else None,
                "ci_high": ci.get("ci_high") if ci else None,
            }
        )
    return rows


def _failure_items(report: RunReport) -> list[tuple[str, int]]:
    known = [category.value for category in FailureCategory]
    extra = sorted(set(report.failure_totals) - set(known))
    return [(name, report.failure_totals.get(name, 0)) for name in known + extra]


def _heatmap(report: RunReport, trajectories: list[Trajectory] | None) -> str:
    if trajectories is None:
        rows = [
            HeatRow(
                label=task.task_id,
                cells=(
                    HeatCell(
                        kind="rate",
                        value=task.c / task.n if task.n else 0.0,
                        glyph="",
                        tooltip=f"{task.task_id}: {task.c} of {task.n} trials succeeded",
                    ),
                ),
                side_text=f"{task.c}/{task.n}",
            )
            for task in report.tasks
        ]
        return heatmap_svg(rows, ["pass rate"], "Success rate per task")
    index = {(t.task_id, t.trial): t for t in trajectories}
    trials = report.config.n_trials
    rows = []
    for task in report.tasks:
        cells: list[HeatCell] = []
        for trial in range(trials):
            record = index.get((task.task_id, trial))
            label = f"{task.task_id}, trial {trial}"
            if record is None:
                cells.append(HeatCell("missing", 0.0, "·", f"{label}: no record"))
            elif record.success:
                cells.append(HeatCell("pass", 1.0, "✓", f"{label}: success"))
            else:
                cells.append(HeatCell("fail", 0.0, "✗", f"{label}: {record.failure.value}"))
        rows.append(HeatRow(label=task.task_id, cells=tuple(cells)))
    return heatmap_svg(rows, [str(i) for i in range(trials)], "Task by trial outcome")


def _trajectory_view(traj: Trajectory) -> dict[str, Any]:
    steps: list[dict[str, Any]] = []
    for step in traj.steps:
        action = step.action
        call = action.tool_call
        result = step.result
        steps.append(
            {
                "index": step.index,
                "kind": action.kind,
                "tool": call.name if call else None,
                "arguments": _short(call.arguments) if call else None,
                "ok": result.ok if result else None,
                "content": _short(result.content) if result else None,
                "error": (result.error or None) if result else None,
                "message": action.message,
                "latency": f"{step.latency_s:.3f} s",
            }
        )
    return {
        "title": f"{traj.task_id} · trial {traj.trial}",
        "success": traj.success,
        "status": "success" if traj.success else traj.failure.value,
        "violations": list(traj.violations),
        "final_state": _short(traj.final_state, 500),
        "steps": steps,
    }


def _run_context(report: RunReport, trajectories: list[Trajectory] | None) -> dict[str, Any]:
    rows = _pass_rows(report)
    curves: list[Curve] = [
        Curve(
            label="pass^k",
            css_class="c0",
            points=tuple(
                CurvePoint(r["k"], r["pass_hat_k"], r["ci_low"], r["ci_high"]) for r in rows
            ),
        )
    ]
    at_points = tuple(
        CurvePoint(r["k"], r["pass_at_k"]) for r in rows if r["pass_at_k"] is not None
    )
    if at_points:
        curves.append(Curve(label="pass@k", css_class="c1", points=at_points, dashed=True))
    k_values = sorted(set(report.config.k_values))
    task_rows = [
        {
            "task_id": task.task_id,
            "tier": task.tier,
            "c": task.c,
            "n": task.n,
            "per_k": [task.pass_hat_k.get(k) for k in k_values],
        }
        for task in report.tasks
    ]
    views = (
        [_trajectory_view(t) for t in sorted(trajectories, key=lambda t: (t.task_id, t.trial))]
        if trajectories is not None
        else None
    )
    return {
        "synthetic": is_synthetic(report.env_info),
        "run_id": report.run_id,
        "agent_kind": report.config.agent.kind,
        "transport": report.config.transport,
        "n_tasks": len(report.tasks),
        "n_trials": report.config.n_trials,
        "wall_s": f"{report.wall_s:.1f}",
        "k_values": k_values,
        "summary_rows": rows,
        "task_rows": task_rows,
        "curves_svg": curves_svg(curves, "pass^k and pass@k against k"),
        "heatmap_svg": _heatmap(report, trajectories),
        "failure_svg": bars_svg(_failure_items(report), "Failure categories"),
        "trajectories": views,
        "env_json": json.dumps(report.env_info, indent=2, sort_keys=True, default=str),
        "tokens_json": json.dumps(report.tokens, indent=2, sort_keys=True, default=str),
    }


def _run_html(
    report: RunReport,
    trajectories: list[Trajectory] | None,
    *,
    report_link: str | None,
) -> str:
    context = _run_context(report, trajectories)
    template = _env().get_template("run.html.j2")
    return template.render(
        page_title=f"passk run {report.run_id}",
        heading=f"Run {report.run_id}",
        css=_css(),
        banner_text=SYNTHETIC_BANNER,
        report_link=report_link,
        **context,
    )


def _markdown(report: RunReport) -> str:
    rows = _pass_rows(report)
    synthetic = is_synthetic(report.env_info)
    lines = [f"# passk run {report.run_id}", ""]
    if synthetic:
        lines += [f"> **SYNTHETIC**:{SYNTHETIC_BANNER}", ""]
    lines += [
        f"- Agent: `{report.config.agent.kind}`",
        f"- Transport: `{report.config.transport}`",
        f"- Tasks: {len(report.tasks)}",
        f"- Trials per task: {report.config.n_trials}",
        f"- Wall time: {report.wall_s:.1f} s",
        "",
        "## Summary",
        "",
        "| k | pass^k | CI from report | pass@k |",
        "|---|---|---|---|",
    ]
    for row in rows:
        ci = (
            f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}]"
            if row["ci_low"] is not None and row["ci_high"] is not None
            else "n/a"
        )
        at = f"{row['pass_at_k']:.3f}" if row["pass_at_k"] is not None else "n/a"
        lines.append(f"| {row['k']} | {row['pass_hat_k']:.3f} | {ci} | {at} |")
    lines += ["", "## Failures", "", "| category | trials |", "|---|---|"]
    lines += [f"| {name} | {count} |" for name, count in _failure_items(report)]
    k_values = sorted(set(report.config.k_values))
    header = "| task | tier | successes | " + " | ".join(f"pass^{k}" for k in k_values) + " |"
    lines += ["", "## Tasks", "", header, "|" + "---|" * (3 + len(k_values))]
    for task in report.tasks:
        cells = [f"{task.pass_hat_k.get(k, float('nan')):.3f}" for k in k_values]
        lines.append(
            f"| {task.task_id} | {task.tier} | {task.c}/{task.n} | " + " | ".join(cells) + " |"
        )
    return "\n".join(lines) + "\n"


def render_report(
    report: RunReport,
    fmt: Literal["md", "json", "html"],
    trajectories: list[Trajectory] | None = None,
) -> str:
    """Render one run report.

    Args:
        report: Run report to render.
        fmt: ``md`` for Markdown, ``json`` for the report JSON, ``html`` for a standalone page.
        trajectories: Optional trajectories; when given, the HTML page includes the
            heatmap of individual trials and the collapsible trajectory viewer.

    Returns:
        The rendered document as text.

    Raises:
        ValueError: If ``fmt`` is not one of the supported formats.
    """
    if fmt == "json":
        return report.model_dump_json(indent=2) + "\n"
    if fmt == "md":
        return _markdown(report)
    if fmt == "html":
        return _run_html(report, trajectories, report_link=None)
    raise ValueError(f"unknown format {fmt!r}; use md, json or html")


def _collect(source: Path | Sequence[RunReport]) -> list[_Entry]:
    entries: list[_Entry] = []
    if isinstance(source, Path):
        if source.is_file():
            paths = [source]
        elif source.is_dir():
            paths = sorted(source.rglob("report.json"))
        else:
            raise FileNotFoundError(f"no such file or directory: {source}")
        for path in paths:
            report = load_report(path)
            entries.append(_Entry(report, load_trajectories(path.parent), _slug(report.run_id)))
    else:
        for report in source:
            entries.append(_Entry(report, None, _slug(report.run_id)))
    if not entries:
        raise ValueError(f"no report.json found under {source}")
    slugs = [entry.slug for entry in entries]
    duplicates = sorted({s for s in slugs if slugs.count(s) > 1})
    if duplicates:
        raise ValueError(f"duplicate run ids in site input: {duplicates}")
    return entries


def _index_html(entries: list[_Entry]) -> str:
    all_k = sorted({k for e in entries for k in e.report.config.k_values})
    runs: list[dict[str, Any]] = []
    curves: list[Curve] = []
    for index, entry in enumerate(entries):
        rows = {r["k"]: r["pass_hat_k"] for r in _pass_rows(entry.report)}
        runs.append(
            {
                "run_id": entry.report.run_id,
                "slug": entry.slug,
                "synthetic": is_synthetic(entry.report.env_info),
                "agent_kind": entry.report.config.agent.kind,
                "n_tasks": len(entry.report.tasks),
                "n_trials": entry.report.config.n_trials,
                "cells": [rows.get(k) for k in all_k],
            }
        )
        points = tuple(CurvePoint(k, rows[k]) for k in sorted(rows))
        if points:
            curves.append(Curve(entry.report.run_id, f"c{index % 6}", points))
    any_synthetic = any(run["synthetic"] for run in runs)
    template = _env().get_template("index.html.j2")
    return template.render(
        page_title="passk reports",
        heading="Reliability reports",
        css=_css(),
        synthetic=any_synthetic,
        banner_text=SYNTHETIC_BANNER,
        runs=runs,
        k_values=all_k,
        curves_svg=curves_svg(curves, "pass^k against k for every run"),
    )


def _llms_text(entries: list[_Entry]) -> str:
    lines = [
        "# passk reports",
        "",
        "> Pass^k reliability reports for tool-using agents. Every number is computed from",
        "> the report.json file next to it. Runs marked SYNTHETIC are scripted or fixture data.",
        "",
        "## Runs",
        "",
    ]
    for entry in entries:
        report = entry.report
        label = " (SYNTHETIC)" if is_synthetic(report.env_info) else ""
        pieces = [f"pass^{r['k']} {r['pass_hat_k']:.3f}" for r in _pass_rows(report)]
        lines.append(f"- {report.run_id}{label}: " + ", ".join(pieces))
        lines.append(f"  - page: runs/{entry.slug}/index.html")
        lines.append(f"  - data: runs/{entry.slug}/report.json")
    lines.append("")
    return "\n".join(lines)


def build_site(results_dir: Path | Sequence[RunReport], out_dir: Path) -> Path:
    """Build the static site from every committed report under ``results_dir``.

    Args:
        results_dir: A directory searched recursively for ``report.json`` files, a single
            report file, or a sequence of in-memory RunReports (no trajectory viewer).
        out_dir: Output directory, created if missing. Existing files are overwritten.

    Returns:
        The path of the written ``index.html``.

    Raises:
        FileNotFoundError: If ``results_dir`` does not exist.
        ValueError: If no report is found or two runs map to the same page directory.
    """
    entries = _collect(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        run_dir = out_dir / "runs" / entry.slug
        run_dir.mkdir(parents=True, exist_ok=True)
        html = _run_html(entry.report, entry.trajectories, report_link="report.json")
        (run_dir / "index.html").write_text(html, encoding="utf-8")
        (run_dir / "report.json").write_text(
            entry.report.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    index = out_dir / "index.html"
    index.write_text(_index_html(entries), encoding="utf-8")
    (out_dir / "llms.txt").write_text(_llms_text(entries), encoding="utf-8")
    return index
