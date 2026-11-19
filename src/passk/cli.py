"""Command-line interface: ``passk``.

Exit codes: 0 = pass, 1 = check failed or regression, 2 = usage or runtime error.
Every command accepts ``--json``. With ``--json`` the command prints its result as JSON;
errors are printed to stderr and, with ``--json``, also as a JSON object on stdout.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import typer
import yaml

import passk
from passk import runner, validate
from passk.diff import diff, load_report, load_trajectories, mean_metric
from passk.gate import gate, load_policy
from passk.numfmt import fixed
from passk.schema import AgentConfig, GatePolicy, RunConfig, RunReport, Trajectory
from passk.site.build import render_report

app = typer.Typer(
    name="passk",
    help="pass^k for tool-using agents: an offline, CI-gated reliability harness.",
    no_args_is_help=True,
    add_completion=False,
)

PACKAGED_TASKS = str(Path(passk.__file__).parent / "tasks" / "labstock" / "*.yaml")
DEMO_LABEL = "SYNTHETIC"


class CliError(Exception):
    """A usage or runtime error that should exit with code 2."""


def _incomplete_message(exc: runner.RunIncomplete) -> str:
    """Explain a stopped run: what was kept, how to continue, and when the quota resets."""
    kept = exc.report.episodes_done
    if kept is None:
        kept = sum(stats.n for stats in exc.report.tasks)
    total = exc.report.episodes_total
    progress = f"{kept} of {total}" if total is not None else f"{kept}"
    where = f" in {exc.run_dir}" if exc.run_dir is not None else ""
    parts = [
        f"run stopped before it finished: {exc.reason}.",
        f"{progress} episodes are kept{where}; a partial report (complete: false) was written.",
    ]
    if exc.reset_at is not None:
        parts.append(f"The provider limit resets at {exc.reset_at} (UTC).")
    parts.append("Re-run the same command to continue from the kept episodes.")
    return " ".join(parts)


def _error(json_output: bool, message: str) -> int:
    if json_output:
        typer.echo(json.dumps({"error": message, "exit_code": 2}, indent=2))
    typer.echo(f"passk: error: {message}", err=True)
    return 2


def _guard(json_output: bool, body: Callable[[], int]) -> None:
    """Run a command body and turn failures into exit code 2 with a clear message."""
    try:
        code = body()
    except NotImplementedError as exc:
        code = _error(json_output, f"not implemented yet ({exc})")
    except CliError as exc:
        code = _error(json_output, str(exc))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        code = _error(json_output, str(exc))
    except Exception as exc:
        if os.environ.get("PASSK_DEBUG"):
            raise
        code = _error(json_output, f"{type(exc).__name__}: {exc}")
    if code:
        raise typer.Exit(code=code)


def _emit_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, default=str))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)
    ]
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  ".join("-" * w for w in widths))
    lines += ["  ".join(c.ljust(widths[i]) for i, c in enumerate(r)) for r in rows]
    return "\n".join(lines)


def _pass_summary(report: RunReport) -> dict[int, float]:
    result: dict[int, float] = {}
    for k in sorted(set(report.config.k_values)):
        value = mean_metric(report, "pass_hat_k", k)
        if value is not None:
            result[k] = value
    return result


def _run_dir_or_report(path_text: str) -> tuple[RunReport, list[Trajectory] | None]:
    path = Path(path_text)
    return load_report(path), load_trajectories(path)


def _load_run_config(path: Path) -> RunConfig:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CliError(f"{path}: run config must be a YAML mapping")
    return RunConfig.model_validate(data)


@app.command("run")
def run_cmd(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to a RunConfig YAML file."),
    ],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
    smoke: Annotated[
        bool,
        typer.Option("--smoke", help="Check the setup: first tier-1 and tier-2 tasks, n=1."),
    ] = False,
    smoke_tasks: Annotated[
        int,
        typer.Option("--smoke-tasks", help="With --smoke: 1 (tier 1 only) or 2 (tiers 1 and 2)."),
    ] = 2,
    out: Annotated[
        str,
        typer.Option("--out", help="Directory that receives <run_id>/. Default: runs."),
    ] = "runs",
    resume: Annotated[
        bool | None,
        typer.Option(
            "--resume/--no-resume",
            help=(
                "Continue an interrupted run of the same configuration. Default: resume when "
                "the configuration matches; --no-resume refuses an existing run."
            ),
        ),
    ] = None,
) -> None:
    """Run tasks x trials for one configuration and write <out>/<run_id>/.

    Each finished episode is saved as it ends. If the run stops (for example on a provider
    quota), the finished episodes are kept and a partial report is written; re-running the same
    command later continues it.
    """

    def body() -> int:
        run_config = _load_run_config(Path(config))
        deps: runner.RunDeps | None = None
        if smoke:
            from passk.envs.base import load_tasks

            if smoke_tasks not in (1, 2):
                raise CliError("--smoke-tasks must be 1 or 2")
            tasks = sorted(load_tasks(run_config.tasks_glob), key=lambda t: t.id)
            picked = [t for t in tasks if t.tier == 1][:1]
            if smoke_tasks == 2:
                picked += [t for t in tasks if t.tier == 2][:1]
            deps = runner.RunDeps(tasks=picked)
            run_config = run_config.model_copy(
                update={"n_trials": 1, "run_id": f"{run_config.run_id}-smoke"}
            )
        try:
            report = runner.run(run_config, out_root=Path(out), deps=deps, resume=resume)
        except runner.RunIncomplete as exc:
            raise CliError(_incomplete_message(exc)) from exc
        if json_output:
            _emit_json(report.model_dump(mode="json"))
            return 0
        if report.env_info.get("synthetic") is True:
            typer.echo(f"{DEMO_LABEL}: scripted agent, not a model result")
        typer.echo(
            f"run {report.run_id}: {len(report.tasks)} tasks, {run_config.n_trials} trials "
            f"-> {Path(out) / report.run_id}"
        )
        for k, value in _pass_summary(report).items():
            typer.echo(f"  pass^{k} {fixed(value, 3)}")
        return 0

    _guard(json_output, body)


@app.command("report")
def report_cmd(
    run_dir: Annotated[str, typer.Argument(help="Path to runs/<run_id>/.")],
    fmt: Annotated[
        str,
        typer.Option("--format", help="md, json or html."),
    ] = "md",
    out: Annotated[
        str | None,
        typer.Option("--out", help="Write the rendered report to this file instead of stdout."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Print the report JSON, whatever --format says."),
    ] = False,
) -> None:
    """Render a run report as Markdown, JSON or HTML."""

    def body() -> int:
        formats: dict[str, Literal["md", "json", "html"]] = {
            "md": "md",
            "json": "json",
            "html": "html",
        }
        if fmt not in formats:
            raise CliError(f"--format must be md, json or html, not {fmt!r}")
        report, trajectories = _run_dir_or_report(run_dir)
        if json_output:
            text = report.model_dump_json(indent=2) + "\n"
        else:
            text = render_report(report, formats[fmt], trajectories)
        if out is None:
            typer.echo(text, nl=False)
        else:
            Path(out).write_text(text, encoding="utf-8")
            typer.echo(f"wrote {out}")
        return 0

    _guard(json_output, body)


@app.command("diff")
def diff_cmd(
    a: Annotated[str, typer.Argument(help="Baseline run directory.")],
    b: Annotated[str, typer.Argument(help="Candidate run directory.")],
    k: Annotated[
        int | None,
        typer.Option("--k", help="k to rank tasks by. Defaults to the largest shared k."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Compare two runs: per-task pass^k deltas and failure shifts."""

    def body() -> int:
        report_a, traj_a = _run_dir_or_report(a)
        report_b, traj_b = _run_dir_or_report(b)
        result = diff(report_a, report_b, k, trajectories_a=traj_a, trajectories_b=traj_b)
        if json_output:
            _emit_json(result)
            return 0
        if result["synthetic"]:
            typer.echo(f"{DEMO_LABEL}: at least one run is synthetic")
        for side, report in (("baseline", report_a), ("candidate", report_b)):
            if not report.complete:
                typer.echo(
                    f"PARTIAL: the {side} run {report.run_id} is incomplete "
                    f"({report.stop_reason}); its numbers cover finished episodes only"
                )
        typer.echo(
            f"baseline {result['baseline']['run_id']} -> candidate {result['candidate']['run_id']}"
            f" at k={result['k']}"
        )
        rows = [
            [
                str(k_value),
                f"{fixed(vals['baseline'], 3)}",
                f"{fixed(vals['candidate'], 3)}",
                f"{fixed(vals['delta'], 3, signed=True)}",
            ]
            for k_value, vals in result["pass_hat_k"].items()
        ]
        typer.echo(_table(["k", "baseline", "candidate", "delta"], rows))
        typer.echo("")
        task_rows = [
            [
                row["task_id"],
                str(row["tier"]),
                f"{fixed(row['pass_hat_k_baseline'], 3)}",
                f"{fixed(row['pass_hat_k_candidate'], 3)}",
                f"{fixed(row['delta'], 3, signed=True)}",
            ]
            for row in result["tasks"]
        ]
        typer.echo(_table(["task", "tier", "baseline", "candidate", "delta"], task_rows))
        if result["only_in_baseline"]:
            typer.echo("only in baseline (not compared): " + ", ".join(result["only_in_baseline"]))
        if result["only_in_candidate"]:
            typer.echo(
                "only in candidate (not compared): " + ", ".join(result["only_in_candidate"])
            )
        shift = result["failure_shift"]
        typer.echo("")
        typer.echo(
            "failure totals (baseline -> candidate): "
            + ", ".join(
                f"{name} {vals['baseline']}->{vals['candidate']}"
                for name, vals in shift["totals"].items()
                if vals["baseline"] or vals["candidate"]
            )
        )
        if shift["paired"]:
            typer.echo(f"paired trials: {shift['paired_trials']}")
        return 0

    _guard(json_output, body)


@app.command("gate")
def gate_cmd(
    a: Annotated[str, typer.Argument(help="Baseline run directory.")],
    b: Annotated[str, typer.Argument(help="Candidate run directory.")],
    policy: Annotated[
        str,
        typer.Option("--policy", help="Path to a GatePolicy YAML file."),
    ],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Exit 0 if the candidate passes the gate, 1 if it regresses."""
    result_code = 0

    def body() -> int:
        nonlocal result_code
        gate_policy: GatePolicy = load_policy(Path(policy))
        baseline = load_report(Path(a))
        candidate = load_report(Path(b))
        for side, report in (("baseline", baseline), ("candidate", candidate)):
            if not report.complete:
                raise CliError(
                    f"refusing to gate an incomplete {side} report ({report.run_id}): "
                    f"{report.stop_reason}"
                )
        decision = gate(baseline, candidate, gate_policy)
        synthetic = any(r.env_info.get("synthetic") is True for r in (baseline, candidate))
        if json_output:
            _emit_json(
                {
                    "passed": decision.passed,
                    "reasons": decision.reasons,
                    "delta": decision.delta,
                    "ci_low": decision.ci_low,
                    "ci_high": decision.ci_high,
                    "policy": gate_policy.model_dump(mode="json"),
                    "baseline": baseline.run_id,
                    "candidate": candidate.run_id,
                    "synthetic": synthetic,
                }
            )
        else:
            if synthetic:
                typer.echo(f"{DEMO_LABEL}: at least one run is synthetic")
            typer.echo("PASS" if decision.passed else "FAIL")
            for reason in decision.reasons:
                typer.echo(f"  - {reason}")
        result_code = 0 if decision.passed else 1
        return 0

    _guard(json_output, body)
    if result_code:
        raise typer.Exit(code=result_code)


@app.command("serve-tools")
def serve_tools_cmd(
    env_name: Annotated[str, typer.Argument(help="Environment to serve, e.g. labstock.")] = (
        "labstock"
    ),
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Accepted for consistency. Stdout is the MCP stream."),
    ] = False,
) -> None:
    """Serve an environment's tools over MCP v2 stdio."""

    def body() -> int:
        if env_name != "labstock":
            raise CliError(f"unknown environment {env_name!r}; only labstock is served")
        from passk import mcp_server

        mcp_server.serve_labstock_stdio()
        return 0

    # Stdout carries the MCP protocol, so errors go to stderr only.
    _guard(False, body)


@app.command("validate")
def validate_cmd(
    k: Annotated[int, typer.Option("--k", help="k at which the gate is applied in E2.")] = 3,
    max_drop: Annotated[
        float, typer.Option("--max-drop", help="max_drop of the E2 gate policy.")
    ] = 0.10,
    pairs: Annotated[
        int, typer.Option("--pairs", help="A/A and A/B comparisons in E2.")
    ] = validate.DEFAULT_PAIRS,
    workers: Annotated[
        int, typer.Option("--workers", help="Worker processes for the simulations.")
    ] = 2,
    smoke: Annotated[
        bool, typer.Option("--smoke", help="Quick check: 10 pairs. Same E1, fewer E2 pairs.")
    ] = False,
    out: Annotated[
        str | None,
        typer.Option("--out", help="Also write validate.json and env.json into this directory."),
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Check estimator and gate calibration (experiments E1 and E2)."""

    def body() -> int:
        from passk.envs.base import load_tasks

        tasks = load_tasks(PACKAGED_TASKS)
        policy = GatePolicy(k=k, max_drop=max_drop)
        n_pairs = 10 if smoke else pairs
        result = validate.run_validation(
            validate.make_labstock_simulator(PACKAGED_TASKS, validate.N_TRIALS),
            tasks,
            policy=policy,
            pairs=n_pairs,
            e2_simulate=validate.make_labstock_simulator(PACKAGED_TASKS, validate.E2_TRIALS),
            workers=workers,
        )
        if out is not None:
            out_dir = Path(out)
            out_dir.mkdir(parents=True, exist_ok=True)
            env_record = runner.environment_record()
            env_record["synthetic"] = True
            (out_dir / "validate.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (out_dir / "env.json").write_text(
                json.dumps(env_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if json_output:
            _emit_json(result)
            return 0
        typer.echo(f"{DEMO_LABEL}: {result['note']}")
        e1_rows = [
            [
                f"{fixed(level['p'], 2)}",
                f"{level['max_abs_error']:.4f}",
                f"{fixed(level['max_abs_z'], 2)}",
            ]
            for level in result["e1"]["levels"]
        ]
        typer.echo("E1 estimator, max absolute error of pass^k over k=1..5 (n=20 per task)")
        typer.echo(_table(["p", "max abs error", "max |z|"], e1_rows))
        e2 = result["e2"]
        typer.echo("")
        typer.echo(f"E2 gate: policy k={e2['policy']['k']}, max_drop={e2['policy']['max_drop']}")
        typer.echo(
            f"  false alarm rate  {fixed(e2['aa']['rate'], 3)} "
            f"({e2['aa']['false_alarms']}/{e2['pairs']}, "
            f"95% CI {fixed(e2['aa']['ci95'][0], 3)} to {fixed(e2['aa']['ci95'][1], 3)})"
        )
        typer.echo(
            f"  power at p 0.05 vs 0.15  {fixed(e2['ab']['power'], 3)} "
            f"({e2['ab']['detections']}/{e2['pairs']}, "
            f"95% CI {fixed(e2['ab']['ci95'][0], 3)} to {fixed(e2['ab']['ci95'][1], 3)})"
        )
        return 0

    _guard(json_output, body)


@app.command("site")
def site_cmd(
    results: Annotated[str, typer.Option("--results", help="Directory searched for reports.")] = (
        "results"
    ),
    out: Annotated[str, typer.Option("--out", help="Output directory.")] = "build/site",
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Build the static Pages site from the committed reports (empty-state page if none)."""

    def body() -> int:
        from passk.site.build import build_site, empty_site

        out_dir = Path(out)
        found = sorted(Path(results).rglob("report.json")) if Path(results).is_dir() else []
        if found:
            index = build_site(Path(results), out_dir)
            status = "built"
        else:
            index = empty_site(out_dir)
            status = "empty"
        if json_output:
            _emit_json({"status": status, "index": str(index), "reports": len(found)})
        else:
            typer.echo(f"site {status}: {index} ({len(found)} reports)")
        return 0

    _guard(json_output, body)


@app.command("demo")
def demo_cmd(
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Run a SYNTHETIC scripted demo in under 30 s with no model and no downloads.

    The demo compares an oracle agent with a noisy one and prints the gate verdict.

    The verdict is expected to be FAIL: that is the point of the demo.

    The exit code is 0 whatever the verdict. It is not a check of the gate.

    For a CI exit code, run `passk gate` on two run directories: 0 on pass, 1 on regression.
    """

    def body() -> int:
        policy = GatePolicy(k=3, max_drop=0.10)
        configs = [
            RunConfig(
                run_id="demo-oracle",
                tasks_glob=PACKAGED_TASKS,
                agent=AgentConfig(kind="oracle"),
                n_trials=5,
                base_seed=1,
                transport="inprocess",
            ),
            RunConfig(
                run_id="demo-noisy-p10",
                tasks_glob=PACKAGED_TASKS,
                agent=AgentConfig(kind="noisy_oracle", params={"p_error": 0.10, "seed": 1}),
                n_trials=5,
                base_seed=1,
                transport="inprocess",
            ),
        ]
        with tempfile.TemporaryDirectory(prefix="passk-demo-") as scratch:
            reports = [runner.run(cfg, out_root=Path(scratch)) for cfg in configs]
        decision = gate(reports[0], reports[1], policy)
        if json_output:
            _emit_json(
                {
                    "label": DEMO_LABEL,
                    "runs": [
                        {
                            "run_id": r.run_id,
                            "pass_hat_k": {str(k): v for k, v in _pass_summary(r).items()},
                        }
                        for r in reports
                    ],
                    "gate": {
                        "baseline": reports[0].run_id,
                        "candidate": reports[1].run_id,
                        "passed": decision.passed,
                        "reasons": decision.reasons,
                    },
                }
            )
            return 0
        typer.echo(f"{DEMO_LABEL}: scripted agents only. Not a model result.")
        headers = ["run", *[f"pass^{k}" for k in range(1, 6)]]
        rows = []
        for report in reports:
            summary = _pass_summary(report)
            rows.append(
                [report.run_id]
                + [f"{fixed(summary[k], 3)}" if k in summary else "n/a" for k in range(1, 6)]
            )
        typer.echo(_table(headers, rows))
        typer.echo("")
        verdict = "PASS" if decision.passed else "FAIL"
        typer.echo(f"gate (baseline demo-oracle, candidate demo-noisy-p10): {verdict}")
        for reason in decision.reasons:
            typer.echo(f"  - {reason}")
        typer.echo(
            "exit code is 0 by design, whatever the verdict. For a CI exit code use "
            "`passk gate` on two run directories."
        )
        return 0

    _guard(json_output, body)


def main() -> None:
    """Entry point used by ``python -m passk``."""
    app()


if __name__ == "__main__":
    main()
