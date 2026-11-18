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
) -> None:
    """Run tasks x trials for one configuration and write runs/<run_id>/."""

    def body() -> int:
        run_config = _load_run_config(Path(config))
        report = runner.run(run_config)
        if json_output:
            _emit_json(report.model_dump(mode="json"))
            return 0
        typer.echo(f"run {report.run_id}: {len(report.tasks)} tasks, {run_config.n_trials} trials")
        for k, value in _pass_summary(report).items():
            typer.echo(f"  pass^{k} {value:.3f}")
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
        typer.echo(
            f"baseline {result['baseline']['run_id']} -> candidate {result['candidate']['run_id']}"
            f" at k={result['k']}"
        )
        rows = [
            [
                str(k_value),
                f"{vals['baseline']:.3f}",
                f"{vals['candidate']:.3f}",
                f"{vals['delta']:+.3f}",
            ]
            for k_value, vals in result["pass_hat_k"].items()
        ]
        typer.echo(_table(["k", "baseline", "candidate", "delta"], rows))
        typer.echo("")
        task_rows = [
            [
                row["task_id"],
                str(row["tier"]),
                f"{row['pass_hat_k_baseline']:.3f}",
                f"{row['pass_hat_k_candidate']:.3f}",
                f"{row['delta']:+.3f}",
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
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Check estimator and gate calibration (experiments E1 and E2)."""

    def body() -> int:
        from passk.envs.base import load_tasks

        tasks = load_tasks(PACKAGED_TASKS)
        policy = GatePolicy(k=k, max_drop=max_drop)
        with tempfile.TemporaryDirectory(prefix="passk-validate-") as scratch:
            simulate = validate.make_labstock_simulator(PACKAGED_TASKS, Path(scratch))
            result = validate.run_validation(simulate, tasks, policy=policy, pairs=pairs)
        if json_output:
            _emit_json(result)
            return 0
        typer.echo(f"{DEMO_LABEL}: {result['note']}")
        e1_rows = [
            [
                f"{level['p']:.2f}",
                f"{level['max_abs_error']:.4f}",
            ]
            for level in result["e1"]["levels"]
        ]
        typer.echo("E1 estimator, max absolute error of pass^k over k=1..5")
        typer.echo(_table(["p", "max abs error"], e1_rows))
        e2 = result["e2"]
        typer.echo("")
        typer.echo(f"E2 gate: policy k={e2['policy']['k']}, max_drop={e2['policy']['max_drop']}")
        typer.echo(
            f"  false alarm rate  {e2['aa']['rate']:.3f} "
            f"({e2['aa']['false_alarms']}/{e2['pairs']}, "
            f"95% CI {e2['aa']['ci95'][0]:.3f} to {e2['aa']['ci95'][1]:.3f})"
        )
        typer.echo(
            f"  power at p 0.05 vs 0.15  {e2['ab']['power']:.3f} "
            f"({e2['ab']['detections']}/{e2['pairs']}, "
            f"95% CI {e2['ab']['ci95'][0]:.3f} to {e2['ab']['ci95'][1]:.3f})"
        )
        return 0

    _guard(json_output, body)


@app.command("demo")
def demo_cmd(
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Run a SYNTHETIC scripted demo in under 30 s with no model and no downloads."""

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
                + [f"{summary[k]:.3f}" if k in summary else "n/a" for k in range(1, 6)]
            )
        typer.echo(_table(headers, rows))
        typer.echo("")
        verdict = "PASS" if decision.passed else "FAIL"
        typer.echo(f"gate (baseline demo-oracle, candidate demo-noisy-p10): {verdict}")
        for reason in decision.reasons:
            typer.echo(f"  - {reason}")
        return 0

    _guard(json_output, body)


def main() -> None:
    """Entry point used by ``python -m passk``."""
    app()


if __name__ == "__main__":
    main()
