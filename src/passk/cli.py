"""Command-line interface: ``passk``.

Exit codes: 0 = pass, 1 = check failed or regression, 2 = usage or runtime error.
Every command accepts ``--json``.
"""

from typing import Annotated

import typer

app = typer.Typer(
    name="passk",
    help="pass^k for tool-using agents: an offline, CI-gated reliability harness.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command("run")
def run_cmd(
    config: Annotated[
        str,
        typer.Option("--config", "-c", help="Path to a RunConfig YAML file."),
    ],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Run tasks x trials for one configuration and write runs/<run_id>/."""
    raise NotImplementedError("passk run: scheduled for work package P-D")


@app.command("report")
def report_cmd(
    run_dir: Annotated[str, typer.Argument(help="Path to runs/<run_id>/.")],
    fmt: Annotated[
        str,
        typer.Option("--format", help="md, json or html."),
    ] = "md",
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Render a run report as Markdown, JSON or HTML."""
    raise NotImplementedError("passk report: scheduled for work package P-E")


@app.command("diff")
def diff_cmd(
    a: Annotated[str, typer.Argument(help="Baseline run directory.")],
    b: Annotated[str, typer.Argument(help="Candidate run directory.")],
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Compare two runs: per-task pass^k deltas and failure shifts."""
    raise NotImplementedError("passk diff: scheduled for work package P-E")


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
    raise NotImplementedError("passk gate: scheduled for work package P-E")


@app.command("serve-tools")
def serve_tools_cmd(
    env_name: Annotated[str, typer.Argument(help="Environment to serve, e.g. labstock.")] = (
        "labstock"
    ),
) -> None:
    """Serve an environment's tools over MCP v2 stdio."""
    raise NotImplementedError("passk serve-tools: scheduled for work package P-C")


@app.command("validate")
def validate_cmd(
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Check estimator and gate calibration (experiments E1 and E2)."""
    raise NotImplementedError("passk validate: scheduled for work package P-E")


@app.command("demo")
def demo_cmd(
    json_output: Annotated[bool, typer.Option("--json", help="Print JSON.")] = False,
) -> None:
    """Run a SYNTHETIC scripted demo in under 30 s with no model and no downloads."""
    raise NotImplementedError("passk demo: scheduled for work package P-F")


def main() -> None:
    """Entry point used by ``python -m passk``."""
    app()


if __name__ == "__main__":
    main()
