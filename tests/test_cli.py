"""CLI smoke tests: help text works for every command."""

import pytest
from typer.testing import CliRunner

from passk.cli import app

runner = CliRunner()

COMMANDS = ["run", "report", "diff", "gate", "serve-tools", "validate", "demo"]


def test_top_level_help_works() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for name in COMMANDS:
        assert name in result.output


@pytest.mark.parametrize("command", COMMANDS)
def test_command_help_works(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0, result.output
