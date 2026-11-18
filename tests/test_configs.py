"""Every committed config loads with the code that reads it."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from passk.gate import load_policy
from passk.schema import RunConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
RUN_CONFIGS = sorted(p for p in CONFIGS.glob("*.yaml") if p.name != "gate.yaml")


@pytest.mark.parametrize("path", RUN_CONFIGS, ids=lambda p: p.name)
def test_run_config_loads(path: Path) -> None:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = RunConfig.model_validate(data)
    assert config.run_id and config.n_trials >= 1
    assert "/Users/" not in path.read_text(encoding="utf-8")  # no user paths in committed files


def test_gate_policy_loads() -> None:
    policy = load_policy(CONFIGS / "gate.yaml")
    assert policy.k == 3 and policy.max_drop == 0.10
