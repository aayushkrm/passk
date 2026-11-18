"""Committed results and installed packages carry no user paths, and envinfo works in a wheel."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

import passk
from passk import runner
from passk.runner import portable_glob
from passk.schema import AgentConfig, RunConfig

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"
FORBIDDEN_PREFIXES = ("/Users/", "/home/", "/private/var/", "C:\\Users\\")
PACKAGED_TASKS = str(Path(passk.__file__).parent / "tasks" / "labstock" / "*.yaml")
REQUIRED_ENV_KEYS = {
    "cores_logical",
    "cores_physical",
    "cpu_flags",
    "cpu_model",
    "git_dirty",
    "git_sha",
    "os",
    "packages",
    "python",
    "utc_time",
}


def test_absolute_glob_inside_the_repo_is_stored_relative() -> None:
    assert portable_glob(PACKAGED_TASKS) == "src/passk/tasks/labstock/*.yaml"


def test_relative_glob_is_unchanged() -> None:
    assert portable_glob("src/passk/tasks/labstock/*.yaml") == "src/passk/tasks/labstock/*.yaml"


def test_absolute_glob_outside_every_root_is_reduced_to_a_pattern(tmp_path: Path) -> None:
    stored = portable_glob(str(tmp_path / "private" / "*.yaml"))
    assert stored == "<external>/*.yaml"
    assert str(tmp_path) not in stored


def test_no_root_is_the_filesystem_root_or_home(tmp_path: Path) -> None:
    # Relative to / or to home would keep the user's path; the guard must skip both.
    stored = portable_glob("/" + "Users/nobody/x/*.yaml")
    assert not stored.startswith("/")
    assert "Users/nobody" not in stored


def test_runner_writes_no_absolute_glob_into_report(tmp_path: Path) -> None:
    config = RunConfig(
        run_id="portable-check",
        tasks_glob=PACKAGED_TASKS,
        agent=AgentConfig(kind="oracle"),
        n_trials=1,
        base_seed=1,
        transport="inprocess",
    )
    runner.run(config, out_root=tmp_path)
    report = json.loads((tmp_path / "portable-check" / "report.json").read_text())
    assert report["config"]["tasks_glob"] == "src/passk/tasks/labstock/*.yaml"
    text = (tmp_path / "portable-check" / "report.json").read_text()
    assert not any(prefix in text for prefix in FORBIDDEN_PREFIXES)


def _committed_text_files() -> list[Path]:
    if not RESULTS.is_dir():
        return []
    suffixes = {".json", ".jsonl", ".txt", ".md", ".html", ".csv"}
    return sorted(p for p in RESULTS.rglob("*") if p.is_file() and p.suffix in suffixes)


def test_no_committed_result_contains_a_user_path() -> None:
    offenders: list[str] = []
    for path in _committed_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if any(prefix in text for prefix in FORBIDDEN_PREFIXES):
            offenders.append(path.relative_to(REPO).as_posix())
    assert offenders == [], f"user paths found in: {offenders}"


def test_committed_env_records_have_the_required_keys() -> None:
    for path in RESULTS.rglob("env.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        missing = REQUIRED_ENV_KEYS - set(record)
        assert not missing, f"{path.relative_to(REPO).as_posix()} lacks {sorted(missing)}"


def test_envinfo_record_has_required_keys_outside_any_checkout(tmp_path: Path) -> None:
    from passk import envinfo

    record = envinfo.collect(tmp_path)
    assert REQUIRED_ENV_KEYS <= set(record)
    assert record["git_sha"] is None  # tmp_path is not a git work tree


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv is needed to build a wheel")
def test_built_wheel_carries_envinfo_tasks_and_templates(tmp_path: Path) -> None:
    out = tmp_path / "dist"
    done = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(out)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    (wheel,) = list(out.glob("passk-*.whl"))
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        assert "passk/envinfo.py" in names
        assert "passk/tasks/labstock/t1_01.yaml" in names
        assert "passk/site/templates/index.html.j2" in names
        source = archive.read("passk/envinfo.py")

    # Load the envinfo module from the wheel bytes, as an installed user would have it.
    module_path = tmp_path / "envinfo_from_wheel.py"
    module_path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("envinfo_from_wheel", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    record = module.collect(tmp_path)
    assert REQUIRED_ENV_KEYS <= set(record)
