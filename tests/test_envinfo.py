"""Tests for scripts/envinfo.py. No network; no hostname or username in output."""

from __future__ import annotations

import getpass
import importlib.util
import json
import socket
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[1]


def load_envinfo() -> ModuleType:
    """Import scripts/envinfo.py by path (scripts is not a package)."""
    spec = importlib.util.spec_from_file_location("envinfo", REPO / "scripts" / "envinfo.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


envinfo = load_envinfo()

PROC_CPUINFO = """processor\t: 0
vendor_id\t: GenuineIntel
model name\t: Intel(R) Core(TM) i7-4770HQ CPU @ 2.20GHz
cpu cores\t: 4
flags\t\t: fpu vme de avx avx2 fma f16c sse4_2
processor\t: 1
model name\t: Intel(R) Core(TM) i7-4770HQ CPU @ 2.20GHz
cpu cores\t: 4
flags\t\t: fpu vme de avx avx2 fma f16c sse4_2
"""

PROC_MEMINFO = "MemTotal:       16318004 kB\nMemFree:         1234 kB\n"


def test_parse_proc_cpuinfo_extracts_model_cores_and_flags() -> None:
    parsed = envinfo.parse_proc_cpuinfo(PROC_CPUINFO)
    assert parsed["cpu_model"] == "Intel(R) Core(TM) i7-4770HQ CPU @ 2.20GHz"
    assert parsed["cores_physical"] == 4
    assert {"avx", "avx2", "fma", "f16c"} <= parsed["flags"]


def test_selected_flags_marks_absent_avx512_false() -> None:
    flags = envinfo.selected_flags({"avx", "avx2", "fma", "f16c"})
    assert flags["avx2"] is True
    assert flags["avx512f"] is False
    assert set(flags) == set(envinfo.SELECTED_FLAGS)


def test_parse_macos_feature_tokens_are_normalised() -> None:
    tokens = envinfo.parse_sysctl_flags("FPU VME AVX1.0 FMA F16C SSE4.2", "AVX2 BMI1")
    assert {"avx", "fma", "f16c", "sse4_2", "avx2", "bmi1"} <= tokens
    assert envinfo.selected_flags(tokens)["avx2"] is True


def test_parse_meminfo_total_bytes() -> None:
    assert envinfo.parse_meminfo_total_bytes(PROC_MEMINFO) == 16318004 * 1024
    assert envinfo.parse_meminfo_total_bytes("MemFree: 1 kB\n") is None


def test_run_text_returns_none_for_missing_command() -> None:
    assert envinfo.run_text(["passk-definitely-not-a-command-xyz"]) is None


def test_package_versions_handles_missing_package() -> None:
    versions = envinfo.package_versions(("pytest", "passk-not-installed-xyz"))
    assert isinstance(versions["pytest"], str)
    assert versions["passk-not-installed-xyz"] is None


def test_collect_has_required_keys_and_utc_timestamp() -> None:
    record = envinfo.collect(REPO, packages=("pytest",))
    for key in (
        "utc_time",
        "git_sha",
        "git_dirty",
        "cpu_model",
        "cpu_flags",
        "cores_logical",
        "ram_bytes",
        "os",
        "python",
        "packages",
    ):
        assert key in record, key
    assert record["utc_time"].endswith("+00:00")
    assert record["git_sha"] is None or len(record["git_sha"]) == 40
    assert set(record["cpu_flags"]) == set(envinfo.SELECTED_FLAGS)
    assert record["packages"]["pytest"] is not None


@pytest.mark.parametrize("forbidden_key", ["hostname", "user", "username", "home", "cwd"])
def test_collect_never_emits_identifying_keys(forbidden_key: str) -> None:
    record = envinfo.collect(REPO, packages=("pytest",))
    assert forbidden_key not in record
    assert forbidden_key not in record.get("os", {})


def test_collect_output_has_no_hostname_or_username() -> None:
    text = json.dumps(envinfo.collect(REPO, packages=("pytest",)))
    hostname = socket.gethostname()
    if len(hostname) >= 4:
        assert hostname not in text
    try:
        user = getpass.getuser()
    except (KeyError, OSError):
        user = ""
    if len(user) >= 4:
        assert user not in text


def test_main_writes_json_file(tmp_path: Path) -> None:
    out = tmp_path / "env.json"
    assert envinfo.main(["--out", str(out), "--packages", "pytest"]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert list(data["packages"]) == ["pytest"]
    assert data["packages"]["pytest"] is not None
