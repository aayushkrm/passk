#!/usr/bin/env python3
"""Print a JSON record of the machine, software and git state behind a result.

Recorded: git SHA and dirty flag, UTC time, CPU model and selected SIMD flags, physical
and logical cores, RAM, OS, Python version and installed package versions.

Never recorded: hostname, username, home or working-directory paths, and
``sys.executable`` (it contains the user's path).

This file is copied into each repo on purpose (no cross-repo dependency). Only the
standard library is used.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

DEFAULT_PACKAGES: tuple[str, ...] = (
    "passk",
    "pydantic",
    "typer",
    "pyyaml",
    "numpy",
    "jinja2",
    "mcp",
    "llama-cpp-python",
)

SELECTED_FLAGS: tuple[str, ...] = ("avx", "avx2", "fma", "f16c", "avx512f", "avx512_vnni")

_MAC_FLAG_ALIASES: dict[str, str] = {
    "avx1.0": "avx",
    "avx512vnni": "avx512_vnni",
}


def run_text(args: list[str], cwd: Path | None = None) -> str | None:
    """Run a command and return its stripped stdout, or None if it cannot run.

    Args:
        args: Command and arguments.
        cwd: Working directory for the command.

    Returns:
        Stripped stdout, or None when the command is missing or fails.
    """
    try:
        proc = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip()


def normalize_flag(token: str) -> str:
    """Map a CPU feature token from sysctl or /proc/cpuinfo to a lowercase flag name.

    Args:
        token: Raw feature token, for example ``AVX1.0`` or ``avx2``.

    Returns:
        Normalised name, for example ``avx``.
    """
    lowered = token.strip().lower()
    return _MAC_FLAG_ALIASES.get(lowered, lowered.replace(".", "_"))


def selected_flags(tokens: set[str]) -> dict[str, bool]:
    """Return a presence map for the SIMD flags that matter for CPU inference.

    Args:
        tokens: Normalised flag names reported by the CPU.

    Returns:
        ``{flag: present}`` for every name in ``SELECTED_FLAGS``.
    """
    return {flag: flag in tokens for flag in SELECTED_FLAGS}


def parse_proc_cpuinfo(text: str) -> dict[str, Any]:
    """Extract CPU model, physical core count and flags from ``/proc/cpuinfo`` text.

    Args:
        text: Contents of ``/proc/cpuinfo``.

    Returns:
        Dict with ``cpu_model`` (str or None), ``cores_physical`` (int or None) and
        ``flags`` (set of normalised names).
    """
    model: str | None = None
    cores: int | None = None
    flags: set[str] = set()
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if key == "model name" and model is None:
            model = value
        elif key == "cpu cores" and cores is None and value.isdigit():
            cores = int(value)
        elif key == "flags" and not flags:
            flags = {normalize_flag(tok) for tok in value.split()}
    return {"cpu_model": model, "cores_physical": cores, "flags": flags}


def parse_meminfo_total_bytes(text: str) -> int | None:
    """Return ``MemTotal`` from ``/proc/meminfo`` text in bytes.

    Args:
        text: Contents of ``/proc/meminfo``.

    Returns:
        Total memory in bytes, or None if the line is absent.
    """
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) * 1024
    return None


def parse_sysctl_flags(*feature_lists: str) -> set[str]:
    """Normalise macOS ``machdep.cpu.features`` style lists into a flag set.

    Args:
        *feature_lists: Raw outputs of ``sysctl -n`` feature keys.

    Returns:
        Set of normalised flag names.
    """
    tokens: set[str] = set()
    for text in feature_lists:
        tokens.update(normalize_flag(tok) for tok in text.split())
    return tokens


def read_text(path: str) -> str | None:
    """Read a text file, returning None if it does not exist or cannot be read.

    Args:
        path: File path to read.

    Returns:
        File contents, or None on failure.
    """
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def sysctl_int(key: str) -> int | None:
    """Return an integer sysctl value (macOS), or None.

    Args:
        key: sysctl key, for example ``hw.memsize``.

    Returns:
        The integer value, or None if unavailable.
    """
    out = run_text(["sysctl", "-n", key])
    if out is not None and out.isdigit():
        return int(out)
    return None


def cpu_and_memory() -> dict[str, Any]:
    """Collect CPU model, flags, cores and RAM for the current OS.

    Returns:
        Dict with ``cpu_model``, ``cpu_flags``, ``cores_physical``, ``cores_logical``
        and ``ram_bytes``. Unknown values are None.
    """
    logical = _logical_cores()
    if sys.platform == "darwin":
        model = run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
        features = run_text(["sysctl", "-n", "machdep.cpu.features"]) or ""
        leaf7 = run_text(["sysctl", "-n", "machdep.cpu.leaf7_features"]) or ""
        tokens = parse_sysctl_flags(features, leaf7)
        return {
            "cpu_model": model,
            "cpu_flags": selected_flags(tokens),
            "cores_physical": sysctl_int("hw.physicalcpu"),
            "cores_logical": sysctl_int("hw.logicalcpu") or logical,
            "ram_bytes": sysctl_int("hw.memsize"),
        }
    cpuinfo = read_text("/proc/cpuinfo")
    parsed = (
        parse_proc_cpuinfo(cpuinfo)
        if cpuinfo
        else {"cpu_model": None, "cores_physical": None, "flags": set()}
    )
    meminfo = read_text("/proc/meminfo")
    return {
        "cpu_model": parsed["cpu_model"],
        "cpu_flags": selected_flags(parsed["flags"]),
        "cores_physical": parsed["cores_physical"],
        "cores_logical": logical,
        "ram_bytes": parse_meminfo_total_bytes(meminfo) if meminfo else None,
    }


def _logical_cores() -> int | None:
    """Return the number of logical CPUs visible to this process, or None."""
    return os.cpu_count()


def git_info(repo: Path) -> dict[str, Any]:
    """Return the git SHA and dirty flag for ``repo``, or None values outside git.

    Args:
        repo: Directory inside the repository.

    Returns:
        Dict with ``git_sha`` and ``git_dirty``.
    """
    sha = run_text(["git", "rev-parse", "HEAD"], cwd=repo)
    status = run_text(["git", "status", "--porcelain"], cwd=repo)
    return {
        "git_sha": sha,
        "git_dirty": None if status is None else bool(status),
    }


def package_versions(names: tuple[str, ...]) -> dict[str, str | None]:
    """Return installed versions of ``names``; None if a package is not installed.

    Args:
        names: Distribution names.

    Returns:
        Map from name to version string.
    """
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def os_description() -> dict[str, Any]:
    """Describe the operating system without host identifiers.

    Returns:
        Dict with ``os``, ``os_release``, ``arch`` and, on macOS, ``macos_version``.
    """
    info: dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
    }
    if sys.platform == "darwin":
        info["macos_version"] = platform.mac_ver()[0] or None
    return info


def collect(repo: Path, packages: tuple[str, ...] = DEFAULT_PACKAGES) -> dict[str, Any]:
    """Collect the full environment record.

    Args:
        repo: Directory inside the repository (used for the git SHA).
        packages: Distribution names whose versions to record.

    Returns:
        A JSON-serialisable dict. Contains no hostname, username or paths.
    """
    record: dict[str, Any] = {
        "utc_time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    record.update(git_info(repo))
    record.update(cpu_and_memory())
    record["os"] = os_description()
    record["python"] = {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
    }
    record["packages"] = package_versions(packages)
    return record


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point.

    Args:
        argv: Arguments, defaulting to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="Write JSON to this file instead of stdout.")
    parser.add_argument(
        "--packages",
        help="Comma-separated distribution names to record (default: a built-in list).",
    )
    args = parser.parse_args(argv)
    packages = (
        tuple(p.strip() for p in args.packages.split(",") if p.strip())
        if args.packages
        else DEFAULT_PACKAGES
    )
    repo = Path(__file__).resolve().parent.parent
    text = json.dumps(collect(repo, packages), indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
