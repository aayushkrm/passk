"""Build the static Pages site and render single-run reports.

The site holds a summary table, pass^k-vs-k curves (inline SVG), a task x trial heatmap,
a failure bar chart, a collapsible trajectory viewer and ``llms.txt``. Templates are
Jinja2 files in this package.
"""

from pathlib import Path
from typing import Literal

from passk.schema import RunReport


def render_report(report: RunReport, fmt: Literal["md", "json", "html"]) -> str:
    """Render one run report.

    Args:
        report: Run report to render.
        fmt: Output format.

    Returns:
        The rendered document as text.
    """
    raise NotImplementedError("render_report: scheduled for work package P-E")


def build_site(results_dir: Path, out_dir: Path) -> Path:
    """Build the static site from every committed report under ``results_dir``.

    Args:
        results_dir: Directory holding ``**/report.json`` files.
        out_dir: Output directory, created if missing.

    Returns:
        The path of the written ``index.html``.
    """
    raise NotImplementedError("build_site: scheduled for work package P-E")
