"""Inline SVG charts for run pages and the site index.

Each chart is a plain ``<svg>`` string with no external references. Colours come from CSS
custom properties and classes defined in ``static/style.css``, so the charts follow light
and dark mode. Labels are escaped here because the strings are inserted into HTML unescaped.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from html import escape
from typing import Literal

PLOT_LEFT = 52
PLOT_RIGHT = 16
PLOT_TOP = 44
PLOT_BOTTOM = 44


@dataclass(frozen=True)
class CurvePoint:
    """One point on a curve against k, with an optional interval.

    Attributes:
        k: The k value on the x axis.
        value: The metric value, between 0 and 1.
        low: Lower bound of the interval, if any.
        high: Upper bound of the interval, if any.
    """

    k: int
    value: float
    low: float | None = None
    high: float | None = None


@dataclass(frozen=True)
class Curve:
    """A labelled series of curve points.

    Attributes:
        label: Legend text.
        css_class: Colour class, one of ``c0`` to ``c5``.
        points: The points, in any order.
        dashed: Draw the line dashed.
    """

    label: str
    css_class: str
    points: tuple[CurvePoint, ...]
    dashed: bool = False


@dataclass(frozen=True)
class HeatCell:
    """One cell of a heatmap.

    Attributes:
        kind: ``pass`` or ``fail`` for trial outcomes, ``missing`` for no record,
            ``rate`` for a shaded success fraction.
        value: Shading for ``rate`` cells, between 0 and 1.
        glyph: Text drawn in the cell (a check mark, a cross, or empty).
        tooltip: Hover text.
    """

    kind: Literal["pass", "fail", "missing", "rate"]
    value: float
    glyph: str
    tooltip: str


@dataclass(frozen=True)
class HeatRow:
    """One row of a heatmap.

    Attributes:
        label: Row label, usually the task id.
        cells: Cells across the row.
        side_text: Text drawn after the row, for example ``3/5``.
    """

    label: str
    cells: tuple[HeatCell, ...]
    side_text: str = ""


def _open(width: int, height: int, title: str) -> str:
    return (
        f'<svg class="chart" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" role="img" aria-label="{escape(title, quote=True)}">'
        f"<title>{escape(title)}</title>"
    )


def _empty(title: str, width: int, height: int, message: str) -> str:
    return (
        _open(width, height, title) + f'<text class="legend" x="{width // 2}" y="{height // 2}" '
        f'text-anchor="middle">{escape(message)}</text></svg>'
    )


def curves_svg(
    curves: Sequence[Curve],
    title: str,
    *,
    width: int = 560,
    height: int = 320,
    ci_label: str = "95% CI",
) -> str:
    """Draw metric-against-k curves with optional interval bars.

    Args:
        curves: Series to draw.
        title: Accessible chart title.
        width: SVG width in user units.
        height: SVG height in user units.
        ci_label: Name of the interval used in hover text.

    Returns:
        An ``<svg>`` element as a string.
    """
    points = [p for curve in curves for p in curve.points]
    if not points:
        return _empty(title, width, height, "no data")
    ks = [p.k for p in points]
    k_min, k_max = min(ks), max(ks)
    span = max(k_max - k_min, 1)
    plot_w = width - PLOT_LEFT - PLOT_RIGHT
    plot_h = height - PLOT_TOP - PLOT_BOTTOM

    def x_of(k: int) -> float:
        return PLOT_LEFT + (k - k_min) / span * plot_w

    def y_of(value: float) -> float:
        clamped = min(max(value, 0.0), 1.0)
        return PLOT_TOP + (1.0 - clamped) * plot_h

    parts: list[str] = [_open(width, height, title)]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = y_of(tick)
        parts.append(
            f'<line class="grid" x1="{PLOT_LEFT}" x2="{width - PLOT_RIGHT}" '
            f'y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="tick" x="{PLOT_LEFT - 8}" y="{y + 4:.1f}" text-anchor="end">'
            f"{tick:.2f}</text>"
        )
    for k in sorted(set(ks)):
        parts.append(
            f'<text class="tick" x="{x_of(k):.1f}" y="{height - PLOT_BOTTOM + 18}" '
            f'text-anchor="middle">{k}</text>'
        )
    parts.append(
        f'<text class="axis-label" x="{PLOT_LEFT + plot_w / 2:.1f}" y="{height - 6}" '
        f'text-anchor="middle">k</text>'
    )

    legend_x = PLOT_LEFT
    for curve in curves:
        dash = " dashed" if curve.dashed else ""
        parts.append(
            f'<line class="series {curve.css_class}{dash}" x1="{legend_x}" '
            f'x2="{legend_x + 18}" y1="18" y2="18"/>'
        )
        parts.append(
            f'<text class="legend" x="{legend_x + 24}" y="22">{escape(curve.label)}</text>'
        )
        legend_x += 36 + 8 * len(curve.label)

    for curve in curves:
        if not curve.points:
            continue
        ordered = sorted(curve.points, key=lambda p: p.k)
        dash = " dashed" if curve.dashed else ""
        coords = " ".join(f"{x_of(p.k):.1f},{y_of(p.value):.1f}" for p in ordered)
        parts.append(f'<polyline class="series {curve.css_class}{dash}" points="{coords}"/>')
        for p in ordered:
            x = x_of(p.k)
            if p.low is not None and p.high is not None:
                parts.append(
                    f'<line class="ci {curve.css_class}" x1="{x:.1f}" x2="{x:.1f}" '
                    f'y1="{y_of(p.high):.1f}" y2="{y_of(p.low):.1f}"/>'
                )
            tip = f"{curve.label} at k={p.k}: {p.value:.3f}"
            if p.low is not None and p.high is not None:
                tip += f" ({ci_label} {p.low:.3f} to {p.high:.3f})"
            parts.append(
                f'<circle class="dot {curve.css_class}" cx="{x:.1f}" cy="{y_of(p.value):.1f}" '
                f'r="4"><title>{escape(tip)}</title></circle>'
            )
    parts.append("</svg>")
    return "".join(parts)


def heatmap_svg(
    rows: Sequence[HeatRow],
    column_labels: Sequence[str],
    title: str,
    *,
    cell: int = 22,
    gap: int = 3,
) -> str:
    """Draw a grid of outcomes or success fractions.

    Args:
        rows: Rows top to bottom.
        column_labels: Labels above the columns; empty to omit them.
        title: Accessible chart title.
        cell: Cell edge length in user units.
        gap: Space between cells.

    Returns:
        An ``<svg>`` element as a string.
    """
    if not rows:
        return _empty(title, 360, 80, "no tasks")
    n_cols = max(len(row.cells) for row in rows)
    label_w = min(max(len(row.label) for row in rows) * 7 + 12, 280)
    side_w = max((len(row.side_text) for row in rows), default=0) * 7 + 12
    head = 22 if column_labels else 8
    width = label_w + n_cols * (cell + gap) + side_w + 8
    height = head + len(rows) * (cell + gap) + 8
    parts: list[str] = [_open(width, height, title)]
    for c_index, text in enumerate(column_labels[:n_cols]):
        x = label_w + c_index * (cell + gap) + cell / 2
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{head - 8}" text-anchor="middle">'
            f"{escape(text)}</text>"
        )
    for r_index, row in enumerate(rows):
        top = head + r_index * (cell + gap)
        parts.append(
            f'<text class="rowlabel" x="{label_w - 6}" y="{top + cell * 0.7:.1f}" '
            f'text-anchor="end">{escape(row.label)}</text>'
        )
        for c_index, item in enumerate(row.cells):
            x = label_w + c_index * (cell + gap)
            style = (
                f' style="fill-opacity:{max(0.12, item.value):.2f}"' if item.kind == "rate" else ""
            )
            parts.append(
                f"<g><title>{escape(item.tooltip)}</title>"
                f'<rect class="cell {item.kind}" x="{x}" y="{top}" width="{cell}" '
                f'height="{cell}" rx="3"{style}/>'
                f'<text class="glyph {item.kind}" x="{x + cell / 2}" y="{top + cell * 0.68:.1f}" '
                f'text-anchor="middle">{escape(item.glyph)}</text></g>'
            )
        if row.side_text:
            parts.append(
                f'<text class="rowlabel" x="{label_w + n_cols * (cell + gap) + 6}" '
                f'y="{top + cell * 0.7:.1f}">{escape(row.side_text)}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def bars_svg(items: Sequence[tuple[str, int]], title: str, *, width: int = 560) -> str:
    """Draw horizontal bars for labelled counts.

    Args:
        items: ``(label, count)`` pairs drawn top to bottom.
        title: Accessible chart title.
        width: SVG width in user units.

    Returns:
        An ``<svg>`` element as a string.
    """
    if not items or all(count == 0 for _, count in items):
        return _empty(title, width, 60, "no failures recorded")
    label_w = 220
    bar_room = width - label_w - 60
    row_h = 26
    top_pad = 8
    height = top_pad + len(items) * row_h + 8
    peak = max(count for _, count in items) or 1
    parts: list[str] = [_open(width, height, title)]
    for index, (label, count) in enumerate(items):
        y = top_pad + index * row_h
        bar_w = bar_room * count / peak
        parts.append(
            f'<text class="rowlabel" x="{label_w - 8}" y="{y + 15}" text-anchor="end">'
            f"{escape(label)}</text>"
        )
        parts.append(
            f'<rect class="bar" x="{label_w}" y="{y + 3}" width="{bar_w:.1f}" height="16" rx="2">'
            f"<title>{escape(label)}: {count}</title></rect>"
        )
        parts.append(
            f'<text class="tick" x="{label_w + bar_w + 6:.1f}" y="{y + 15}">{count}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)
