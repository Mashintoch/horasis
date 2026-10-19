"""Small helpers shared by the concrete `Backend` adapters."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TypeVar

from horasis.foundation.schemas import BoundingBox, Confidence, Point

T = TypeVar("T")


def clamp01(value: float) -> float:
    """Clamp `value` into the valid confidence range ``[0, 1]``."""
    return max(0.0, min(1.0, float(value)))


def to_confidence(value: float) -> Confidence:
    """Build a `Confidence` from a possibly slightly out-of-range engine score."""
    return Confidence(value=clamp01(value))


def polygon_to_box(points: Iterable[Sequence[float]]) -> BoundingBox:
    """Axis-aligned box that encloses a polygon."""
    pts = [(float(p[0]), float(p[1])) for p in points]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return BoundingBox(x_min=min(xs), y_min=min(ys), x_max=max(xs), y_max=max(ys))


def box_to_polygon(box: BoundingBox) -> list[Point]:
    """Four corners of `box`, clockwise from top-left."""
    return [
        (box.x_min, box.y_min),
        (box.x_max, box.y_min),
        (box.x_max, box.y_max),
        (box.x_min, box.y_max),
    ]


def order_into_lines(
    items: Sequence[T], box_of: Callable[[T], BoundingBox]
) -> list[tuple[int, T]]:
    """Sort items into reading order and number their text lines.

    Items whose vertical centres are within half a line height of the line
    being built share that line; each line is ordered left to right.

    Returns:
        ``(line_index, item)`` pairs in reading order.
    """
    if not items:
        return []

    def cy(it: T) -> float:
        b = box_of(it)
        return (b.y_min + b.y_max) / 2

    lines: list[list[T]] = []
    line_cy: list[float] = []
    line_h: list[float] = []
    for item in sorted(items, key=cy):
        b = box_of(item)
        height = max(b.y_max - b.y_min, 1e-6)
        if lines and abs(cy(item) - line_cy[-1]) <= 0.5 * max(height, line_h[-1]):
            lines[-1].append(item)
            n = len(lines[-1])
            line_cy[-1] += (cy(item) - line_cy[-1]) / n
            line_h[-1] = max(line_h[-1], height)
        else:
            lines.append([item])
            line_cy.append(cy(item))
            line_h.append(height)

    ordered: list[tuple[int, T]] = []
    for idx, line in enumerate(lines):
        for item in sorted(line, key=lambda it: box_of(it).x_min):
            ordered.append((idx, item))
    return ordered
