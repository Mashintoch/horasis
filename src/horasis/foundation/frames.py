"""Frame iteration over video files and cameras (extra: ``horasis[video]``)."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from horasis.foundation.errors import ValidationError, VisualizationUnavailableError


def _default_capture(source: str | int) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise VisualizationUnavailableError(
            "video streaming needs OpenCV: pip install 'horasis[video]'"
        ) from exc
    return cv2.VideoCapture(source)


def iter_frames(
    source: str | int | Path,
    *,
    every_n_frames: int = 1,
    max_frames: int | None = None,
    capture_factory: Callable[[str | int], Any] | None = None,
) -> Iterator[tuple[int, float, Any]]:
    """Yield ``(frame_index, timestamp_seconds, rgb_array)`` for sampled frames.

    The capture is released when the generator finishes or is closed.

    Args:
        source: A video file path, or an ``int`` camera index (webcam).
        every_n_frames: Yield one frame out of every N read (1 = all). Skipped
            frames are still decoded, since video codecs require it.
        max_frames: Stop after yielding this many frames (a webcam never ends
            on its own).
        capture_factory: Builds a ``cv2.VideoCapture``-like object (for tests).

    Raises:
        ValidationError: if the source cannot be opened or `every_n_frames` < 1.
        VisualizationUnavailableError: if OpenCV is not installed.
    """
    if every_n_frames < 1:
        raise ValidationError("every_n_frames must be >= 1")
    target: str | int = source if isinstance(source, int) else str(source)
    capture = (capture_factory or _default_capture)(target)
    if not capture.isOpened():
        capture.release()
        raise ValidationError(f"cannot open video source {source!r}")
    try:
        fps = float(capture.get(5)) or 0.0  # cv2.CAP_PROP_FPS
        started = time.monotonic()
        yielded = 0
        index = -1
        while max_frames is None or yielded < max_frames:
            ok, bgr = capture.read()
            if not ok:
                return
            index += 1
            if index % every_n_frames:
                continue
            if isinstance(source, int) or fps <= 0:
                stamp = time.monotonic() - started
            else:
                stamp = index / fps
            yield index, stamp, bgr[..., ::-1]
            yielded += 1
    finally:
        capture.release()
