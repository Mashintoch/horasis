"""Image and video annotation helpers for Horasis."""

from __future__ import annotations

# pylint: disable=import-outside-toplevel,too-many-arguments,too-many-locals,no-member,mixed-line-endings

import io
from pathlib import Path
from typing import TYPE_CHECKING, Any

from horasis.foundation.schemas import (
    Detection,
    DetectionResult,
    FaceResult,
    Kinesis,
    Physis,
    VideoDetectionResult,
)
from horasis.foundation.sphalma import ValidationError, VisualizationUnavailableError

if TYPE_CHECKING:
    from PIL import Image as PILImageModule

    PILImage = PILImageModule.Image


def _load_pillow() -> Any:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise VisualizationUnavailableError(
            "The 'Pillow' package is required for horasis.viz. "
            "Install it with: pip install 'horasis[viz]'"
        ) from exc
    return Image, ImageDraw, ImageFont


def _open_image(image: str | Path | bytes | Physis) -> PILImage:
    Image, _, _ = _load_pillow()

    if isinstance(image, Physis):
        physis = image
    elif isinstance(image, bytes):
        physis = Physis.from_bytes(image)
    elif isinstance(image, str):
        physis = Physis.from_source(image)
    elif isinstance(image, Path):
        physis = Physis.from_path(image)
    else:
        raise ValidationError(f"Unsupported image input type: {type(image).__name__}")

    if physis.data is not None:
        return Image.open(io.BytesIO(physis.data)).convert("RGB")
    if physis.path is not None:
        return Image.open(physis.path).convert("RGB")
    raise ValidationError("Physis has neither `path` nor `data` set")


def _color_for_label(label: str) -> tuple[int, int, int]:
    """Deterministic, reasonably distinct color per label (no external palette dep)."""
    h = hash(label) & 0xFFFFFF
    r, g, b = (h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF
    # Keep colors away from near-black/near-white so they stay visible on most images.
    return (r % 200 + 40, g % 200 + 40, b % 200 + 40)


def _draw_boxes(
    image: PILImage,
    detections: list[Detection],
    *,
    show_labels: bool,
    show_confidence: bool,
    box_color: tuple[int, int, int] | None,
    text_color: tuple[int, int, int],
    line_width: int,
    font_size: int,
) -> PILImage:
    _, ImageDraw, ImageFont = _load_pillow()

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        color = box_color or _color_for_label(det.label)
        box = det.box
        draw.rectangle(
            [(box.x_min, box.y_min), (box.x_max, box.y_max)],
            outline=color,
            width=line_width,
        )

        if not (show_labels or show_confidence):
            continue

        parts = []
        if show_labels:
            parts.append(det.label)
        if show_confidence:
            parts.append(f"{det.confidence.value:.2f}")
        caption = " ".join(parts)

        text_bbox = draw.textbbox((0, 0), caption, font=font)
        text_w, text_h = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        label_y = max(box.y_min - text_h - 4, 0)
        draw.rectangle(
            [(box.x_min, label_y), (box.x_min + text_w + 6, label_y + text_h + 4)],
            fill=color,
        )
        draw.text((box.x_min + 3, label_y + 2), caption, fill=text_color, font=font)

    return annotated


def _resolve_image_output_path(output_path: str | Path | None) -> tuple[Path, dict[str, Any]]:
    """Normalize image output targets.

    If no path is given, save to the current working directory as
    `annotated.png`.
    If callers pass an existing directory, write `annotated.png` inside it.
    If callers pass a path without an extension, force PNG output so Pillow
    does not need to infer a format from the filename.
    """
    path = Path.cwd() / "annotated.png" if output_path is None else Path(output_path)
    save_kwargs: dict[str, Any] = {}

    if path.exists() and path.is_dir():
        path = path / "annotated.png"

    if path.suffix == "":
        save_kwargs["format"] = "PNG"

    return path, save_kwargs


def render_detections(
    image: str | Path | bytes | Physis,
    result: DetectionResult,
    *,
    show_labels: bool = True,
    show_confidence: bool = True,
    box_color: tuple[int, int, int] | None = None,
    text_color: tuple[int, int, int] = (255, 255, 255),
    line_width: int = 3,
    font_size: int = 16,
    output_path: str | Path | None = None,
) -> PILImage:
    """Draw `result.detections` on `image` and return the annotated image.

    Args:
        image: The original input -- a file path, raw bytes, or `Physis`.
        result: A `DetectionResult` produced by `Skopos.detection.detect(...)`.
        show_labels: Draw each detection's class label.
        show_confidence: Draw each detection's confidence score.
        box_color: Fixed RGB color for every box; if omitted, a color is
            derived deterministically per label so classes stay visually
            distinct across calls.
        line_width: Box border thickness in pixels.
        font_size: Caption font size in points.
        output_path: If given, save the annotated image there. If omitted,
            the image is saved to the current working directory as
            `annotated.png`.

    Raises:
        VisualizationUnavailableError: if Pillow is not installed.
    """
    pil_image = _open_image(image)
    annotated = _draw_boxes(
        pil_image,
        result.detections,
        show_labels=show_labels,
        show_confidence=show_confidence,
        box_color=box_color,
        text_color=text_color,
        line_width=line_width,
        font_size=font_size,
    )
    output_path, save_kwargs = _resolve_image_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path, **save_kwargs)
    return annotated


def render_faces(
    image: str | Path | bytes | Physis,
    result: FaceResult,
    *,
    show_labels: bool = True,
    show_confidence: bool = True,
    box_color: tuple[int, int, int] | None = None,
    text_color: tuple[int, int, int] = (255, 255, 255),
    line_width: int = 3,
    font_size: int = 16,
    output_path: str | Path | None = None,
) -> PILImage:
    """Draw `result.faces` on `image` and return the annotated image.

    Same parameters and behavior as `render_detections`, applied to a
    `FaceResult` instead of a `DetectionResult`.
    """
    pil_image = _open_image(image)
    annotated = _draw_boxes(
        pil_image,
        result.faces,
        show_labels=show_labels,
        show_confidence=show_confidence,
        box_color=box_color,
        text_color=text_color,
        line_width=line_width,
        font_size=font_size,
    )
    output_path, save_kwargs = _resolve_image_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(output_path, **save_kwargs)
    return annotated


def _load_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise VisualizationUnavailableError(
            "The 'opencv-python-headless' package is required for video annotation. "
            "Install it with: pip install 'horasis[viz]'"
        ) from exc
    return cv2


class _Cv2VideoWriter:
    """Fallback writer using `cv2.VideoWriter` with the `mp4v` codec.

    Used only when `imageio-ffmpeg` isn't installed. `mp4v` is far less
    space-efficient than H.264 (the PyPI OpenCV wheels omit H.264 encoding
    support for licensing reasons) -- expect noticeably larger output files
    than the source video when this fallback is in use.
    """

    def __init__(self, cv2: Any, path: str, fps: float, width: int, height: int) -> None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    def write(self, frame_bgr: Any) -> None:
        self._writer.write(frame_bgr)

    def release(self) -> None:
        self._writer.release()


class _FfmpegVideoWriter:
    """Writer using `imageio` + the bundled static FFmpeg binary from
    `imageio-ffmpeg`, encoding with H.264. Produces file sizes far closer
    to (often smaller than) typical camera/CCTV source footage than the
    `mp4v` fallback -- H.264 is what most cameras already encode with.
    """

    def __init__(self, cv2: Any, path: str, fps: float) -> None:
        import imageio

        self._cv2 = cv2
        self._writer = imageio.get_writer(
            path, fps=fps, codec="libx264", ffmpeg_params=["-crf", "23", "-preset", "veryfast"]
        )

    def write(self, frame_bgr: Any) -> None:
        rgb = self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)
        self._writer.append_data(rgb)

    def release(self) -> None:
        self._writer.close()


def _open_video_writer(cv2: Any, path: str, fps: float, width: int, height: int) -> Any:
    """Open the best available video writer for `path`."""
    try:
        import imageio  # noqa: F401
        import imageio_ffmpeg  # noqa: F401
    except ImportError:
        return _Cv2VideoWriter(cv2, path, fps, width, height)
    return _FfmpegVideoWriter(cv2, path, fps)


def _video_capture_source(video: str | Path | bytes | Kinesis) -> tuple[Any, str | None]:
    """Normalize `video` into something `cv2.VideoCapture` can open.

    Returns (source, temp_file_path_to_clean_up_or_None). Mirrors
    `UltralyticsAdapter._video_source_for` -- duplicated rather than
    imported, since `horasis.viz` intentionally depends only on the
    foundation layer, not the backend layer.
    """
    if isinstance(video, Kinesis):
        kinesis = video
    elif isinstance(video, bytes):
        kinesis = Kinesis.from_bytes(video)
    elif isinstance(video, (str, Path)):
        source = str(video)
        kinesis = (
            Kinesis.from_url(source)
            if source.startswith(("http://", "https://"))
            else Kinesis.from_path(source)
        )
    else:
        raise ValidationError(f"Unsupported video input type: {type(video).__name__}")

    if kinesis.url is not None:
        return kinesis.url, None
    if kinesis.path is not None:
        return str(kinesis.path), None
    if kinesis.data is not None:
        import tempfile

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(kinesis.data)
        temp.close()
        return temp.name, temp.name
    raise ValidationError("Kinesis has neither `path`, `url`, nor `data` set")


def render_video(
    video: str | Path | bytes | Kinesis,
    result: VideoDetectionResult,
    *,
    output_path: str | Path,
    show_labels: bool = True,
    show_confidence: bool = True,
    box_color: tuple[int, int, int] | None = None,
    text_color: tuple[int, int, int] = (255, 255, 255),
    line_width: int = 3,
    font_size: int = 16,
) -> Path:
    """Draw `result.frames` back onto `video` and write an annotated video file.

    Re-reads `video` frame by frame (a separate pass from whatever produced
    `result` via `ObjectDetector.detect_video`), draws boxes only on the
    frames present in `result.frames` (others pass through unannotated,
    e.g. if `sample_every_n_frames` skipped them), and writes the result to
    `output_path`.

    Args:
        video: The same source originally passed to `detect_video` --
            a local path, URL, raw bytes, or `Kinesis`.
        result: The `VideoDetectionResult` from `detect_video`.
        output_path: Where to write the annotated video (e.g. `"out.mp4"`).
        show_labels: Draw each detection's class label.
        show_confidence: Draw each detection's confidence score.
        box_color: Fixed RGB color for every box; if omitted, derived per
            label, same as `render_detections`.
        line_width: Box border thickness in pixels.
        font_size: Caption font size in points.

    Returns:
        `Path(output_path)`, for convenience.

    Raises:
        VisualizationUnavailableError: if Pillow or opencv-python-headless
            is not installed.
        ValidationError: if `video` cannot be opened.
    """
    import os

    cv2 = _load_cv2()
    from PIL import Image as PILImage

    frames_by_index = {f.frame_index: f.detections for f in result.frames}

    source, temp_file = _video_capture_source(video)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if temp_file is not None:
            os.unlink(temp_file)
        raise ValidationError(f"Could not open video: {video!r}")

    fps = result.fps or cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _open_video_writer(cv2, str(output_path), fps, width, height)

    try:
        idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            detections = frames_by_index.get(idx)
            if detections:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                pil_frame = PILImage.fromarray(frame_rgb)
                annotated_pil = _draw_boxes(
                    pil_frame,
                    detections,
                    show_labels=show_labels,
                    show_confidence=show_confidence,
                    box_color=box_color,
                    text_color=text_color,
                    line_width=line_width,
                    font_size=font_size,
                )
                import numpy as np

                frame_bgr = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

            writer.write(frame_bgr)
            idx += 1
    finally:
        cap.release()
        writer.release()
        if temp_file is not None:
            os.unlink(temp_file)

    return output_path
