"""Image and video annotation helpers for Horasis."""

from __future__ import annotations

# pylint: disable=import-outside-toplevel,too-many-arguments,too-many-locals,no-member,mixed-line-endings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from horasis.foundation.imaging import load_rgb, to_physis
from horasis.foundation.schemas import (
    ClassificationResult,
    Detection,
    DetectionResult,
    FaceResult,
    Kinesis,
    ModerationResult,
    OcrResult,
    Physis,
    PoseResult,
    SegmentationResult,
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


def _open_image(image: Any) -> PILImage:
    """Load any supported image input as upright RGB via the shared loader."""
    _load_pillow()
    return load_rgb(to_physis(image))


def _color_for_label(label: str) -> tuple[int, int, int]:
    """Deterministic, reasonably distinct color per label (no external palette dep)."""
    h = hash(label) & 0xFFFFFF
    r, g, b = (h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF
    # Keep colors away from near-black/near-white so they stay visible on most images.
    return (r % 200 + 40, g % 200 + 40, b % 200 + 40)


def _draw_boxes(
    image: PILImage,
    detections: Sequence[Detection],
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

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")  # noqa: SIM115
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


def render_ocr(
    image: Any,
    result: OcrResult,
    *,
    box_color: tuple[int, int, int] | None = None,
    text_color: tuple[int, int, int] = (255, 255, 255),
    line_width: int = 2,
    font_size: int = 14,
    show_confidence: bool = False,
    **_ignored: Any,
) -> PILImage:
    """Draw each OCR glyph's polygon (or box) with its text; nothing is saved."""
    _, ImageDraw, ImageFont = _load_pillow()
    annotated = _open_image(image).copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    for g in result.glyphs:
        color = box_color or (230, 40, 40)
        if g.polygon:
            draw.polygon([tuple(pt) for pt in g.polygon], outline=color, width=line_width)
            x, y = min(p[0] for p in g.polygon), min(p[1] for p in g.polygon)
        elif g.box is not None:
            draw.rectangle(
                [(g.box.x_min, g.box.y_min), (g.box.x_max, g.box.y_max)],
                outline=color,
                width=line_width,
            )
            x, y = g.box.x_min, g.box.y_min
        else:
            continue
        caption = f"{g.text} {g.confidence.value:.2f}" if show_confidence else g.text
        draw.text((x, max(y - font_size - 2, 0)), caption, fill=color, font=font)
    return annotated


def render_caption(
    image: Any,
    lines: Sequence[str],
    *,
    text_color: tuple[int, int, int] = (255, 255, 255),
    font_size: int = 16,
    **_ignored: Any,
) -> PILImage:
    """Return a copy of `image` with a text banner of `lines` along the top."""
    _, ImageDraw, ImageFont = _load_pillow()
    annotated = _open_image(image).copy()
    draw = ImageDraw.Draw(annotated)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    banner_h = (font_size + 4) * max(len(lines), 1) + 6
    draw.rectangle([(0, 0), (annotated.width, banner_h)], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((4, 3 + i * (font_size + 4)), line, fill=text_color, font=font)
    return annotated


_SKELETON = (
    (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
)  # fmt: skip


def render_segments(image: Any, result: SegmentationResult, *, line_width: int = 2, **_ignored: Any) -> PILImage:
    """Draw each segment's outline (and label colour) on a copy of `image`."""
    _, ImageDraw, _ = _load_pillow()
    annotated = _open_image(image).copy()
    draw = ImageDraw.Draw(annotated)
    for seg in result.segments:
        color = _color_for_label(seg.label)
        if len(seg.polygon) >= 3:
            draw.polygon([tuple(p) for p in seg.polygon], outline=color, width=line_width)
        else:
            draw.rectangle(
                [(seg.box.x_min, seg.box.y_min), (seg.box.x_max, seg.box.y_max)],
                outline=color,
                width=line_width,
            )
    return annotated


def render_poses(
    image: Any, result: PoseResult, *, line_width: int = 2, min_keypoint_conf: float = 0.3, **_ignored: Any
) -> PILImage:
    """Draw each person's skeleton and keypoints (COCO-17 models) on a copy of `image`."""
    _, ImageDraw, _ = _load_pillow()
    annotated = _open_image(image).copy()
    draw = ImageDraw.Draw(annotated)
    for pose in result.poses:
        color = _color_for_label(pose.label)
        pts = pose.keypoints
        if len(pts) == 17:
            for a, b in _SKELETON:
                if min(pts[a].confidence.value, pts[b].confidence.value) >= min_keypoint_conf:
                    draw.line([(pts[a].x, pts[a].y), (pts[b].x, pts[b].y)], fill=color, width=line_width)
        for kp in pts:
            if kp.confidence.value >= min_keypoint_conf:
                r = line_width + 1
                draw.ellipse([(kp.x - r, kp.y - r), (kp.x + r, kp.y + r)], fill=color)
    return annotated


def draw_result(image: Any, result: Any, **options: Any) -> PILImage:
    """Draw any image-based result on `image` and return it, without saving.

    Detections and faces get boxes, OCR gets polygons and text, and
    classification / moderation results get a caption banner.
    """
    if isinstance(result, OcrResult):
        return render_ocr(image, result, **options)
    if isinstance(result, ClassificationResult):
        top = sorted(result.entries, key=lambda e: -e.confidence.value)[:5]
        return render_caption(image, [f"{e.label} {e.confidence.value:.2f}" for e in top], **options)
    if isinstance(result, ModerationResult):
        return render_caption(
            image,
            [f"{'FLAGGED ' if lb.flagged else ''}{lb.category} {lb.confidence.value:.2f}" for lb in result.labels],
            **options,
        )
    if isinstance(result, SegmentationResult):
        return render_segments(image, result, **options)
    if isinstance(result, PoseResult):
        return render_poses(image, result, **options)
    items = getattr(result, "detections", None) or getattr(result, "faces", None)
    if items is None and not hasattr(result, "frames"):
        items = []
    if items is None:
        raise ValidationError("Video results cannot be drawn on a single image; use annotate()")
    keys = {"show_labels", "show_confidence", "box_color", "text_color", "line_width", "font_size"}
    kwargs = {k: v for k, v in options.items() if k in keys}
    return _draw_boxes(
        _open_image(image),
        items,
        show_labels=kwargs.get("show_labels", True),
        show_confidence=kwargs.get("show_confidence", True),
        box_color=kwargs.get("box_color"),
        text_color=kwargs.get("text_color", (255, 255, 255)),
        line_width=kwargs.get("line_width", 3),
        font_size=kwargs.get("font_size", 16),
    )
