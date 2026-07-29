from __future__ import annotations

import time
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from horasis.foundation.sphalma import ValidationError


class Praxis(str, Enum):
    """The task an inference call performs."""

    DETECT = "detect"
    CLASSIFY = "classify"
    OCR = "ocr"
    FACE = "face"
    MODERATE = "moderate"


class Doxa(BaseModel):
    """A confidence score in ``[0.0, 1.0]``."""

    model_config = ConfigDict(frozen=True)

    value: float = Field(ge=0.0, le=1.0)

    def __float__(self) -> float:
        return self.value

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Doxa({self.value:.4f})"


class Physis(BaseModel):
    """Backend-agnostic representation of an input image.

    Exactly one of ``path``, ``data``, or ``array_shape`` is expected to be
    meaningful for a given source; adapters decide how to load the bytes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source: str = Field(description="Original source identifier: a path, URL, or '<bytes>'.")
    path: str | Path | None = None
    data: bytes | None = None
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> Physis:
        path_str = str(path)
        if urlparse(path_str).scheme in ("http", "https"):
            return cls(source=path_str, path=path_str)
        return cls(source=path_str, path=Path(path))

    @classmethod
    def from_bytes(cls, data: bytes, *, source: str = "<bytes>") -> Physis:
        return cls(source=source, data=data)

    @classmethod
    def from_url(cls, url: str, *, timeout: float = 30.0) -> Physis:
        """Fetch `url` and wrap the downloaded bytes as a `Physis`.

        Raises:
            ValidationError: if the URL cannot be fetched.
        """

        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                data = response.read()
        except Exception as exc:
            raise ValidationError(f"Failed to fetch image from URL: {url}") from exc
        return cls(source=url, data=data)

    @classmethod
    def from_source(cls, source: str) -> Physis:
        """Build a `Physis` from a string that is either a URL or a local file path."""
        if source.startswith(("http://", "https://")):
            return cls.from_url(source)
        return cls.from_path(source)


def _check_url_is_video(url: str, *, timeout: float) -> None:
    """Best-effort check that `url` responds with video content.

    Tries a HEAD request first, then falls back to a 1-byte ranged GET
    (some servers reject HEAD). If neither succeeds -- e.g. the server
    doesn't support either -- verification is skipped rather than
    blocking a URL we simply couldn't check. Only raises when a
    Content-Type was actually returned and it clearly isn't video.
    """
    import urllib.request

    content_type: str | None = None
    attempts: list[dict[str, Any]] = [
        {"method": "HEAD"},
        {"headers": {"Range": "bytes=0-0"}},
    ]
    for kwargs in attempts:
        try:
            req = urllib.request.Request(url, **kwargs)
            with urllib.request.urlopen(req, timeout=timeout) as response:  # noqa: S310
                content_type = response.headers.get("Content-Type")
            break
        except Exception:
            continue

    if content_type is None:
        return  # couldn't determine -- don't block a possibly-valid URL

    normalized = content_type.split(";")[0].strip().lower()
    if normalized.startswith("video/") or normalized == "application/octet-stream":
        return

    raise ValidationError(
        f"This URL doesn't look like a direct link to a video file "
        f"(server reported content-type {content_type!r}). Link to the raw "
        f"video file itself, not a webpage, player, or platform page. "
        f"Pass verify=False to skip this check if you're confident the URL is correct."
    )


class Kinesis(BaseModel):
    """Backend-agnostic representation of an input video."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source: str = Field(description="Original source identifier: a path, URL, or '<bytes>'.")
    path: Path | None = None
    url: str | None = None
    data: bytes | None = None
    fps: float | None = Field(default=None, description="Frames per second, if known.")

    @classmethod
    def from_path(cls, path: str | Path) -> Kinesis:
        p = Path(path)
        return cls(source=str(p), path=p)

    @classmethod
    def from_bytes(cls, data: bytes, *, source: str = "<bytes>") -> Kinesis:
        return cls(source=source, data=data)

    @classmethod
    def from_url(cls, url: str, *, verify: bool = True, timeout: float = 10.0) -> "Kinesis":
        """Wrap a video URL for streaming by the backend (not downloaded here).

        Args:
            url: A direct link to a video file -- not a webpage, player, or
                platform "watch" page. Works the same regardless of which
                site or CDN hosts it, as long as the URL itself serves raw
                video bytes.
            verify: If True (default), makes a lightweight check (HEAD, or
                a 1-byte ranged GET as a fallback) to confirm the URL
                actually responds with video content before accepting it,
                so a bad link fails immediately with a clear message
                instead of a confusing error deep inside the backend later.
                If the server doesn't answer that check at all, verification
                is skipped rather than blocking a possibly-valid URL.
            timeout: Seconds to wait for the verification request.

        Raises:
            ValidationError: if `verify` is True and the URL clearly does
                not point directly to a video file (e.g. it returns an
                HTML page).
        """
        if verify:
            _check_url_is_video(url, timeout=timeout)
        return cls(source=url, url=url)

    @classmethod
    def from_source(cls, source: str) -> "Kinesis":
        """Build a `Kinesis` from a string that is either a URL or a local file path."""
        if source.startswith(("http://", "https://")):
            return cls.from_url(source)
        return cls.from_path(source)


class BoundingBox(BaseModel):
    """Axis-aligned pixel bounding box, ``(x_min, y_min, x_max, y_max)``."""

    model_config = ConfigDict(frozen=True)

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @field_validator("x_max")
    @classmethod
    def _x_order(cls, v: float, info: Any) -> float:
        x_min = info.data.get("x_min")
        if x_min is not None and v < x_min:
            raise ValueError("x_max must be >= x_min")
        return v


class Glyph(BaseModel):
    """A single recognized unit of text produced by an OCR task."""

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: Doxa
    box: BoundingBox | None = None


class Theoria(BaseModel):
    """Base class for every result ("view") returned by the public API.

    Attributes:
        praxis: The task type that produced this result.
        backend: Name of the backend adapter that produced the result.
        latency_ms: Wall-clock inference time in milliseconds.
        created_at: Unix timestamp when the result was constructed.
    """

    model_config = ConfigDict(frozen=True)

    praxis: Praxis
    backend: str
    latency_ms: float
    created_at: float = Field(default_factory=lambda: time.time())


class Detection(BaseModel):
    """A single detected object instance."""

    model_config = ConfigDict(frozen=True)

    label: str
    confidence: Doxa
    box: BoundingBox
    face_estimate: FaceEstimate | None = None


class DetectionResult(Theoria):
    """Result of an object-detection :class:`Praxis`."""

    praxis: Praxis = Praxis.DETECT
    detections: list[Detection] = Field(default_factory=list)

    def annotate(
        self,
        image: Any,
        *,
        show_labels: bool = True,
        show_confidence: bool = True,
        output_path: Any = None,
        **options: Any,
    ) -> Any:
        """Draw `self.detections` on `image` and return the annotated image.

        Convenience wrapper over :func:`horasis.viz.render_detections`.
        Requires the optional `viz` extra: `pip install horasis[viz]`.
        """
        from horasis.viz import render_detections

        return render_detections(
            image,
            self,
            show_labels=show_labels,
            show_confidence=show_confidence,
            output_path=output_path,
            **options,
        )


class FrameDetections(BaseModel):
    """Detections found in a single sampled frame of a video."""

    model_config = ConfigDict(frozen=True)

    frame_index: int
    timestamp_s: float | None = None
    detections: list[Detection] = Field(default_factory=list)


class VideoDetectionResult(Theoria):
    """Result of running object detection over a video (:class:`Kinesis` input).

    Frames are sampled and processed in order; `frames` holds one
    :class:`FrameDetections` per *processed* frame (see
    `sample_every_n_frames` on `ObjectDetector.detect_video`) -- not
    necessarily one per frame in the source video.
    """

    praxis: Praxis = Praxis.DETECT
    frames: list[FrameDetections] = Field(default_factory=list)
    frame_count: int = 0
    fps: float | None = None

    def annotate(
        self,
        video: Any,
        *,
        output_path: Any,
        show_labels: bool = True,
        show_confidence: bool = True,
        **options: Any,
    ) -> Any:
        """Draw `self.frames` back onto `video` and write an annotated video file.

        Convenience wrapper over :func:`horasis.viz.render_video`. Requires
        the optional `viz` extra: `pip install horasis[viz]`. Note this
        re-reads `video` in a separate pass from whatever originally
        produced this result via `detect_video`.
        """
        from horasis.viz import render_video

        return render_video(
            video,
            self,
            output_path=output_path,
            show_labels=show_labels,
            show_confidence=show_confidence,
            **options,
        )


class ClassificationEntry(BaseModel):
    """A single label/score pair from an image classification task."""

    model_config = ConfigDict(frozen=True)

    label: str
    confidence: Doxa


class ClassificationResult(Theoria):
    """Result of an image-classification :class:`Praxis`."""

    praxis: Praxis = Praxis.CLASSIFY
    entries: list[ClassificationEntry] = Field(default_factory=list)

    @property
    def top(self) -> ClassificationEntry | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.confidence.value)


class OcrResult(Theoria):
    """Result of an OCR / text-extraction :class:`Praxis`."""

    praxis: Praxis = Praxis.OCR
    glyphs: list[Glyph] = Field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(g.text for g in self.glyphs)


class FaceResult(Theoria):
    """Result of a face-detection :class:`Praxis`."""

    praxis: Praxis = Praxis.FACE
    faces: list[Detection] = Field(default_factory=list)

    def annotate(
        self,
        image: Any,
        *,
        show_labels: bool = True,
        show_confidence: bool = True,
        output_path: Any = None,
        **options: Any,
    ) -> Any:
        """Draw `self.faces` on `image` and return the annotated image."""
        from horasis.viz import render_faces

        return render_faces(
            image,
            self,
            show_labels=show_labels,
            show_confidence=show_confidence,
            output_path=output_path,
            **options,
        )


class ModerationLabel(BaseModel):
    """A single content-moderation category and its score."""

    model_config = ConfigDict(frozen=True)

    category: str
    confidence: Doxa
    flagged: bool


class ModerationResult(Theoria):
    """Result of a content-moderation :class:`Praxis`."""

    praxis: Praxis = Praxis.MODERATE
    labels: list[ModerationLabel] = Field(default_factory=list)

    @property
    def is_flagged(self) -> bool:
        return any(label.flagged for label in self.labels)


class FaceEstimate(BaseModel):
    """Rough, model-estimated attributes for a detected face.

    These are statistical estimates from a computer vision model, not
    verified facts -- treat them as approximate, expect real error rates,
    and do not use them as the sole basis for any consequential decision
    about a person.
    """

    model_config = ConfigDict(frozen=True)

    estimated_age: float | None = None
    estimated_gender: str | None = None
    gender_confidence: Doxa | None = None
