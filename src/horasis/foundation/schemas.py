from __future__ import annotations

import logging
import time
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Generic, Literal, Self, TypeVar
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator

from horasis.foundation.sphalma import Sphalma, ValidationError

logger = logging.getLogger(__name__)


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

    Build one with the ``from_*`` constructors, or let the public API do it
    for you. Backends should read pixels through
    :func:`horasis.foundation.imaging.load_rgb`, never from ``path``/``data``
    directly -- that is what guarantees size limits and EXIF orientation.

    Attributes:
        source: Original source identifier: a path, URL, ``"<bytes>"``, etc.
        path: Local file path, when the image came from one.
        data: Encoded image bytes, when supplied and not yet decoded.
        pixels: Decoded, upright RGB ``PIL.Image`` once the image was loaded.
        width, height: Decoded size in pixels, once known.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    source: str = Field(description="Original source identifier: a path, URL, or '<bytes>'.")
    path: str | Path | None = None
    data: bytes | None = None
    pixels: Any = Field(default=None, exclude=True, repr=False)
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> Physis:
        """Wrap a local file path. URLs are rejected; use `from_url` explicitly."""
        path_str = str(path)
        if urlparse(path_str).scheme in ("http", "https"):
            raise ValidationError(
                f"{path_str!r} is a URL; use Physis.from_url() (network fetch is opt-in)"
            )
        return cls(source=path_str, path=Path(path))

    @classmethod
    def from_bytes(cls, data: bytes, *, source: str = "<bytes>") -> Physis:
        return cls(source=source, data=data)

    @classmethod
    def from_pil(cls, image: Any) -> Physis:
        """Wrap a ``PIL.Image.Image`` (decoded and EXIF-corrected immediately)."""
        from horasis.foundation.imaging import to_physis

        return to_physis(image)

    @classmethod
    def from_array(cls, array: Any) -> Physis:
        """Wrap a numpy array in RGB order (see :func:`~horasis.foundation.imaging.array_to_image`)."""
        from horasis.foundation.imaging import to_physis

        return to_physis(array)

    @classmethod
    def from_url(cls, url: str, *, timeout: float = 30.0, max_bytes: int = 20 * 1024 * 1024) -> Physis:
        """Fetch `url` and wrap the downloaded bytes as a `Physis`.

        Calling this directly *is* the opt-in to a network request.

        Raises:
            ValidationError: if the URL cannot be fetched or exceeds `max_bytes`.
        """
        from horasis.foundation.imaging import fetch_url

        return cls(source=url, data=fetch_url(url, timeout=timeout, max_bytes=max_bytes))

    @classmethod
    def from_source(cls, source: str, *, allow_urls: bool = False) -> Physis:
        """Build a `Physis` from a local file path, or a URL if `allow_urls` is True."""
        if source.startswith(("http://", "https://")):
            if not allow_urls:
                raise ValidationError(
                    "URL inputs are disabled; pass allow_urls=True (or set "
                    "Kanon.inputs.allow_urls) to opt in."
                )
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

    content_type: str | None = None
    attempts: list[dict[str, Any]] = [
        {"method": "HEAD"},
        {"headers": {"Range": "bytes=0-0"}},
    ]
    for kwargs in attempts:
        try:
            req = urllib.request.Request(url, **kwargs)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                content_type = response.headers.get("Content-Type")
            break
        except (urllib.error.URLError, urllib.error.HTTPError) as exc:
            logger.debug(f"Failed to check URL {url} with kwargs {kwargs}: {exc}")
            continue

    if content_type is None:
        return 

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
    def from_url(cls, url: str, *, verify: bool = True, timeout: float = 10.0) -> Kinesis:
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
    def from_source(cls, source: str) -> Kinesis:
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


Point = tuple[float, float]


class Glyph(BaseModel):
    """A recognized unit of text produced by an OCR task.

    Attributes:
        text: The recognized string.
        confidence: Recognition confidence.
        box: Axis-aligned bounding box.
        polygon: The tight outline as 4+ ``(x, y)`` points, when the engine
            provides one (rotated or skewed text).
        line_index: Zero-based reading-order line this glyph belongs to.
        level: ``"word"`` for engine segments, ``"line"`` for merged lines.
    """

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: Doxa
    box: BoundingBox | None = None
    polygon: list[Point] | None = None
    line_index: int | None = None
    level: Literal["word", "line"] = "word"


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

    # Name of the list field holding this result's items (used by `filter`).
    _items_field: ClassVar[str | None] = None

    def to_dict(self) -> dict[str, Any]:
        """The result as plain JSON-compatible Python data."""
        return self.model_dump(mode="json")

    def to_json(self, *, indent: int | None = None) -> str:
        """The result as a JSON string."""
        return self.model_dump_json(indent=indent)

    def filter(
        self, *, min_conf: float | None = None, labels: Iterable[str] | None = None
    ) -> Self:
        """Return a copy keeping only items that match.

        Args:
            min_conf: Keep items with confidence >= this.
            labels: Keep items whose label (class name, OCR text, or
                moderation category) is in this collection.

        Results without a list of items (none today) are returned unchanged.
        """
        field = self._items_field
        if field is None:
            return self
        wanted = set(labels) if labels is not None else None
        kept = [
            item
            for item in getattr(self, field)
            if (min_conf is None or item.confidence.value >= min_conf)
            and (wanted is None or _label_of(item) in wanted)
        ]
        return self.model_copy(update={field: kept})

    def draw(self, image: Any, **options: Any) -> Any:
        """Return a PIL image with this result drawn on `image` (nothing is saved).

        Needs the optional ``viz`` extra only for video; images need Pillow,
        which is a core dependency. Extra `options` go to the drawing code
        (``show_labels``, ``show_confidence``, ``box_color``, ``line_width``,
        ``font_size``, ``text_color``).
        """
        from horasis.viz import draw_result

        return draw_result(image, self, **options)


def _label_of(item: Any) -> str:
    for attr in ("label", "category", "text"):
        value = getattr(item, attr, None)
        if isinstance(value, str):
            return value
    return ""


class FaceEstimate(BaseModel):
    """Rough, model-estimated attributes for a detected face.

    These come from a computer-vision model's statistical estimate, not
    verified facts about the person. Expect real error rates -- do not
    treat these as ground truth, and do not use them as the sole basis
    for any decision about a person.
    """

    model_config = ConfigDict(frozen=True)

    estimated_age: float | None = None
    estimated_gender: str | None = None
    gender_confidence: Doxa | None = None


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
    _items_field: ClassVar[str | None] = "detections"

    def to_coco(
        self,
        *,
        image_id: int = 1,
        file_name: str | None = None,
        width: int | None = None,
        height: int | None = None,
        category_ids: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Export as a COCO-format dict (``images``, ``annotations``, ``categories``).

        Boxes become ``[x, y, width, height]``. Category ids come from
        `category_ids` if given, otherwise 1..N over the sorted labels seen.
        The model score is kept as an extra ``score`` field on each annotation.
        """
        return detections_to_coco(
            [(image_id, file_name, width, height, self.detections)], category_ids=category_ids
        )

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

    def filter(
        self, *, min_conf: float | None = None, labels: Iterable[str] | None = None
    ) -> Self:
        """Filter the detections inside every frame (frames themselves are kept)."""
        wanted = set(labels) if labels is not None else None
        frames = [
            f.model_copy(
                update={
                    "detections": [
                        d
                        for d in f.detections
                        if (min_conf is None or d.confidence.value >= min_conf)
                        and (wanted is None or d.label in wanted)
                    ]
                }
            )
            for f in self.frames
        ]
        return self.model_copy(update={"frames": frames})

    def to_coco(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        category_ids: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """COCO export with one "image" per processed frame (id = frame index + 1)."""
        return detections_to_coco(
            [
                (f.frame_index + 1, f"frame_{f.frame_index:06d}", width, height, f.detections)
                for f in self.frames
            ],
            category_ids=category_ids,
        )

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
    _items_field: ClassVar[str | None] = "entries"

    @property
    def top(self) -> ClassificationEntry | None:
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.confidence.value)


class OcrResult(Theoria):
    """Result of an OCR / text-extraction :class:`Praxis`.

    ``glyphs`` are the engine's word-level segments in reading order. When
    they carry a ``line_index``, :attr:`lines` merges them into line-level
    glyphs and :attr:`text` separates lines with newlines.
    """

    praxis: Praxis = Praxis.OCR
    glyphs: list[Glyph] = Field(default_factory=list)
    _items_field: ClassVar[str | None] = "glyphs"

    @property
    def lines(self) -> list[Glyph]:
        """Line-level glyphs (text joined left to right, mean confidence, union box)."""
        grouped: dict[int, list[Glyph]] = {}
        for g in self.glyphs:
            if g.line_index is not None:
                grouped.setdefault(g.line_index, []).append(g)
        out: list[Glyph] = []
        for idx in sorted(grouped):
            words = grouped[idx]
            boxes = [w.box for w in words if w.box is not None]
            union = (
                BoundingBox(
                    x_min=min(b.x_min for b in boxes),
                    y_min=min(b.y_min for b in boxes),
                    x_max=max(b.x_max for b in boxes),
                    y_max=max(b.y_max for b in boxes),
                )
                if boxes
                else None
            )
            mean = sum(w.confidence.value for w in words) / len(words)
            out.append(
                Glyph(
                    text=" ".join(w.text for w in words),
                    confidence=Doxa(value=mean),
                    box=union,
                    line_index=idx,
                    level="line",
                )
            )
        return out

    @property
    def text(self) -> str:
        lines = self.lines
        if lines and all(g.line_index is not None for g in self.glyphs):
            return "\n".join(line.text for line in lines)
        return " ".join(g.text for g in self.glyphs)


class Face(Detection):
    """A detected face: a box plus optional 5-point landmarks.

    Detection only -- no identity is recognized or inferred.

    Attributes:
        landmarks: Named ``(x, y)`` points: ``right_eye``, ``left_eye``,
            ``nose``, ``mouth_right``, ``mouth_left`` (as seen in the image,
            i.e. "right" is the viewer's left, following YuNet's convention).
    """

    landmarks: dict[str, Point] | None = None


class FaceResult(Theoria):
    """Result of a face-detection :class:`Praxis`."""

    praxis: Praxis = Praxis.FACE
    faces: list[Face] = Field(default_factory=list)
    _items_field: ClassVar[str | None] = "faces"

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
    _items_field: ClassVar[str | None] = "labels"

    @property
    def is_flagged(self) -> bool:
        return any(label.flagged for label in self.labels)


def detections_to_coco(
    images: Iterable[tuple[int, str | None, int | None, int | None, list[Detection]]],
    *,
    category_ids: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Build a COCO detection dict from ``(image_id, file_name, width, height, detections)``."""
    entries = list(images)
    if category_ids is None:
        labels = sorted({d.label for _, _, _, _, dets in entries for d in dets})
        category_ids = {name: i for i, name in enumerate(labels, start=1)}
    coco_images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    next_id = 1
    for image_id, file_name, width, height, dets in entries:
        record: dict[str, Any] = {"id": image_id}
        if file_name is not None:
            record["file_name"] = file_name
        if width is not None:
            record["width"] = width
        if height is not None:
            record["height"] = height
        coco_images.append(record)
        for d in dets:
            if d.label not in category_ids:
                continue
            w = d.box.x_max - d.box.x_min
            h = d.box.y_max - d.box.y_min
            annotations.append(
                {
                    "id": next_id,
                    "image_id": image_id,
                    "category_id": category_ids[d.label],
                    "bbox": [d.box.x_min, d.box.y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "score": d.confidence.value,
                }
            )
            next_id += 1
    return {
        "images": coco_images,
        "annotations": annotations,
        "categories": [{"id": cid, "name": name} for name, cid in category_ids.items()],
    }


T = TypeVar("T")


@dataclass(frozen=True)
class BatchItem(Generic[T]):
    """Outcome for one input of a ``*_batch`` call: a result or an error, never both.

    Attributes:
        index: Position of the input in the submitted list.
        source: Short description of the input (path, ``"<bytes>"``, ...).
        result: The result, when the item succeeded.
        error: The `Sphalma` that stopped this item (quota, validation, backend...).
    """

    index: int
    source: str
    result: T | None = None
    error: Sphalma | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> T:
        """The result, or raise this item's error."""
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


class FrameResult(BaseModel):
    """Detections for one frame of a stream, with its position in the source."""

    model_config = ConfigDict(frozen=True)

    frame_index: int
    timestamp_s: float
    result: DetectionResult
