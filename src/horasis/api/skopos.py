from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from horasis.access.keys import ApiKeyValidator, InMemoryApiKeyValidator
from horasis.access.limen import AccessContext, limen
from horasis.access.prosopon import Prosopon
from horasis.backend.taxis import Taxis
from horasis.backend.ultralytics_adapter import UltralyticsAdapter
from horasis.foundation.aisthesis import Aisthesis
from horasis.foundation.kanon import Kanon
from horasis.foundation.schemas import (
    ClassificationResult,
    DetectionResult,
    FaceResult,
    Kinesis,
    ModerationResult,
    OcrResult,
    Physis,
    Praxis,
    VideoDetectionResult,
)
from horasis.foundation.sphalma import ValidationError
from horasis.orchestration.organon import Organon


def _to_physis(image: str | Path | bytes | Physis) -> Physis:
    """Normalize any accepted image input into a `Physis`."""
    if isinstance(image, Physis):
        return image
    if isinstance(image, bytes):
        return Physis.from_bytes(image)
    if isinstance(image, str):
        return Physis.from_source(image)
    if isinstance(image, Path):
        return Physis.from_path(image)
    raise ValidationError(f"Unsupported image input type: {type(image).__name__}")


def _to_kinesis(video: str | Path | bytes | Kinesis) -> Kinesis:
    """Normalize any accepted video input into a `Kinesis`."""
    if isinstance(video, Kinesis):
        return video
    if isinstance(video, bytes):
        return Kinesis.from_bytes(video)
    if isinstance(video, str):
        return Kinesis.from_source(video)
    if isinstance(video, Path):
        return Kinesis.from_path(video)
    raise ValidationError(f"Unsupported video input type: {type(video).__name__}")


class _Capability:
    """Shared base for the small, single-purpose public-API classes."""

    def __init__(self, *, access: AccessContext, organon: Organon) -> None:
        self._access = access
        self._organon = organon


class ObjectDetector(_Capability):
    """Public API for object detection."""

    @limen(Praxis.DETECT)
    def detect(
        self,
        image: str | Path | bytes | Physis,
        *,
        backend: str | None = None,
        principal: Prosopon | None = None,
    ) -> DetectionResult:
        """Detect objects in `image`.

        Args:
            image: A file path, raw bytes, or a pre-built `Physis`.
            api_key: Caller's `horasis` API key.
            backend: Optional explicit backend name to use.

        Returns:
            A `DetectionResult` listing every detected object instance.
        """
        result = self._organon.run(Praxis.DETECT, _to_physis(image), backend=backend)
        assert isinstance(result, DetectionResult)
        return result

    def detect_and_annotate(
        self,
        image: str | Path | bytes | Physis,
        *,
        api_key: str,
        backend: str | None = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        output_path: str | Path | None = None,
        **viz_options: object,
    ) -> tuple[DetectionResult, object]:
        """Detect objects in `image` and return an annotated copy in one call.

        Equivalent to calling `detect()` then `result.annotate()` yourself,
        except `image` is only loaded/fetched once and reused for both
        steps -- so a URL is only downloaded a single time, not twice.

        Args:
            image: A file path, URL, raw bytes, or a pre-built `Physis`.
            api_key: Caller's `horasis` API key.
            backend: Optional explicit backend name to use for detection.
            show_labels: Draw each detection's class label on the image.
            show_confidence: Draw each detection's confidence score on the image.
            output_path: If given, also save the annotated image to this path
                (e.g. `"bus_annotated.jpg"`).
            **viz_options: Additional drawing options, forwarded to
                `DetectionResult.annotate()` -> `horasis.viz.render_detections()`.
                Accepted keys:
                    - box_color (tuple[int, int, int]): fixed RGB color for
                      every box. If omitted, each label gets its own
                      deterministic color so classes stay visually distinct.
                    - text_color (tuple[int, int, int]): RGB color for the
                      label/confidence caption text. Defaults to white.
                    - line_width (int): box border thickness in pixels.
                      Defaults to 3.
                    - font_size (int): caption font size in points.
                      Defaults to 16.

        Returns:
            A `(result, annotated_image)` tuple:
                - `result`: the `DetectionResult`, with the raw list of
                  detections (label, confidence, box) if you need the data
                  as well as the picture.
                - `annotated_image`: a `PIL.Image.Image` with boxes and
                  (optionally) labels/confidence drawn on it.

        Raises:
            VisualizationUnavailableError: if Pillow is not installed.
                Install it with: pip install 'horasis[viz]'
        """
        physis = _to_physis(image)
        result = self.detect(physis, api_key=api_key, backend=backend)
        annotated = result.annotate(
            physis,
            show_labels=show_labels,
            show_confidence=show_confidence,
            output_path=output_path,
            **viz_options,
        )
        return result, annotated

    def detect_and_annotate_batch(
        self,
        images: Sequence[str | Path | bytes | Physis],
        *,
        api_key: str,
        backend: str | None = None,
        show_labels: bool = True,
        show_confidence: bool = True,
        output_dir: str | Path | None = None,
        filenames: Sequence[str] | None = None,
        output_paths: Sequence[str | Path] | None = None,
        **viz_options: object,
    ) -> list[tuple[DetectionResult, object]]:
        """Run `detect_and_annotate` over each image in `images`, in order.

        Args:
            images: Any mix of file paths, URLs, raw bytes, or `Physis`.
            output_dir: Directory to save annotated images into. Used with
                `filenames` if given, otherwise falls back to auto-numbered
                names (`annotated_0.jpg`, `annotated_1.jpg`, ...).
            filenames: Exact filenames (not full paths) to use inside
                `output_dir`, one per image, in order -- e.g. pass your own
                database IDs: `[f"{record.id}.jpg" for record in records]`.
                Must be the same length as `images`. Ignored if
                `output_paths` is given.
            output_paths: Full explicit paths, one per image, in order --
                takes precedence over `output_dir`/`filenames` and can point
                anywhere (not just inside a single directory). Must be the
                same length as `images`.
            **viz_options: Additional drawing options applied to every image
                in the batch. See `detect_and_annotate()` for the accepted
                keys (`box_color`, `text_color`, `line_width`, `font_size`).

        Returns:
            One `(result, annotated_image)` pair per input image, in order.

        Note:
            Each image consumes one unit of quota (same as calling `detect`
            that many times individually). If a `QuotaExceededError` or any
            other error is raised partway through, the exception propagates
            immediately and results already collected are lost -- this does
            not attempt partial-batch recovery.
        """
        if output_paths is not None and len(output_paths) != len(images):
            raise ValidationError("output_paths must be the same length as images")
        if filenames is not None and len(filenames) != len(images):
            raise ValidationError("filenames must be the same length as images")

        results: list[tuple[DetectionResult, object]] = []
        for idx, image in enumerate(images):
            if output_paths is not None:
                output_path = Path(output_paths[idx])
            elif output_dir is not None and filenames is not None:
                output_path = Path(output_dir) / filenames[idx]
            elif output_dir is not None:
                output_path = Path(output_dir) / f"annotated_{idx}.jpg"
            else:
                output_path = None

            results.append(
                self.detect_and_annotate(
                    image,
                    api_key=api_key,
                    backend=backend,
                    show_labels=show_labels,
                    show_confidence=show_confidence,
                    output_path=output_path,
                    **viz_options,
                )
            )
        return results

    @limen(Praxis.DETECT)
    def detect_video(
        self,
        video: str | Path | bytes | Kinesis,
        *,
        backend: str | None = None,
        sample_every_n_frames: int = 1,
        principal: Prosopon | None = None,
    ) -> VideoDetectionResult:
        """Detect objects across a video, frame by frame.

        Consumes one unit of quota per call, regardless of how many frames
        the video contains -- not one unit per frame.

        Args:
            video: A local file path, URL, raw bytes, or a pre-built `Kinesis`.
                URLs are streamed directly by the backend rather than
                downloaded up front (videos can be large).
            backend: Optional explicit backend name to use.
            sample_every_n_frames: Only run inference on every Nth frame
                (1 = every frame, 2 = every other frame, etc.) -- a simple
                way to trade detail for speed on long videos.

        Returns:
            A `VideoDetectionResult` with one `FrameDetections` per
            processed frame, in order.

        Raises:
            BackendUnavailableError: if the resolved backend does not
                support video for this praxis.
        """
        kinesis = _to_kinesis(video)
        result = self._organon.run_video(
            Praxis.DETECT,
            kinesis,
            backend=backend,
            sample_every_n_frames=sample_every_n_frames,
        )
        assert isinstance(result, VideoDetectionResult)
        return result

    @limen(Praxis.DETECT)
    def detect_video_and_annotate(
        self,
        video: str | Path | bytes | Kinesis,
        *,
        output_path: str | Path,
        backend: str | None = None,
        sample_every_n_frames: int = 1,
        show_labels: bool = True,
        show_confidence: bool = True,
        box_color: tuple[int, int, int] | None = None,
        text_color: tuple[int, int, int] = (255, 255, 255),
        line_width: int = 3,
        font_size: int = 16,
        principal: Prosopon | None = None,
    ) -> tuple[VideoDetectionResult, Path]:
        """Detect and annotate a video in a single decode pass.

        Unlike calling `detect_video()` then `VideoDetectionResult.annotate()`
        separately -- which decodes (or downloads, for a URL) the video
        *twice* -- this reuses the same frame the backend already decoded
        internally while running inference, so the video is only ever
        decoded once. For a remote URL, this also means only one fetch
        instead of two.

        Consumes one unit of quota, same as `detect_video`.

        Args:
            video: A local path, URL, raw bytes, or `Kinesis`.
            output_path: Where to write the annotated video (e.g. `"out.mp4"`).
            backend: Optional explicit backend name to use.
            sample_every_n_frames: Only run inference on every Nth frame
                (1 = every frame, 2 = every other frame, etc.).
            show_labels: Draw each detection's class label.
            show_confidence: Draw each detection's confidence score.
            box_color: Fixed RGB color for every box; if omitted, derived
                per label, same as `render_detections`.
            text_color: RGB color for the caption text.
            line_width: Box border thickness in pixels.
            font_size: Caption font size in points.

        Returns:
            A `(result, output_path)` tuple: the `VideoDetectionResult`
            with the raw per-frame detection data, and the path to the
            written annotated video file.

        Raises:
            VisualizationUnavailableError: if Pillow or
                opencv-python-headless is not installed.
                Install both with: pip install 'horasis[viz]'
            BackendUnavailableError: if the resolved backend does not
                support the combined single-pass operation for this praxis
                (the default Ultralytics backend does, for detection).
        """
        kinesis = _to_kinesis(video)
        result, out_path = self._organon.run_video_and_annotate(
            Praxis.DETECT,
            kinesis,
            output_path=output_path,
            backend=backend,
            sample_every_n_frames=sample_every_n_frames,
            show_labels=show_labels,
            show_confidence=show_confidence,
            box_color=box_color,
            text_color=text_color,
            line_width=line_width,
            font_size=font_size,
        )
        assert isinstance(result, VideoDetectionResult)
        return result, Path(out_path)


class ImageClassifier(_Capability):
    """Public API for whole-image classification."""

    @limen(Praxis.CLASSIFY)
    def classify(
        self,
        image: str | Path | bytes | Physis,
        *,
        backend: str | None = None,
        principal: Prosopon | None = None,
    ) -> ClassificationResult:
        """Classify `image`, returning ranked label/confidence entries."""
        result = self._organon.run(Praxis.CLASSIFY, _to_physis(image), backend=backend)
        assert isinstance(result, ClassificationResult)
        return result


class TextExtractor(_Capability):
    """Public API for OCR / text extraction."""

    @limen(Praxis.OCR)
    def extract_text(
        self,
        image: str | Path | bytes | Physis,
        *,
        backend: str | None = None,
        principal: Prosopon | None = None,
    ) -> OcrResult:
        """Extract text from `image` as a sequence of recognized glyphs."""
        result = self._organon.run(Praxis.OCR, _to_physis(image), backend=backend)
        assert isinstance(result, OcrResult)
        return result


class FaceDetector(_Capability):
    """Public API for face detection."""

    @limen(Praxis.FACE)
    def detect_faces(
        self,
        image: str | Path | bytes | Physis,
        *,
        backend: str | None = None,
        principal: Prosopon | None = None,
    ) -> FaceResult:
        """Detect faces in `image`."""
        result = self._organon.run(Praxis.FACE, _to_physis(image), backend=backend)
        assert isinstance(result, FaceResult)
        return result


class ContentModerator(_Capability):
    """Public API for content moderation."""

    @limen(Praxis.MODERATE)
    def moderate(
        self,
        image: str | Path | bytes | Physis,
        *,
        backend: str | None = None,
        principal: Prosopon | None = None,
    ) -> ModerationResult:
        """Score `image` against moderation categories."""
        result = self._organon.run(Praxis.MODERATE, _to_physis(image), backend=backend)
        assert isinstance(result, ModerationResult)
        return result


class Skopos:
    """The `horasis` client. Construct one and call its capability attributes.

    Example:
        >>> validator = InMemoryApiKeyValidator()
        >>> validator.register_simple("demo-key", principal_id="alice", tier=Klimax.FREE)
        >>> client = Skopos.build(validator=validator)
        >>> result = client.detection.detect("photo.jpg", api_key="demo-key")

    Attributes:
        detection: :class:`ObjectDetector`
        classification: :class:`ImageClassifier`
        ocr: :class:`TextExtractor`
        face: :class:`FaceDetector`
        moderation: :class:`ContentModerator`
    """

    def __init__(
        self,
        *,
        kanon: Kanon,
        taxis: Taxis,
        validator: ApiKeyValidator,
        aisthesis: Aisthesis | None = None,
    ) -> None:
        self._kanon = kanon
        self._taxis = taxis
        self._validator = validator
        access = AccessContext.build(kanon=kanon, validator=validator, aisthesis=aisthesis)
        organon = Organon(kanon=kanon, taxis=taxis)

        self.detection = ObjectDetector(access=access, organon=organon)
        self.classification = ImageClassifier(access=access, organon=organon)
        self.ocr = TextExtractor(access=access, organon=organon)
        self.face = FaceDetector(access=access, organon=organon)
        self.moderation = ContentModerator(access=access, organon=organon)

    @property
    def kanon(self) -> Kanon:
        return self._kanon

    @property
    def taxis(self) -> Taxis:
        return self._taxis

    @classmethod
    def build(
        cls,
        *,
        kanon: Kanon | None = None,
        validator: ApiKeyValidator | None = None,
        aisthesis: Aisthesis | None = None,
    ) -> Skopos:
        """Convenience constructor wiring up the default Ultralytics backend.

        Args:
            kanon: Configuration; defaults to `Kanon.default()`.
            validator: API key validator; defaults to an empty in-memory
                validator (register keys via the returned client's
                `_validator`, or supply your own production validator).
            aisthesis: Telemetry sink; defaults to a no-op.
        """
        kanon = kanon or Kanon.default()
        validator = validator or InMemoryApiKeyValidator()
        taxis = Taxis()
        taxis.register(UltralyticsAdapter())
        return cls(kanon=kanon, taxis=taxis, validator=validator, aisthesis=aisthesis)
