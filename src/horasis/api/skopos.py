from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from horasis.access.keys import ApiKeyValidator, InMemoryApiKeyValidator
from horasis.access.limen import AccessContext, limen
from horasis.access.metron import Metron, QuotaStore, UsageReport
from horasis.access.prosopon import Prosopon
from horasis.access.ratelimit import RateLimiter, RedisRateLimiter
from horasis.backend.mechane import Mechane
from horasis.backend.taxis import Taxis
from horasis.backend.ultralytics_adapter import UltralyticsAdapter
from horasis.foundation.aisthesis import Aisthesis
from horasis.foundation.frames import iter_frames
from horasis.foundation.imaging import ImageInput, to_physis
from horasis.foundation.kanon import Kanon
from horasis.foundation.schemas import (
    BatchItem,
    ClassificationResult,
    DetectionResult,
    FaceResult,
    FrameResult,
    Kinesis,
    ModerationResult,
    OcrResult,
    PoseResult,
    Praxis,
    SegmentationResult,
    VideoDetectionResult,
)
from horasis.foundation.sphalma import (
    BackendUnavailableError,
    ConfigurationError,
    InvalidApiKeyError,
    Sphalma,
    ValidationError,
)
from horasis.orchestration.kairos import Kairos
from horasis.orchestration.organon import Organon

logger = logging.getLogger(__name__)


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


def _describe(image: object) -> str:
    """Short, safe label for an input (never the pixel data)."""
    if isinstance(image, str | Path):
        return str(image)
    return getattr(image, "source", None) or f"<{type(image).__name__}>"


class _Capability:
    """Shared base for the small, single-purpose public-API classes."""

    def __init__(self, *, access: AccessContext, organon: Organon) -> None:
        self._access = access
        self._organon = organon

    def _run_batch(
        self,
        fn: Callable[..., Any],
        images: Sequence[ImageInput],
        api_key: str,
        **kwargs: Any,
    ) -> list[BatchItem[Any]]:
        """Run `fn` over `images`; each item succeeds or fails on its own.

        Every item goes through the full access path, so quota and rate
        limits are charged per item, but only after the item passes input
        validation -- items that fail it are not charged. An invalid API key aborts the whole
        batch (it would fail identically for every item); anything else
        that is a `Sphalma` is recorded on that item and the batch goes on.
        """
        items: list[BatchItem[Any]] = []
        for index, image in enumerate(images):
            try:
                items.append(
                    BatchItem(index, _describe(image), result=fn(image, api_key=api_key, **kwargs))
                )
            except InvalidApiKeyError:
                raise
            except Sphalma as exc:
                items.append(BatchItem(index, _describe(image), error=exc))
        return items


class ObjectDetector(_Capability):
    """Public API for object detection."""

    @limen(Praxis.DETECT)
    def detect(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        estimate_demographics: bool = False,
        options: Mapping[str, Any] | None = None,
        principal: Prosopon | None = None,
    ) -> DetectionResult:
        """Detect objects in `image`.

        Args:
            image: A file path, raw bytes, or a pre-built `Physis`.
            api_key: Caller's `horasis` API key.
            backend: Optional explicit backend name to use.
            estimate_demographics: If True, every detection labeled
                `"person"` also gets a `face_estimate` (rough, model-based
                age and gender estimate) attached. Off by default -- opt in
                per call. Requires the optional `demographics` extra:
                `pip install horasis[demographics]`. These are statistical
                estimates with real error rates, not verified facts; treat
                them accordingly.

        Returns:
            A `DetectionResult` listing every detected object instance.
        """
        result = self._organon.run(
            Praxis.DETECT,
            to_physis(image, self._access.kanon.inputs),
            backend=backend,
            estimate_demographics=estimate_demographics,
            **(options or {}),
        )
        assert isinstance(result, DetectionResult)
        return result

    def detect_and_annotate(
        self,
        image: ImageInput,
        *,
        api_key: str,
        backend: str | None = None,
        estimate_demographics: bool = False,
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
        physis = to_physis(image, self._access.kanon.inputs)
        # `@limen` injects the `api_key` keyword; the decorated signature does not declare it.
        result = self.detect(  # type: ignore[call-arg]
            physis, api_key=api_key, backend=backend, estimate_demographics=estimate_demographics
        )
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
        images: Sequence[ImageInput],
        *,
        api_key: str,
        backend: str | None = None,
        estimate_demographics: bool = False,
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
                    estimate_demographics=estimate_demographics,
                    **viz_options,
                )
            )
        return results

    @limen(Praxis.DETECT, validate_image=False)
    def detect_video(
        self,
        video: str | Path | bytes | Kinesis,
        *,
        backend: str | None = None,
        sample_every_n_frames: int = 1,
        estimate_demographics: bool = False,
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
            estimate_demographics: If True, every detection labeled
                `"person"` on a processed frame also gets a `face_estimate`
                attached (see `ObjectDetector.detect`). Off by default.
                This only runs on frames already selected by
                `sample_every_n_frames` -- it does not add its own
                sampling, but it does add real per-person cost on top of
                detection, so a higher `sample_every_n_frames` matters even
                more with this on. Requires the optional `demographics`
                extra: `pip install horasis[demographics]`.

        Returns:
            A `VideoDetectionResult` with one `FrameDetections` per
            processed frame, in order.

        Raises:
            BackendUnavailableError: if the resolved backend does not
                support video for this praxis (the default Ultralytics
                backend currently only supports video for detection, not
                classification, OCR, face, or moderation).
        """
        kinesis = _to_kinesis(video)
        result = self._organon.run_video(
            Praxis.DETECT,
            kinesis,
            backend=backend,
            sample_every_n_frames=sample_every_n_frames,
            estimate_demographics=estimate_demographics,
        )
        assert isinstance(result, VideoDetectionResult)
        return result

    @limen(Praxis.DETECT, validate_image=False)
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
        estimate_demographics: bool = False,
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
            estimate_demographics: If True, every detection labeled
                `"person"` on a processed frame also gets a `face_estimate`
                attached (see `ObjectDetector.detect`). Off by default.
                Note this attaches the estimate to the returned data but
                does not currently draw it on the annotated video -- the
                box/label/confidence drawing is unaffected. Requires the
                optional `demographics` extra:
                `pip install horasis[demographics]`.

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
            estimate_demographics=estimate_demographics,
        )
        assert isinstance(result, VideoDetectionResult)
        return result, Path(str(out_path))

    def detect_batch(
        self,
        images: Sequence[ImageInput],
        *,
        api_key: str,
        backend: str | None = None,
        estimate_demographics: bool = False,
    ) -> list[BatchItem[DetectionResult]]:
        """Detect objects in each of `images`, with a result or error per item.

        Quota (and rate limits) are charged per item. Results come back in
        input order; failed items have ``item.error`` set instead of raising.
        """
        return self._run_batch(
            self.detect,
            images,
            api_key,
            backend=backend,
            estimate_demographics=estimate_demographics,
        )

    def stream(
        self,
        source: str | Path | int,
        *,
        api_key: str,
        every_n_frames: int = 1,
        max_frames: int | None = None,
        backend: str | None = None,
    ) -> Iterator[FrameResult]:
        """Detect objects frame by frame in a video file or a webcam.

        Args:
            source: Video file path, or an ``int`` camera index. ``http(s)``
                URLs need ``Kanon.inputs.allow_urls``.
            api_key: Caller's API key. **Each processed frame is charged as one
                detection call**, so a quota or rate error ends the stream.
            every_n_frames: Process one frame out of every N (1 = all).
            max_frames: Stop after this many processed frames (a webcam never
                ends by itself; you can also just stop iterating).
            backend: Optional explicit backend name.

        Yields:
            A :class:`FrameResult` per processed frame. Requires the optional
            ``video`` extra (``pip install 'horasis[video]'``).
        """
        is_url = isinstance(source, str) and source.startswith(("http://", "https://"))
        if is_url and not self._access.kanon.inputs.allow_urls:
            raise ValidationError(
                "URL video sources are disabled; set Kanon.inputs.allow_urls=True to opt in"
            )
        policy = self._access.kanon.inputs
        for index, stamp, rgb in iter_frames(
            source, every_n_frames=every_n_frames, max_frames=max_frames
        ):
            result = self.detect(  # type: ignore[call-arg]
                to_physis(rgb, policy), api_key=api_key, backend=backend
            )
            yield FrameResult(frame_index=index, timestamp_s=stamp, result=result)


class ImageClassifier(_Capability):
    """Public API for whole-image classification."""

    @limen(Praxis.CLASSIFY)
    def classify(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
        principal: Prosopon | None = None,
    ) -> ClassificationResult:
        """Classify `image`, returning ranked label/confidence entries."""
        result = self._organon.run(
            Praxis.CLASSIFY,
            to_physis(image, self._access.kanon.inputs),
            backend=backend,
            **(options or {}),
        )
        assert isinstance(result, ClassificationResult)
        return result

    def classify_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[ClassificationResult]]:
        """Classify each of `images`; a result or error per item, quota charged per item."""
        return self._run_batch(self.classify, images, api_key, backend=backend)


class TextExtractor(_Capability):
    """Public API for OCR / text extraction."""

    @limen(Praxis.OCR)
    def extract_text(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
        principal: Prosopon | None = None,
    ) -> OcrResult:
        """Extract text from `image` as a sequence of recognized glyphs."""
        result = self._organon.run(
            Praxis.OCR,
            to_physis(image, self._access.kanon.inputs),
            backend=backend,
            **(options or {}),
        )
        assert isinstance(result, OcrResult)
        return result

    def extract_text_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[OcrResult]]:
        """OCR each of `images`; a result or error per item, quota charged per item."""
        return self._run_batch(self.extract_text, images, api_key, backend=backend)


class FaceDetector(_Capability):
    """Public API for face detection."""

    @limen(Praxis.FACE)
    def detect_faces(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
        principal: Prosopon | None = None,
    ) -> FaceResult:
        """Detect faces in `image`."""
        result = self._organon.run(
            Praxis.FACE,
            to_physis(image, self._access.kanon.inputs),
            backend=backend,
            **(options or {}),
        )
        assert isinstance(result, FaceResult)
        return result

    def detect_faces_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[FaceResult]]:
        """Detect faces in each of `images`; a result or error per item, charged per item."""
        return self._run_batch(self.detect_faces, images, api_key, backend=backend)


class ContentModerator(_Capability):
    """Public API for content moderation."""

    @limen(Praxis.MODERATE)
    def moderate(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
        principal: Prosopon | None = None,
    ) -> ModerationResult:
        """Score `image` against moderation categories."""
        result = self._organon.run(
            Praxis.MODERATE,
            to_physis(image, self._access.kanon.inputs),
            backend=backend,
            **(options or {}),
        )
        assert isinstance(result, ModerationResult)
        return result

    def moderate_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[ModerationResult]]:
        """Moderate each of `images`; a result or error per item, charged per item."""
        return self._run_batch(self.moderate, images, api_key, backend=backend)


class ObjectSegmenter(_Capability):
    """Public API for instance segmentation (boxes plus mask outlines)."""

    @limen(Praxis.SEGMENT)
    def segment(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
        principal: Prosopon | None = None,
    ) -> SegmentationResult:
        """Segment object instances in `image`; each `Segment` has a box and a polygon."""
        result = self._organon.run(
            Praxis.SEGMENT,
            to_physis(image, self._access.kanon.inputs),
            backend=backend,
            **(options or {}),
        )
        assert isinstance(result, SegmentationResult)
        return result

    def segment_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[SegmentationResult]]:
        """Segment each of `images`; a result or error per item, charged per item."""
        return self._run_batch(self.segment, images, api_key, backend=backend)


class PoseEstimator(_Capability):
    """Public API for human pose estimation (body keypoints)."""

    @limen(Praxis.POSE)
    def estimate(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
        principal: Prosopon | None = None,
    ) -> PoseResult:
        """Estimate body poses in `image`; each `Pose` has a box and named keypoints."""
        result = self._organon.run(
            Praxis.POSE,
            to_physis(image, self._access.kanon.inputs),
            backend=backend,
            **(options or {}),
        )
        assert isinstance(result, PoseResult)
        return result

    def estimate_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[PoseResult]]:
        """Estimate poses in each of `images`; a result or error per item, charged per item."""
        return self._run_batch(self.estimate, images, api_key, backend=backend)


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
        segmentation: :class:`ObjectSegmenter`
        pose: :class:`PoseEstimator`
    """

    def __init__(
        self,
        *,
        kanon: Kanon,
        taxis: Taxis,
        validator: ApiKeyValidator,
        aisthesis: Aisthesis | None = None,
        metron: Metron | None = None,
        rate_limiter: RateLimiter | None = None,
        kairos: Kairos | None = None,
    ) -> None:
        self._kanon = kanon
        self._taxis = taxis
        self._validator = validator
        access = AccessContext.build(
            kanon=kanon,
            validator=validator,
            aisthesis=aisthesis,
            metron=metron,
            rate_limiter=rate_limiter,
        )
        self._access = access
        taxis.configure(
            device=kanon.device,
            max_loaded=kanon.max_loaded_models,
            auto_download_models=kanon.auto_download_models,
        )
        self._kairos = kairos or Kairos()
        organon = Organon(kanon=kanon, taxis=taxis, kairos=self._kairos)

        self.detection = ObjectDetector(access=access, organon=organon)
        self.classification = ImageClassifier(access=access, organon=organon)
        self.ocr = TextExtractor(access=access, organon=organon)
        self.face = FaceDetector(access=access, organon=organon)
        self.moderation = ContentModerator(access=access, organon=organon)
        self.segmentation = ObjectSegmenter(access=access, organon=organon)
        self.pose = PoseEstimator(access=access, organon=organon)

    @property
    def kairos(self) -> Kairos:
        """Pipeline hooks; attach logging/timing or your own (see `horasis.orchestration`)."""
        return self._kairos

    def warmup(self, backends: Sequence[str] | None = None) -> list[str]:
        """Load models now instead of on the first request.

        Args:
            backends: Names to warm. If omitted, every registered backend is
                tried and those whose dependencies/models are missing are skipped
                (logged); if you name a backend, its failure is raised.

        Returns:
            Names of the backends that were warmed.
        """
        warmed: list[str] = []
        for name in backends if backends is not None else self._taxis.names():
            backend = self._taxis.get(name)
            try:
                backend.warm_up()
            except BackendUnavailableError:
                if backends is not None:
                    raise
                logger.info("warmup: skipped unavailable backend %r", name)
                continue
            self._taxis.touch(backend)
            warmed.append(name)
        return warmed

    def usage(self, api_key: str) -> UsageReport:
        """Report what `api_key` has used and has left (consumes nothing).

        Raises:
            InvalidApiKeyError: (or Expired/Revoked) if the key is not valid.
        """
        principal = self._access.validator.validate(api_key)
        return self._access.usage(principal)

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._access.rate_limiter

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
        backends: Sequence[Mechane] = (),
        quota_store: QuotaStore | None = None,
        rate_limiter: RateLimiter | None = None,
        kairos: Kairos | None = None,
    ) -> Skopos:
        """Convenience constructor wiring up the default Ultralytics backend.

        Args:
            kanon: Configuration; defaults to `Kanon.default()`.
            validator: API key validator; defaults to an empty in-memory
                validator (register keys via the returned client's
                `_validator`, or supply your own production validator).
            aisthesis: Telemetry sink; defaults to a no-op.
            quota_store: Where quota counters live (default in-memory; use
                `SQLiteQuotaStore` or `RedisQuotaStore` to share/persist them).
            rate_limiter: Per-minute rate limiter; defaults to the in-process
                token bucket, or a `RedisRateLimiter` when
                ``Kanon.rate_limit_store == "redis"`` (it reuses the Redis client of
                a `RedisQuotaStore` passed as `quota_store`). Use
                `horasis.access.redis_backends()` to share one client explicitly.
            backends: Extra `Mechane` backends to register after Ultralytics
                (e.g. `EasyOCRAdapter()`, `YuNetFaceAdapter(model_path=...)`).
                Route to them with `Kanon.backend_order`, or per call with
                `backend="name"`.
        """
        kanon = kanon or Kanon.default()
        validator = validator or InMemoryApiKeyValidator()
        if rate_limiter is None and kanon.rate_limit_store == "redis":
            shared_client = getattr(quota_store, "client", None)
            if shared_client is None and not kanon.redis_url:
                raise ConfigurationError(
                    "Kanon.rate_limit_store is 'redis' but no redis_url (or Redis quota store) was given"
                )
            rate_limiter = (
                RedisRateLimiter(shared_client)
                if shared_client is not None
                else RedisRateLimiter(url=str(kanon.redis_url))
            )
        taxis = Taxis()
        taxis.register(UltralyticsAdapter())
        for backend in backends:
            taxis.register(backend)
        return cls(
            kanon=kanon,
            taxis=taxis,
            validator=validator,
            aisthesis=aisthesis,
            metron=Metron(quota_store) if quota_store else None,
            rate_limiter=rate_limiter,
            kairos=kairos,
        )
