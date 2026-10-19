from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from horasis.access.guard import AccessContext, requires_access
from horasis.access.keys import ApiKeyValidator, InMemoryApiKeyValidator
from horasis.access.principal import Principal
from horasis.access.quota import QuotaManager, QuotaStore, UsageReport
from horasis.access.ratelimit import RateLimiter, RedisRateLimiter
from horasis.backend.base import Backend
from horasis.backend.registry import BackendRegistry
from horasis.backend.ultralytics_adapter import UltralyticsAdapter
from horasis.foundation.config import Config
from horasis.foundation.errors import (
    BackendUnavailableError,
    ConfigurationError,
    HorasisError,
    InvalidApiKeyError,
    ValidationError,
)
from horasis.foundation.frames import iter_frames
from horasis.foundation.imaging import ImageInput, to_image
from horasis.foundation.schemas import (
    BatchItem,
    ClassificationResult,
    DetectionResult,
    FaceResult,
    FrameResult,
    ModerationResult,
    OcrResult,
    PoseResult,
    SegmentationResult,
    Task,
    Video,
    VideoDetectionResult,
)
from horasis.foundation.telemetry import Telemetry
from horasis.orchestration.hooks import Hooks
from horasis.orchestration.pipeline import Pipeline

logger = logging.getLogger(__name__)


def _to_video(video: str | Path | bytes | Video) -> Video:
    """Convert any accepted video input into a `Video`."""
    if isinstance(video, Video):
        return video
    if isinstance(video, bytes):
        return Video.from_bytes(video)
    if isinstance(video, str):
        return Video.from_source(video)
    if isinstance(video, Path):
        return Video.from_path(video)
    raise ValidationError(f"unsupported video input type: {type(video).__name__}")


def _describe(image: object) -> str:
    """Return a short label for an input, never its pixel data."""
    if isinstance(image, str | Path):
        return str(image)
    return getattr(image, "source", None) or f"<{type(image).__name__}>"


class _Capability:
    """Base class for the capability classes."""

    def __init__(self, *, access: AccessContext, pipeline: Pipeline) -> None:
        self._access = access
        self._pipeline = pipeline

    def _run_batch(
        self,
        fn: Callable[..., Any],
        images: Sequence[ImageInput],
        api_key: str,
        **kwargs: Any,
    ) -> list[BatchItem[Any]]:
        """Run `fn` over `images`; each item succeeds or fails on its own.

        Quota and rate limits are charged per item, after the item passes input
        validation, so invalid items cost nothing. An invalid API key aborts the
        batch, since every item would fail the same way. Any other `HorasisError`
        is recorded on its item and the batch continues.
        """
        items: list[BatchItem[Any]] = []
        for index, image in enumerate(images):
            try:
                items.append(
                    BatchItem(index, _describe(image), result=fn(image, api_key=api_key, **kwargs))
                )
            except InvalidApiKeyError:
                raise
            except HorasisError as exc:
                items.append(BatchItem(index, _describe(image), error=exc))
        return items


class ObjectDetector(_Capability):
    """Object detection."""

    @requires_access(Task.DETECT)
    def detect(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        estimate_demographics: bool = False,
        options: Mapping[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> DetectionResult:
        """Detect objects in `image`.

        Args:
            image: A file path, raw bytes or an `Image`.
            backend: Backend name to use instead of the configured routing.
            estimate_demographics: Attach a `face_estimate` (model-estimated age
                and gender) to every ``"person"`` detection. Needs the
                ``demographics`` extra. The estimates have real error rates and
                are not facts about anyone.
        """
        result = self._pipeline.run(
            Task.DETECT,
            to_image(image, self._access.config.inputs),
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
        """Detect objects in `image` and return the result with an annotated copy.

        Same as `detect` followed by `result.annotate`, except `image` is loaded
        once, so a URL is downloaded only once.

        Args:
            image: A file path, URL, raw bytes or an `Image`.
            backend: Backend name to use for detection.
            output_path: Also save the annotated image here.
            **viz_options: Drawing options for `horasis.viz.render_detections`:
                ``box_color`` (RGB tuple; default is a fixed colour per label),
                ``text_color`` (RGB tuple, default white), ``line_width``
                (pixels, default 3) and ``font_size`` (points, default 16).

        Returns:
            ``(result, annotated_image)``, where ``annotated_image`` is a
            ``PIL.Image.Image``.

        Raises:
            VisualizationUnavailableError: if Pillow is missing.
        """
        img = to_image(image, self._access.config.inputs)
        # requires_access consumes api_key, which detect's own signature does not declare
        result = self.detect(  # type: ignore[call-arg]
            img, api_key=api_key, backend=backend, estimate_demographics=estimate_demographics
        )
        annotated = result.annotate(
            img,
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
        """Run `detect_and_annotate` on each of `images`, in order.

        Each image costs one detection call. An error stops the batch and
        propagates; results collected so far are lost.

        Args:
            images: Any mix of file paths, URLs, raw bytes or `Image`.
            output_dir: Directory for the annotated images, named from
                `filenames` or else ``annotated_0.jpg``, ``annotated_1.jpg``, ...
            filenames: File names inside `output_dir`, one per image. Ignored
                when `output_paths` is given.
            output_paths: Full paths, one per image. Overrides `output_dir` and
                `filenames`.
            **viz_options: Drawing options for every image; see
                `detect_and_annotate`.

        Returns:
            One ``(result, annotated_image)`` pair per image.
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

    @requires_access(Task.DETECT, validate_image=False)
    def detect_video(
        self,
        video: str | Path | bytes | Video,
        *,
        backend: str | None = None,
        sample_every_n_frames: int = 1,
        estimate_demographics: bool = False,
        principal: Principal | None = None,
    ) -> VideoDetectionResult:
        """Detect objects in a video, frame by frame.

        One call costs one detection unit, however many frames the video has.

        Args:
            video: A local path, URL, raw bytes or a `Video`. URLs are streamed
                by the backend, not downloaded first.
            backend: Backend name to use instead of the configured routing.
            sample_every_n_frames: Run inference on every Nth frame only.
            estimate_demographics: As in `detect`, for the sampled frames only.
                Adds a per-person cost on top of detection.

        Returns:
            A `VideoDetectionResult` with one `FrameDetections` per processed
            frame.

        Raises:
            BackendUnavailableError: if the backend does not support video for
                this task. The Ultralytics backend supports it for detection only.
        """
        vid = _to_video(video)
        result = self._pipeline.run_video(
            Task.DETECT,
            vid,
            backend=backend,
            sample_every_n_frames=sample_every_n_frames,
            estimate_demographics=estimate_demographics,
        )
        assert isinstance(result, VideoDetectionResult)
        return result

    @requires_access(Task.DETECT, validate_image=False)
    def detect_video_and_annotate(
        self,
        video: str | Path | bytes | Video,
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
        principal: Principal | None = None,
    ) -> tuple[VideoDetectionResult, Path]:
        """Detect objects in a video and write an annotated copy in one decode pass.

        `detect_video` followed by `VideoDetectionResult.annotate` decodes (or,
        for a URL, downloads) the video twice; this reuses each decoded frame.
        Costs one detection unit, like `detect_video`.

        Args:
            video: A local path, URL, raw bytes or a `Video`.
            output_path: Where to write the annotated video, e.g. ``"out.mp4"``.
            backend: Backend name to use instead of the configured routing.
            sample_every_n_frames: Run inference on every Nth frame only.
            box_color: Fixed RGB colour for every box; default is one per label.
            text_color: RGB colour for the caption text.
            line_width: Box border thickness in pixels.
            font_size: Caption font size in points.
            estimate_demographics: As in `detect`. The estimates are returned in
                the result but not drawn.

        Returns:
            ``(result, output_path)``.

        Raises:
            VisualizationUnavailableError: if Pillow or OpenCV is missing
                (``pip install 'horasis[viz]'``).
            BackendUnavailableError: if the backend does not support the combined
                operation for this task.
        """
        vid = _to_video(video)
        result, out_path = self._pipeline.run_video_and_annotate(
            Task.DETECT,
            vid,
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
        """Detect objects in each of `images`, with a result or an error per item.

        Quota and rate limits are charged per item. Failed items have
        ``item.error`` set instead of raising.
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
        """Detect objects frame by frame in a video file or from a camera.

        Needs the ``video`` extra.

        Args:
            source: A video file path or an ``int`` camera index. http(s) URLs
                need ``Config.inputs.allow_urls``.
            api_key: Each processed frame is charged as one detection call, so a
                quota or rate error ends the stream.
            every_n_frames: Process one frame out of every N.
            max_frames: Stop after this many processed frames. A camera never
                ends by itself.
            backend: Backend name to use instead of the configured routing.

        Yields:
            A :class:`FrameResult` per processed frame.
        """
        is_url = isinstance(source, str) and source.startswith(("http://", "https://"))
        if is_url and not self._access.config.inputs.allow_urls:
            raise ValidationError(
                "URL video sources are disabled; set Config.inputs.allow_urls=True"
            )
        policy = self._access.config.inputs
        for index, stamp, rgb in iter_frames(
            source, every_n_frames=every_n_frames, max_frames=max_frames
        ):
            result = self.detect(  # type: ignore[call-arg]
                to_image(rgb, policy), api_key=api_key, backend=backend
            )
            yield FrameResult(frame_index=index, timestamp_s=stamp, result=result)


class ImageClassifier(_Capability):
    """Whole-image classification."""

    @requires_access(Task.CLASSIFY)
    def classify(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> ClassificationResult:
        """Classify `image` into ranked label and confidence entries."""
        result = self._pipeline.run(
            Task.CLASSIFY,
            to_image(image, self._access.config.inputs),
            backend=backend,
            **(options or {}),
        )
        assert isinstance(result, ClassificationResult)
        return result

    def classify_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[ClassificationResult]]:
        """Classify each of `images`, with a result or an error per item."""
        return self._run_batch(self.classify, images, api_key, backend=backend)


class TextExtractor(_Capability):
    """OCR."""

    @requires_access(Task.OCR)
    def extract_text(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> OcrResult:
        """Extract the text regions in `image`."""
        result = self._pipeline.run(
            Task.OCR,
            to_image(image, self._access.config.inputs),
            backend=backend,
            **(options or {}),
        )
        assert isinstance(result, OcrResult)
        return result

    def extract_text_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[OcrResult]]:
        """Run OCR on each of `images`, with a result or an error per item."""
        return self._run_batch(self.extract_text, images, api_key, backend=backend)


class FaceDetector(_Capability):
    """Face detection."""

    @requires_access(Task.FACE)
    def detect_faces(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> FaceResult:
        """Detect faces in `image`."""
        result = self._pipeline.run(
            Task.FACE,
            to_image(image, self._access.config.inputs),
            backend=backend,
            **(options or {}),
        )
        assert isinstance(result, FaceResult)
        return result

    def detect_faces_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[FaceResult]]:
        """Detect faces in each of `images`, with a result or an error per item."""
        return self._run_batch(self.detect_faces, images, api_key, backend=backend)


class ContentModerator(_Capability):
    """Content moderation."""

    @requires_access(Task.MODERATE)
    def moderate(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> ModerationResult:
        """Score `image` against moderation categories."""
        result = self._pipeline.run(
            Task.MODERATE,
            to_image(image, self._access.config.inputs),
            backend=backend,
            **(options or {}),
        )
        assert isinstance(result, ModerationResult)
        return result

    def moderate_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[ModerationResult]]:
        """Moderate each of `images`, with a result or an error per item."""
        return self._run_batch(self.moderate, images, api_key, backend=backend)


class ObjectSegmenter(_Capability):
    """Instance segmentation (boxes and mask outlines)."""

    @requires_access(Task.SEGMENT)
    def segment(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> SegmentationResult:
        """Segment object instances in `image`; each `Segment` has a box and a polygon."""
        result = self._pipeline.run(
            Task.SEGMENT,
            to_image(image, self._access.config.inputs),
            backend=backend,
            **(options or {}),
        )
        assert isinstance(result, SegmentationResult)
        return result

    def segment_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[SegmentationResult]]:
        """Segment each of `images`, with a result or an error per item."""
        return self._run_batch(self.segment, images, api_key, backend=backend)


class PoseEstimator(_Capability):
    """Human pose estimation (body keypoints)."""

    @requires_access(Task.POSE)
    def estimate(
        self,
        image: ImageInput,
        *,
        backend: str | None = None,
        options: Mapping[str, Any] | None = None,
        principal: Principal | None = None,
    ) -> PoseResult:
        """Estimate body poses in `image`; each `Pose` has a box and named keypoints."""
        result = self._pipeline.run(
            Task.POSE,
            to_image(image, self._access.config.inputs),
            backend=backend,
            **(options or {}),
        )
        assert isinstance(result, PoseResult)
        return result

    def estimate_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[PoseResult]]:
        """Estimate poses in each of `images`, with a result or an error per item."""
        return self._run_batch(self.estimate, images, api_key, backend=backend)


class Client:
    """The horasis client. Build one and call its capability attributes.

    Example:
        >>> validator = InMemoryApiKeyValidator()
        >>> validator.register_simple("demo-key", principal_id="alice", tier=Tier.FREE)
        >>> client = Client.build(validator=validator)
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
        config: Config,
        registry: BackendRegistry,
        validator: ApiKeyValidator,
        telemetry: Telemetry | None = None,
        quota_manager: QuotaManager | None = None,
        rate_limiter: RateLimiter | None = None,
        hooks: Hooks | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._validator = validator
        access = AccessContext.build(
            config=config,
            validator=validator,
            telemetry=telemetry,
            quota_manager=quota_manager,
            rate_limiter=rate_limiter,
        )
        self._access = access
        registry.configure(
            device=config.device,
            max_loaded=config.max_loaded_models,
            auto_download_models=config.auto_download_models,
        )
        self._hooks = hooks or Hooks()
        pipeline = Pipeline(config=config, registry=registry, hooks=self._hooks)

        self.detection = ObjectDetector(access=access, pipeline=pipeline)
        self.classification = ImageClassifier(access=access, pipeline=pipeline)
        self.ocr = TextExtractor(access=access, pipeline=pipeline)
        self.face = FaceDetector(access=access, pipeline=pipeline)
        self.moderation = ContentModerator(access=access, pipeline=pipeline)
        self.segmentation = ObjectSegmenter(access=access, pipeline=pipeline)
        self.pose = PoseEstimator(access=access, pipeline=pipeline)

    @property
    def hooks(self) -> Hooks:
        """Pipeline hooks; see `horasis.orchestration` for the built-in ones."""
        return self._hooks

    def warmup(self, backends: Sequence[str] | None = None) -> list[str]:
        """Load models now instead of on the first request.

        Args:
            backends: Names to warm. If omitted, every registered backend is
                tried and those with missing dependencies or models are skipped
                and logged. A named backend that fails raises.

        Returns:
            Names of the backends warmed.
        """
        warmed: list[str] = []
        for name in backends if backends is not None else self._registry.names():
            backend = self._registry.get(name)
            try:
                backend.warm_up()
            except BackendUnavailableError:
                if backends is not None:
                    raise
                logger.info("warmup: skipped unavailable backend %r", name)
                continue
            self._registry.touch(backend)
            warmed.append(name)
        return warmed

    def usage(self, api_key: str) -> UsageReport:
        """Report what `api_key` has used and has left, without charging anything.

        Raises:
            InvalidApiKeyError: or a subclass, if the key is not valid.
        """
        principal = self._access.validator.validate(api_key)
        return self._access.usage(principal)

    @property
    def rate_limiter(self) -> RateLimiter:
        return self._access.rate_limiter

    @property
    def config(self) -> Config:
        return self._config

    @property
    def registry(self) -> BackendRegistry:
        return self._registry

    @classmethod
    def build(
        cls,
        *,
        config: Config | None = None,
        validator: ApiKeyValidator | None = None,
        telemetry: Telemetry | None = None,
        backends: Sequence[Backend] = (),
        quota_store: QuotaStore | None = None,
        rate_limiter: RateLimiter | None = None,
        hooks: Hooks | None = None,
    ) -> Client:
        """Build a client with the default Ultralytics backend registered.

        Args:
            config: Defaults to `Config.default()`.
            validator: Defaults to an empty `InMemoryApiKeyValidator`.
            telemetry: Defaults to a no-op sink.
            quota_store: Defaults to in-memory. Use `SQLiteQuotaStore` or
                `RedisQuotaStore` to share or persist counters.
            rate_limiter: Defaults to the in-process token bucket, or to a
                `RedisRateLimiter` when ``Config.rate_limit_store == "redis"``
                (reusing the client of a `RedisQuotaStore` passed as
                `quota_store`). `horasis.access.redis_backends()` builds both
                on one client.
            backends: Extra backends to register after Ultralytics. Route to them
                with `Config.backend_order` or per call with ``backend="name"``.
        """
        config = config or Config.default()
        validator = validator or InMemoryApiKeyValidator()
        if rate_limiter is None and config.rate_limit_store == "redis":
            shared_client = getattr(quota_store, "client", None)
            if shared_client is None and not config.redis_url:
                raise ConfigurationError(
                    "Config.rate_limit_store is 'redis' but no redis_url or Redis quota store was given"
                )
            rate_limiter = (
                RedisRateLimiter(shared_client)
                if shared_client is not None
                else RedisRateLimiter(url=str(config.redis_url))
            )
        registry = BackendRegistry()
        registry.register(UltralyticsAdapter())
        for backend in backends:
            registry.register(backend)
        return cls(
            config=config,
            registry=registry,
            validator=validator,
            telemetry=telemetry,
            quota_manager=QuotaManager(quota_store) if quota_store else None,
            rate_limiter=rate_limiter,
            hooks=hooks,
        )
