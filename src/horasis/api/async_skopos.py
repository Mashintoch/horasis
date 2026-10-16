"""AsyncSkopos -- the asyncio face of the client.

Backends are blocking (model inference, HTTP), so every call is run on a
thread pool and awaited; the event loop is never blocked. Access control,
quotas and hooks behave exactly as in the synchronous :class:`Skopos`, which
this class wraps.
"""

from __future__ import annotations

import asyncio
import functools
from collections.abc import AsyncIterator, Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar

from typing_extensions import Self

from horasis.access.metron import UsageReport
from horasis.api.skopos import Skopos, _describe
from horasis.foundation.imaging import ImageInput
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
)
from horasis.foundation.sphalma import InvalidApiKeyError, Sphalma

R = TypeVar("R")
_DONE = object()


class _AsyncBase:
    def __init__(self, executor: ThreadPoolExecutor) -> None:
        self._executor = executor

    async def _run(self, fn: Callable[..., R], *args: Any, **kwargs: Any) -> R:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, functools.partial(fn, *args, **kwargs))

    async def _batch(
        self,
        fn: Callable[..., Any],
        images: Sequence[ImageInput],
        api_key: str,
        **kwargs: Any,
    ) -> list[BatchItem[Any]]:
        """Run items concurrently (bounded by the pool); same semantics as the sync batch."""

        async def one(index: int, image: ImageInput) -> BatchItem[Any]:
            try:
                return BatchItem(
                    index,
                    _describe(image),
                    result=await self._run(fn, image, api_key=api_key, **kwargs),
                )
            except InvalidApiKeyError:
                raise
            except Sphalma as exc:
                return BatchItem(index, _describe(image), error=exc)

        return list(await asyncio.gather(*(one(i, img) for i, img in enumerate(images))))


class AsyncObjectDetector(_AsyncBase):
    def __init__(self, sync: Any, executor: ThreadPoolExecutor) -> None:
        super().__init__(executor)
        self._sync = sync

    async def detect(
        self,
        image: ImageInput,
        *,
        api_key: str,
        backend: str | None = None,
        estimate_demographics: bool = False,
    ) -> DetectionResult:
        """Async :meth:`ObjectDetector.detect`."""
        result: DetectionResult = await self._run(
            self._sync.detect,
            image,
            api_key=api_key,
            backend=backend,
            estimate_demographics=estimate_demographics,
        )
        return result

    async def detect_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[DetectionResult]]:
        """Async batch; items run concurrently, each with its own result or error."""
        return await self._batch(self._sync.detect, images, api_key, backend=backend)

    async def stream(
        self,
        source: str | Path | int,
        *,
        api_key: str,
        every_n_frames: int = 1,
        max_frames: int | None = None,
        backend: str | None = None,
    ) -> AsyncIterator[FrameResult]:
        """Async :meth:`ObjectDetector.stream`: ``async for frame in ...``."""
        iterator = iter(
            self._sync.stream(
                source,
                api_key=api_key,
                every_n_frames=every_n_frames,
                max_frames=max_frames,
                backend=backend,
            )
        )
        try:
            while True:
                item = await self._run(next, iterator, _DONE)
                if item is _DONE:
                    return
                yield item
        finally:
            await self._run(iterator.close)


class AsyncImageClassifier(_AsyncBase):
    def __init__(self, sync: Any, executor: ThreadPoolExecutor) -> None:
        super().__init__(executor)
        self._sync = sync

    async def classify(
        self, image: ImageInput, *, api_key: str, backend: str | None = None
    ) -> ClassificationResult:
        result: ClassificationResult = await self._run(
            self._sync.classify, image, api_key=api_key, backend=backend
        )
        return result

    async def classify_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[ClassificationResult]]:
        return await self._batch(self._sync.classify, images, api_key, backend=backend)


class AsyncTextExtractor(_AsyncBase):
    def __init__(self, sync: Any, executor: ThreadPoolExecutor) -> None:
        super().__init__(executor)
        self._sync = sync

    async def extract_text(
        self, image: ImageInput, *, api_key: str, backend: str | None = None
    ) -> OcrResult:
        result: OcrResult = await self._run(
            self._sync.extract_text, image, api_key=api_key, backend=backend
        )
        return result

    async def extract_text_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[OcrResult]]:
        return await self._batch(self._sync.extract_text, images, api_key, backend=backend)


class AsyncFaceDetector(_AsyncBase):
    def __init__(self, sync: Any, executor: ThreadPoolExecutor) -> None:
        super().__init__(executor)
        self._sync = sync

    async def detect_faces(
        self, image: ImageInput, *, api_key: str, backend: str | None = None
    ) -> FaceResult:
        result: FaceResult = await self._run(
            self._sync.detect_faces, image, api_key=api_key, backend=backend
        )
        return result

    async def detect_faces_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[FaceResult]]:
        return await self._batch(self._sync.detect_faces, images, api_key, backend=backend)


class AsyncContentModerator(_AsyncBase):
    def __init__(self, sync: Any, executor: ThreadPoolExecutor) -> None:
        super().__init__(executor)
        self._sync = sync

    async def moderate(
        self, image: ImageInput, *, api_key: str, backend: str | None = None
    ) -> ModerationResult:
        result: ModerationResult = await self._run(
            self._sync.moderate, image, api_key=api_key, backend=backend
        )
        return result

    async def moderate_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[ModerationResult]]:
        return await self._batch(self._sync.moderate, images, api_key, backend=backend)


class AsyncObjectSegmenter(_AsyncBase):
    def __init__(self, sync: Any, executor: ThreadPoolExecutor) -> None:
        super().__init__(executor)
        self._sync = sync

    async def segment(
        self, image: ImageInput, *, api_key: str, backend: str | None = None
    ) -> SegmentationResult:
        result: SegmentationResult = await self._run(
            self._sync.segment, image, api_key=api_key, backend=backend
        )
        return result

    async def segment_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[SegmentationResult]]:
        return await self._batch(self._sync.segment, images, api_key, backend=backend)


class AsyncPoseEstimator(_AsyncBase):
    def __init__(self, sync: Any, executor: ThreadPoolExecutor) -> None:
        super().__init__(executor)
        self._sync = sync

    async def estimate(
        self, image: ImageInput, *, api_key: str, backend: str | None = None
    ) -> PoseResult:
        result: PoseResult = await self._run(
            self._sync.estimate, image, api_key=api_key, backend=backend
        )
        return result

    async def estimate_batch(
        self, images: Sequence[ImageInput], *, api_key: str, backend: str | None = None
    ) -> list[BatchItem[PoseResult]]:
        return await self._batch(self._sync.estimate, images, api_key, backend=backend)


class AsyncSkopos:
    """Async client mirroring :class:`Skopos` (``await client.detection.detect(...)``).

    Args:
        client: The synchronous client to wrap.
        max_workers: Thread-pool size = how many blocking inferences may run
            at once. Defaults to Python's ``ThreadPoolExecutor`` default.

    Use as an async context manager (or call :meth:`close`) to shut the pool down::

        async with AsyncSkopos.build(validator=validator) as client:
            result = await client.detection.detect("photo.jpg", api_key=key)
    """

    def __init__(self, client: Skopos, *, max_workers: int | None = None) -> None:
        self._client = client
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="horasis")
        self.detection = AsyncObjectDetector(client.detection, self._executor)
        self.classification = AsyncImageClassifier(client.classification, self._executor)
        self.ocr = AsyncTextExtractor(client.ocr, self._executor)
        self.face = AsyncFaceDetector(client.face, self._executor)
        self.moderation = AsyncContentModerator(client.moderation, self._executor)
        self.segmentation = AsyncObjectSegmenter(client.segmentation, self._executor)
        self.pose = AsyncPoseEstimator(client.pose, self._executor)

    @classmethod
    def build(cls, *, max_workers: int | None = None, **build_kwargs: Any) -> AsyncSkopos:
        """Like :meth:`Skopos.build`; extra keyword arguments are forwarded to it."""
        return cls(Skopos.build(**build_kwargs), max_workers=max_workers)

    @property
    def sync(self) -> Skopos:
        """The wrapped synchronous client (hooks, taxis, kanon...)."""
        return self._client

    async def usage(self, api_key: str) -> UsageReport:
        report: UsageReport = await asyncio.get_running_loop().run_in_executor(
            self._executor, self._client.usage, api_key
        )
        return report

    async def warmup(self, backends: Sequence[str] | None = None) -> list[str]:
        """Async :meth:`Skopos.warmup` -- load models without blocking the loop."""
        warmed: list[str] = await asyncio.get_running_loop().run_in_executor(
            self._executor, self._client.warmup, backends
        )
        return warmed

    def close(self) -> None:
        self._executor.shutdown(wait=True)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
