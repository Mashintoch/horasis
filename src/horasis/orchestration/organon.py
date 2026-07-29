from __future__ import annotations

from horasis.backend.taxis import Taxis
from horasis.foundation.kanon import Kanon
from horasis.foundation.schemas import Kinesis, Physis, Praxis, Theoria
from horasis.orchestration.kairos import Kairos, Moment


class Organon:
    """Resolves a backend and runs inference for a single request.

    Args:
        kanon: Runtime configuration, used for `default_backend`.
        taxis: Backend registry to resolve against.
        kairos: Hook system fired around resolution and inference.
    """

    def __init__(self, *, kanon: Kanon, taxis: Taxis, kairos: Kairos | None = None) -> None:
        self._kanon = kanon
        self._taxis = taxis
        self._kairos = kairos or Kairos()

    @property
    def kairos(self) -> Kairos:
        return self._kairos

    def run(
        self,
        praxis: Praxis,
        image: Physis,
        *,
        backend: str | None = None,
        **options: object,
    ) -> Theoria:
        """Execute `praxis` against `image`, returning a `Theoria` result.

        Args:
            praxis: The task to perform.
            image: The input image, already normalized to `Physis`.
            backend: Optional explicit backend name; otherwise resolution
                falls back to `Kanon.default_backend` and then to the first
                registered backend that supports `praxis`.
            **options: Passed through verbatim to the resolved backend.
        """
        self._kairos.before_resolve(praxis=praxis, image=image)
        resolved = self._taxis.resolve(
            praxis, preferred=backend, default=self._kanon.default_backend
        )
        self._kairos.after_resolve(praxis=praxis, backend=resolved)

        self._kairos.before_infer(praxis=praxis, backend=resolved, image=image)
        result = resolved.infer(praxis, image, **options)
        self._kairos.after_infer(praxis=praxis, backend=resolved, result=result)
        return result

    def run_video(
        self,
        praxis: Praxis,
        video: Kinesis,
        *,
        backend: str | None = None,
        **options: object,
    ) -> Theoria:
        """Execute `praxis` against a video, returning a `Theoria` result.

        Same resolution and hook-firing shape as `run`, but resolves via
        `Taxis.resolve_video` and calls `Mechane.infer_video`. Kairos hooks
        fire with `video=` in the payload instead of `image=`.

        Args:
            praxis: The task to perform.
            video: The input video, already normalized to `Kinesis`.
            backend: Optional explicit backend name; otherwise resolution
                falls back to `Kanon.default_backend` and then to the first
                registered backend that supports `praxis` for video.
            **options: Passed through verbatim to the resolved backend.
        """
        self._kairos.fire(Moment.BEFORE_RESOLVE, praxis=praxis, video=video)
        resolved = self._taxis.resolve_video(
            praxis, preferred=backend, default=self._kanon.default_backend
        )
        self._kairos.fire(Moment.AFTER_RESOLVE, praxis=praxis, backend=resolved)

        self._kairos.fire(Moment.BEFORE_INFER, praxis=praxis, backend=resolved, video=video)
        result = resolved.infer_video(praxis, video, **options)
        self._kairos.fire(Moment.AFTER_INFER, praxis=praxis, backend=resolved, result=result)
        return result

    def run_video_and_annotate(
        self,
        praxis: Praxis,
        video: Kinesis,
        *,
        output_path: object,
        backend: str | None = None,
        **options: object,
    ) -> tuple[Theoria, object]:
        """Execute `praxis` against a video and write an annotated copy, in one pass.

        Prefer this over calling `run_video` followed by `Theoria.annotate()`
        separately when performance matters -- a backend that supports it
        decodes the video only once. Falls back to `BackendUnavailableError`
        if no registered backend supports the combined operation for
        `praxis`; callers can catch that and fall back to the two-call
        approach with a different backend if needed.

        Args:
            praxis: The task to perform.
            video: The input video, already normalized to `Kinesis`.
            output_path: Where to write the annotated video.
            backend: Optional explicit backend name; otherwise resolution
                falls back to `Kanon.default_backend` and then to the first
                registered backend that supports the combined operation.
            **options: Passed through verbatim to the resolved backend
                (e.g. `sample_every_n_frames`, `show_labels`, `box_color`).

        Returns:
            A `(result, output_path)` tuple.
        """
        self._kairos.fire(Moment.BEFORE_RESOLVE, praxis=praxis, video=video)
        resolved = self._taxis.resolve_video_annotate(
            praxis, preferred=backend, default=self._kanon.default_backend
        )
        self._kairos.fire(Moment.AFTER_RESOLVE, praxis=praxis, backend=resolved)

        self._kairos.fire(Moment.BEFORE_INFER, praxis=praxis, backend=resolved, video=video)
        result, out_path = resolved.infer_video_and_annotate(
            praxis, video, output_path=output_path, **options
        )
        self._kairos.fire(Moment.AFTER_INFER, praxis=praxis, backend=resolved, result=result)
        return result, out_path
