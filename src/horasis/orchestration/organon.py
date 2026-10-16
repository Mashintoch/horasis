from __future__ import annotations

from collections.abc import Callable
from typing import Any

from horasis.backend.mechane import Mechane
from horasis.backend.taxis import Taxis
from horasis.foundation.kanon import Kanon
from horasis.foundation.schemas import Kinesis, ModerationResult, Physis, Praxis, Theoria
from horasis.orchestration.kairos import Kairos, Moment


class Organon:
    """Resolves a backend and runs inference for a single request.

    Every run fires the :class:`Kairos` hooks (before/after resolve, before/
    after infer, and ``ON_ERROR`` if anything raises) and tells the registry
    which backend was just used so its LRU model cap can be enforced.

    Args:
        kanon: Runtime configuration, used for `default_backend`.
        taxis: Backend registry to resolve against.
        kairos: Hook system fired around resolution and inference.
    """

    def __init__(self, *, kanon: Kanon, taxis: Taxis, kairos: Kairos | None = None) -> None:
        self._kanon = kanon
        self._taxis = taxis
        self._kairos = kairos or Kairos()

    def _routing(self, praxis: Praxis) -> dict[str, list[str]]:
        """Config-driven backend preference and fallback for `praxis`."""
        return {
            "order": self._kanon.backend_order.get(praxis.value, []),
            "fallback": self._kanon.fallback_order,
        }

    def _apply_policy(self, result: Theoria) -> Theoria:
        """Re-flag moderation results using the thresholds configured in `Kanon`.

        Whatever a backend decided, `Kanon.moderation_threshold(s)` and
        `moderation_safe_categories` are authoritative, so behaviour is the
        same across moderation backends (local or remote).
        """
        if not isinstance(result, ModerationResult):
            return result
        safe = {c.lower() for c in self._kanon.moderation_safe_categories}
        labels = [
            label.model_copy(
                update={
                    "flagged": label.category.lower() not in safe
                    and label.confidence.value
                    >= self._kanon.moderation_threshold_for(label.category)
                }
            )
            for label in result.labels
        ]
        return result.model_copy(update={"labels": labels})

    @property
    def kairos(self) -> Kairos:
        return self._kairos

    def _execute(
        self,
        praxis: Praxis,
        subject: dict[str, Any],
        resolve: Callable[..., Mechane],
        backend: str | None,
        infer: Callable[[Mechane], Any],
    ) -> Any:
        """Shared pipeline: hooks -> resolve -> hooks -> infer -> hooks (+ errors)."""
        try:
            self._kairos.fire(Moment.BEFORE_RESOLVE, praxis=praxis, **subject)
            resolved = resolve(
                praxis,
                preferred=backend,
                default=self._kanon.default_backend,
                **self._routing(praxis),
            )
            self._kairos.fire(Moment.AFTER_RESOLVE, praxis=praxis, backend=resolved)
            self._kairos.fire(Moment.BEFORE_INFER, praxis=praxis, backend=resolved, **subject)
            outcome = infer(resolved)
            self._taxis.touch(resolved)
            result = outcome[0] if isinstance(outcome, tuple) else outcome
            self._kairos.fire(Moment.AFTER_INFER, praxis=praxis, backend=resolved, result=result)
            return outcome
        except Exception as exc:
            self._kairos.fire(Moment.ON_ERROR, praxis=praxis, error=exc)
            raise

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
                follows `Kanon.backend_order`, `default_backend`, then
                `fallback_order`, then any backend that supports `praxis`.
            **options: Passed through verbatim to the resolved backend.
        """
        result: Theoria = self._execute(
            praxis,
            {"image": image},
            self._taxis.resolve,
            backend,
            lambda b: self._apply_policy(b.infer(praxis, image, **options)),
        )
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

        Same resolution and hooks as `run`, but resolves via
        `Taxis.resolve_video` and calls `Mechane.infer_video`. Hooks receive
        ``video=`` instead of ``image=``.
        """
        result: Theoria = self._execute(
            praxis,
            {"video": video},
            self._taxis.resolve_video,
            backend,
            lambda b: b.infer_video(praxis, video, **options),
        )
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

        Prefer this over `run_video` followed by `Theoria.annotate()` when
        performance matters -- a supporting backend decodes the video once.
        Raises `BackendUnavailableError` if no backend supports the combined
        operation; callers can fall back to the two-call approach.

        Returns:
            A `(result, output_path)` tuple.
        """
        outcome: tuple[Theoria, object] = self._execute(
            praxis,
            {"video": video},
            self._taxis.resolve_video_annotate,
            backend,
            lambda b: b.infer_video_and_annotate(praxis, video, output_path=output_path, **options),
        )
        return outcome
