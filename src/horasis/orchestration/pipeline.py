from __future__ import annotations

from collections.abc import Callable
from typing import Any

from horasis.backend.base import Backend
from horasis.backend.registry import BackendRegistry
from horasis.foundation.config import Config
from horasis.foundation.schemas import Image, ModerationResult, Result, Task, Video
from horasis.orchestration.hooks import HookEvent, Hooks


class Pipeline:
    """Resolves a backend and runs inference for a single request.

    Every run fires the :class:`Hooks` around resolution and inference (and
    ``ON_ERROR`` if anything raises), and tells the registry which backend was
    used so it can enforce its loaded-model cap.

    Args:
        config: Runtime configuration, used for `default_backend`.
        registry: Backend registry to resolve against.
        hooks: Hook system fired around resolution and inference.
    """

    def __init__(self, *, config: Config, registry: BackendRegistry, hooks: Hooks | None = None) -> None:
        self._config = config
        self._registry = registry
        self._hooks = hooks or Hooks()

    def _routing(self, task: Task) -> dict[str, list[str]]:
        """Return the configured backend preference and fallback for `task`."""
        return {
            "order": self._config.backend_order.get(task.value, []),
            "fallback": self._config.fallback_order,
        }

    def _apply_policy(self, result: Result) -> Result:
        """Re-flag moderation results with the thresholds in `Config`.

        `Config` overrides whatever the backend decided, so local and remote
        moderation backends behave the same.
        """
        if not isinstance(result, ModerationResult):
            return result
        safe = {c.lower() for c in self._config.moderation_safe_categories}
        labels = [
            label.model_copy(
                update={
                    "flagged": label.category.lower() not in safe
                    and label.confidence.value
                    >= self._config.moderation_threshold_for(label.category)
                }
            )
            for label in result.labels
        ]
        return result.model_copy(update={"labels": labels})

    @property
    def hooks(self) -> Hooks:
        return self._hooks

    def _execute(
        self,
        task: Task,
        subject: dict[str, Any],
        resolve: Callable[..., Backend],
        backend: str | None,
        infer: Callable[[Backend], Any],
    ) -> Any:
        """Resolve and infer, firing hooks before and after each step and on error."""
        try:
            self._hooks.fire(HookEvent.BEFORE_RESOLVE, task=task, **subject)
            resolved = resolve(
                task,
                preferred=backend,
                default=self._config.default_backend,
                **self._routing(task),
            )
            self._hooks.fire(HookEvent.AFTER_RESOLVE, task=task, backend=resolved)
            self._hooks.fire(HookEvent.BEFORE_INFER, task=task, backend=resolved, **subject)
            outcome = infer(resolved)
            self._registry.touch(resolved)
            result = outcome[0] if isinstance(outcome, tuple) else outcome
            self._hooks.fire(HookEvent.AFTER_INFER, task=task, backend=resolved, result=result)
            return outcome
        except Exception as exc:
            self._hooks.fire(HookEvent.ON_ERROR, task=task, error=exc)
            raise

    def run(
        self,
        task: Task,
        image: Image,
        *,
        backend: str | None = None,
        **options: object,
    ) -> Result:
        """Run `task` on `image` and return the result.

        Args:
            image: The input, already converted to `Image`.
            backend: Backend name to use. Otherwise resolution follows
                `Config.backend_order`, `default_backend`, `fallback_order`,
                then any backend that supports `task`.
            **options: Passed unchanged to the backend.
        """
        result: Result = self._execute(
            task,
            {"image": image},
            self._registry.resolve,
            backend,
            lambda b: self._apply_policy(b.infer(task, image, **options)),
        )
        return result

    def run_video(
        self,
        task: Task,
        video: Video,
        *,
        backend: str | None = None,
        **options: object,
    ) -> Result:
        """Run `task` on a video and return the result.

        Like `run`, but resolves with `BackendRegistry.resolve_video` and calls
        `Backend.infer_video`. Hooks receive ``video=`` instead of ``image=``.
        """
        result: Result = self._execute(
            task,
            {"video": video},
            self._registry.resolve_video,
            backend,
            lambda b: b.infer_video(task, video, **options),
        )
        return result

    def run_video_and_annotate(
        self,
        task: Task,
        video: Video,
        *,
        output_path: object,
        backend: str | None = None,
        **options: object,
    ) -> tuple[Result, object]:
        """Run `task` on a video and write an annotated copy in one decode pass.

        Returns:
            A ``(result, output_path)`` tuple.

        Raises:
            BackendUnavailableError: if no backend supports the combined
                operation. Fall back to `run_video` and `Result.annotate`.
        """
        outcome: tuple[Result, object] = self._execute(
            task,
            {"video": video},
            self._registry.resolve_video_annotate,
            backend,
            lambda b: b.infer_video_and_annotate(task, video, output_path=output_path, **options),
        )
        return outcome
