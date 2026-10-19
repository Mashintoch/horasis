from __future__ import annotations

from abc import ABC, abstractmethod

from horasis.foundation.errors import BackendUnavailableError
from horasis.foundation.schemas import Image, Result, Task, Video


class Backend(ABC):
    """Interface every inference backend implements."""

    name: str

    supported_tasks: frozenset[Task]
    supported_video_tasks: frozenset[Task] = frozenset()
    supported_video_annotate_tasks: frozenset[Task] = frozenset()

    #: Explicit device for this backend (``auto``/``cpu``/``cuda``/``mps``), or
    #: ``None`` to follow ``Config.device`` (set via ``BackendRegistry.configure``).
    device: str | None = None
    default_device: str = "auto"
    #: Whether backends may download missing model files on first use
    #: (``Config.auto_download_models``, set via ``BackendRegistry.configure``).
    default_auto_download: bool = False

    @abstractmethod
    def infer(self, task: Task, image: Image, **options: object) -> Result:
        """Run `task` on `image` and return the matching :class:`Result` subclass.

        Raises:
            AdapterError: if inference fails.
            BackendUnavailableError: if `task` is not supported by this backend.
        """
        raise NotImplementedError

    def infer_video(self, task: Task, video: Video, **options: object) -> Result:
        """Run `task` over a video and return the matching :class:`Result` subclass.

        Optional; most backends only implement `infer`. The default raises.

        Raises:
            AdapterError: if inference fails.
            BackendUnavailableError: if this backend does not support video,
                or does not support `task` for video.
        """
        raise BackendUnavailableError(f"{type(self).__name__} does not support video inference")

    def infer_video_and_annotate(
        self, task: Task, video: Video, *, output_path: object, **options: object
    ) -> tuple[Result, object]:
        """Run video inference and write an annotated video in one decode pass.

        Optional. Calling `infer_video` and then `Result.annotate` decodes the
        video twice; a backend that implements this reuses each decoded frame
        for both. Requires the ``viz`` extra. The default raises.

        Returns:
            A ``(result, output_path)`` tuple.

        Raises:
            AdapterError: if inference fails.
            BackendUnavailableError: if this backend does not support this
                combined operation, or does not support `task` for it.
        """
        raise BackendUnavailableError(
            f"{type(self).__name__} does not support combined video detect-and-annotate"
        )

    def supports(self, task: Task) -> bool:
        return task in self.supported_tasks

    def supports_video(self, task: Task) -> bool:
        return task in self.supported_video_tasks

    def supports_video_annotate(self, task: Task) -> bool:
        return task in self.supported_video_annotate_tasks

    def effective_device(self) -> str:
        """Return this backend's own device setting, else the `Config` default."""
        return self.device or self.default_device

    def is_loaded(self) -> bool:
        """Return True if this backend holds model weights in memory."""
        return False

    def unload(self) -> None:
        """Release loaded models; they reload on next use. The default does nothing."""
        return

    def warm_up(self) -> None:
        """Load models ahead of the first call. The default does nothing."""
        return
