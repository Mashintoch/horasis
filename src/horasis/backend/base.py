from __future__ import annotations

from abc import ABC, abstractmethod

from horasis.foundation.errors import BackendUnavailableError
from horasis.foundation.schemas import Image, Result, Task, Video


class Backend(ABC):
    """Abstract backend. Depends only on the foundation layer."""

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
        """Run inference and return a :class:`Result` subclass instance.

        Raises:
            AdapterError: if inference fails for any reason.
            BackendUnavailableError: if `task` is not supported by this backend.
        """
        raise NotImplementedError

    def infer_video(self, task: Task, video: Video, **options: object) -> Result:
        """Run inference over a video and return a :class:`Result` subclass instance.

        Optional extension point -- most backends only implement `infer`
        for still images. The default implementation always raises.

        Raises:
            AdapterError: if inference fails for any reason.
            BackendUnavailableError: if this backend does not support video,
                or does not support `task` for video.
        """
        raise BackendUnavailableError(f"{type(self).__name__} does not support video inference")

    def infer_video_and_annotate(
        self, task: Task, video: Video, *, output_path: object, **options: object
    ) -> tuple[Result, object]:
        """Run video inference and write an annotated video in a single pass.

        Optional, performance-oriented extension point: a backend that
        implements this can decode the video once and reuse the same
        decoded frames for both inference and drawing, instead of the
        caller doing `infer_video()` then `Result.annotate()` separately
        (which decodes the video twice). Requires the optional `viz` extra.
        The default implementation always raises.

        Returns:
            A `(result, output_path)` tuple.

        Raises:
            AdapterError: if inference fails for any reason.
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
        """The device to use: this backend's own setting, else the Config default."""
        return self.device or self.default_device

    def is_loaded(self) -> bool:
        """Whether this backend currently holds model weights in memory."""
        return False

    def unload(self) -> None:
        """Release loaded models; they reload lazily on next use. Default: no-op."""
        return

    def warm_up(self) -> None:
        """Optional hook to preload models/weights. Default is a no-op."""
        return
