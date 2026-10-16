from __future__ import annotations

from abc import ABC, abstractmethod

from horasis.foundation.schemas import Kinesis, Physis, Praxis, Theoria
from horasis.foundation.sphalma import BackendUnavailableError


class Mechane(ABC):
    """Abstract backend. Depends only on the foundation layer."""

    name: str

    supported_praxeis: frozenset[Praxis]
    supported_video_praxeis: frozenset[Praxis] = frozenset()
    supported_video_annotate_praxeis: frozenset[Praxis] = frozenset()

    #: Explicit device for this backend (``auto``/``cpu``/``cuda``/``mps``), or
    #: ``None`` to follow ``Kanon.device`` (set via ``Taxis.configure``).
    device: str | None = None
    default_device: str = "auto"

    @abstractmethod
    def infer(self, praxis: Praxis, image: Physis, **options: object) -> Theoria:
        """Run inference and return a :class:`Theoria` subclass instance.

        Raises:
            AdapterError: if inference fails for any reason.
            BackendUnavailableError: if `praxis` is not supported by this backend.
        """
        raise NotImplementedError

    def infer_video(self, praxis: Praxis, video: Kinesis, **options: object) -> Theoria:
        """Run inference over a video and return a :class:`Theoria` subclass instance.

        Optional extension point -- most backends only implement `infer`
        for still images. The default implementation always raises.

        Raises:
            AdapterError: if inference fails for any reason.
            BackendUnavailableError: if this backend does not support video,
                or does not support `praxis` for video.
        """
        raise BackendUnavailableError(f"{type(self).__name__} does not support video inference")

    def infer_video_and_annotate(
        self, praxis: Praxis, video: Kinesis, *, output_path: object, **options: object
    ) -> tuple[Theoria, object]:
        """Run video inference and write an annotated video in a single pass.

        Optional, performance-oriented extension point: a backend that
        implements this can decode the video once and reuse the same
        decoded frames for both inference and drawing, instead of the
        caller doing `infer_video()` then `Theoria.annotate()` separately
        (which decodes the video twice). Requires the optional `viz` extra.
        The default implementation always raises.

        Returns:
            A `(result, output_path)` tuple.

        Raises:
            AdapterError: if inference fails for any reason.
            BackendUnavailableError: if this backend does not support this
                combined operation, or does not support `praxis` for it.
        """
        raise BackendUnavailableError(
            f"{type(self).__name__} does not support combined video detect-and-annotate"
        )

    def supports(self, praxis: Praxis) -> bool:
        return praxis in self.supported_praxeis

    def supports_video(self, praxis: Praxis) -> bool:
        return praxis in self.supported_video_praxeis

    def supports_video_annotate(self, praxis: Praxis) -> bool:
        return praxis in self.supported_video_annotate_praxeis

    def effective_device(self) -> str:
        """The device to use: this backend's own setting, else the Kanon default."""
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
