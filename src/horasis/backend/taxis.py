from __future__ import annotations

from collections.abc import Callable

from horasis.backend.mechane import Mechane
from horasis.foundation.schemas import Praxis
from horasis.foundation.sphalma import BackendUnavailableError, ConfigurationError


class Taxis:
    """Registry mapping backend names to :class:`Mechane` instances."""

    def __init__(self) -> None:
        self._backends: dict[str, Mechane] = {}

    def register(self, backend: Mechane) -> None:
        if not backend.name:
            raise ConfigurationError("Mechane.name must be a non-empty string")
        self._backends[backend.name] = backend

    def unregister(self, name: str) -> None:
        self._backends.pop(name, None)

    def get(self, name: str) -> Mechane:
        try:
            return self._backends[name]
        except KeyError as exc:
            raise BackendUnavailableError(f"No backend registered as {name!r}") from exc

    def names(self) -> list[str]:
        return list(self._backends)

    def resolve(self, praxis: Praxis, *, preferred: str | None, default: str) -> Mechane:
        """Pick the backend to use for `praxis` on image input.

        Resolution order:
            1. `preferred`, if given, must exist and support `praxis`.
            2. `default` (typically `Kanon.default_backend`), if it supports `praxis`.
            3. The first registered backend that supports `praxis`.

        Raises:
            BackendUnavailableError: if no suitable backend can be found.
        """
        return self._resolve(
            praxis,
            preferred=preferred,
            default=default,
            check=lambda b: b.supports(praxis),
            kind="",
        )

    def resolve_video(self, praxis: Praxis, *, preferred: str | None, default: str) -> Mechane:
        """Pick the backend to use for `praxis` on video (`Kinesis`) input.

        Same resolution order as `resolve`, but checks `supports_video`
        instead of `supports`.

        Raises:
            BackendUnavailableError: if no suitable backend can be found.
        """
        return self._resolve(
            praxis,
            preferred=preferred,
            default=default,
            check=lambda b: b.supports_video(praxis),
            kind="video ",
        )

    def resolve_video_annotate(
        self, praxis: Praxis, *, preferred: str | None, default: str
    ) -> Mechane:
        """Pick the backend to use for combined `praxis` video detect+annotate.

        Same resolution order as `resolve`, but checks
        `supports_video_annotate` instead of `supports`.

        Raises:
            BackendUnavailableError: if no suitable backend can be found.
        """
        return self._resolve(
            praxis,
            preferred=preferred,
            default=default,
            check=lambda b: b.supports_video_annotate(praxis),
            kind="video-annotate ",
        )

    def _resolve(
        self,
        praxis: Praxis,
        *,
        preferred: str | None,
        default: str,
        check: Callable[[Mechane], bool],
        kind: str,
    ) -> Mechane:
        if preferred is not None:
            backend = self.get(preferred)
            if not check(backend):
                raise BackendUnavailableError(
                    f"Backend {preferred!r} does not support {kind}praxis {praxis.value!r}"
                )
            return backend

        default_backend = self._backends.get(default)
        if default_backend is not None and check(default_backend):
            return default_backend

        for backend in self._backends.values():
            if check(backend):
                return backend

        raise BackendUnavailableError(
            f"No registered backend supports {kind}praxis {praxis.value!r}"
        )
