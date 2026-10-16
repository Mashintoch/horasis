from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence

from horasis.backend.mechane import Mechane
from horasis.foundation.schemas import Praxis
from horasis.foundation.sphalma import (
    BackendUnavailableError,
    ConfigurationError,
    NoBackendError,
)


class Taxis:
    """Registry mapping backend names to :class:`Mechane` instances."""

    def __init__(self) -> None:
        self._backends: dict[str, Mechane] = {}
        self._lock = threading.Lock()
        self._recent: OrderedDict[str, None] = OrderedDict()
        self._device = "auto"
        self._max_loaded: int | None = None

    def register(self, backend: Mechane) -> None:
        if not backend.name:
            raise ConfigurationError("Mechane.name must be a non-empty string")
        backend.default_device = self._device
        self._backends[backend.name] = backend

    def unregister(self, name: str) -> None:
        self._backends.pop(name, None)
        self._recent.pop(name, None)

    def configure(self, *, device: str = "auto", max_loaded: int | None = None) -> None:
        """Apply `Kanon.device` / `max_loaded_models` to current and future backends."""
        self._device = device
        self._max_loaded = max_loaded
        for backend in self._backends.values():
            backend.default_device = device

    def touch(self, backend: Mechane) -> None:
        """Mark `backend` most recently used and unload LRU backends over the cap."""
        with self._lock:
            self._recent[backend.name] = None
            self._recent.move_to_end(backend.name)
            cap = self._max_loaded
            if cap is None:
                return
            loaded = [n for n in self._recent if n in self._backends and self._backends[n].is_loaded()]
            for name in loaded:
                if len(loaded) <= cap:
                    break
                if name == backend.name:
                    continue
                self._backends[name].unload()
                loaded.remove(name)

    def get(self, name: str) -> Mechane:
        try:
            return self._backends[name]
        except KeyError as exc:
            raise BackendUnavailableError(f"No backend registered as {name!r}") from exc

    def names(self) -> list[str]:
        return list(self._backends)

    def resolve(
        self,
        praxis: Praxis,
        *,
        preferred: str | None,
        default: str,
        order: Sequence[str] = (),
        fallback: Sequence[str] = (),
    ) -> Mechane:
        """Pick the backend to use for `praxis` on image input.

        Resolution order:
            1. `preferred`, if given, must exist and support `praxis`.
            2. `order` (typically `Kanon.backend_order[praxis]`): the first
               registered backend listed there that supports `praxis`.
            3. `default` (typically `Kanon.default_backend`), if it supports `praxis`.
            4. `fallback` (typically `Kanon.fallback_order`), first match.
            5. The first registered backend that supports `praxis`.

        Raises:
            BackendUnavailableError: if no suitable backend can be found.
        """
        return self._resolve(
            praxis,
            preferred=preferred,
            default=default,
            order=order,
            fallback=fallback,
            check=lambda b: b.supports(praxis),
            kind="",
        )

    def resolve_video(
        self,
        praxis: Praxis,
        *,
        preferred: str | None,
        default: str,
        order: Sequence[str] = (),
        fallback: Sequence[str] = (),
    ) -> Mechane:
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
            order=order,
            fallback=fallback,
            check=lambda b: b.supports_video(praxis),
            kind="video ",
        )

    def resolve_video_annotate(
        self,
        praxis: Praxis,
        *,
        preferred: str | None,
        default: str,
        order: Sequence[str] = (),
        fallback: Sequence[str] = (),
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
            order=order,
            fallback=fallback,
            check=lambda b: b.supports_video_annotate(praxis),
            kind="video-annotate ",
        )

    def _resolve(
        self,
        praxis: Praxis,
        *,
        preferred: str | None,
        default: str,
        order: Sequence[str],
        fallback: Sequence[str],
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

        # Config preference, then the default, then the fallback list, then anything.
        candidates = [*order, default, *fallback, *self._backends]
        considered: list[str] = []
        for name in dict.fromkeys(candidates):
            backend_or_none = self._backends.get(name)
            if backend_or_none is None:
                continue
            considered.append(name)
            if check(backend_or_none):
                return backend_or_none

        registered = self.names()
        raise NoBackendError(
            f"No registered backend supports {kind}praxis {praxis.value!r}. "
            f"Registered backends: {registered or 'none'}. "
            "Register one that supports it, or install the matching extra "
            "(e.g. horasis[ocr], horasis[moderation]).",
            context={"praxis": praxis.value, "registered": registered, "considered": considered},
        )
