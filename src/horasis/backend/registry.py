from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence

from horasis.backend.base import Backend
from horasis.foundation.errors import (
    BackendUnavailableError,
    ConfigurationError,
    NoBackendError,
)
from horasis.foundation.schemas import Task


class BackendRegistry:
    """Registry mapping backend names to :class:`Backend` instances."""

    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}
        self._lock = threading.Lock()
        self._recent: OrderedDict[str, None] = OrderedDict()
        self._device = "auto"
        self._auto_download = False
        self._max_loaded: int | None = None

    def register(self, backend: Backend) -> None:
        if not backend.name:
            raise ConfigurationError("Backend.name must be a non-empty string")
        backend.default_device = self._device
        backend.default_auto_download = self._auto_download
        self._backends[backend.name] = backend

    def unregister(self, name: str) -> None:
        self._backends.pop(name, None)
        self._recent.pop(name, None)

    def configure(
        self,
        *,
        device: str = "auto",
        max_loaded: int | None = None,
        auto_download_models: bool = False,
    ) -> None:
        """Apply `Config.device` / `max_loaded_models` / `auto_download_models` to all backends."""
        self._device = device
        self._max_loaded = max_loaded
        self._auto_download = auto_download_models
        for backend in self._backends.values():
            backend.default_device = device
            backend.default_auto_download = auto_download_models

    def touch(self, backend: Backend) -> None:
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

    def get(self, name: str) -> Backend:
        try:
            return self._backends[name]
        except KeyError as exc:
            raise BackendUnavailableError(f"No backend registered as {name!r}") from exc

    def names(self) -> list[str]:
        return list(self._backends)

    def resolve(
        self,
        task: Task,
        *,
        preferred: str | None,
        default: str,
        order: Sequence[str] = (),
        fallback: Sequence[str] = (),
    ) -> Backend:
        """Pick the backend to use for `task` on image input.

        Resolution order:
            1. `preferred`, if given, must exist and support `task`.
            2. `order` (typically `Config.backend_order[task]`): the first
               registered backend listed there that supports `task`.
            3. `default` (typically `Config.default_backend`), if it supports `task`.
            4. `fallback` (typically `Config.fallback_order`), first match.
            5. The first registered backend that supports `task`.

        Raises:
            BackendUnavailableError: if no suitable backend can be found.
        """
        return self._resolve(
            task,
            preferred=preferred,
            default=default,
            order=order,
            fallback=fallback,
            check=lambda b: b.supports(task),
            kind="",
        )

    def resolve_video(
        self,
        task: Task,
        *,
        preferred: str | None,
        default: str,
        order: Sequence[str] = (),
        fallback: Sequence[str] = (),
    ) -> Backend:
        """Pick the backend to use for `task` on video (`Video`) input.

        Same resolution order as `resolve`, but checks `supports_video`
        instead of `supports`.

        Raises:
            BackendUnavailableError: if no suitable backend can be found.
        """
        return self._resolve(
            task,
            preferred=preferred,
            default=default,
            order=order,
            fallback=fallback,
            check=lambda b: b.supports_video(task),
            kind="video ",
        )

    def resolve_video_annotate(
        self,
        task: Task,
        *,
        preferred: str | None,
        default: str,
        order: Sequence[str] = (),
        fallback: Sequence[str] = (),
    ) -> Backend:
        """Pick the backend to use for combined `task` video detect+annotate.

        Same resolution order as `resolve`, but checks
        `supports_video_annotate` instead of `supports`.

        Raises:
            BackendUnavailableError: if no suitable backend can be found.
        """
        return self._resolve(
            task,
            preferred=preferred,
            default=default,
            order=order,
            fallback=fallback,
            check=lambda b: b.supports_video_annotate(task),
            kind="video-annotate ",
        )

    def _resolve(
        self,
        task: Task,
        *,
        preferred: str | None,
        default: str,
        order: Sequence[str],
        fallback: Sequence[str],
        check: Callable[[Backend], bool],
        kind: str,
    ) -> Backend:
        if preferred is not None:
            backend = self.get(preferred)
            if not check(backend):
                raise BackendUnavailableError(
                    f"Backend {preferred!r} does not support {kind}task {task.value!r}"
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
            f"No registered backend supports {kind}task {task.value!r}. "
            f"Registered backends: {registered or 'none'}. "
            "Register one that supports it, or install the matching extra "
            "(e.g. horasis[ocr], horasis[moderation]).",
            context={"task": task.value, "registered": registered, "considered": considered},
        )
