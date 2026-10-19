"""Built-in :class:`~horasis.orchestration.hooks.Hooks`."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from horasis.orchestration.hooks import HookEvent, Hooks


class LoggingHook:
    """Logs each pipeline event to a :mod:`logging` logger.

    Args:
        logger: Logger to use; defaults to ``horasis.pipeline``.
        level: Level for normal events (errors always log at WARNING).

    Usage::

        LoggingHook().attach(client.hooks)
    """

    def __init__(self, logger: logging.Logger | None = None, level: int = logging.DEBUG) -> None:
        self._log = logger or logging.getLogger("horasis.pipeline")
        self._level = level

    def attach(self, hooks: Hooks) -> LoggingHook:
        for event in HookEvent:
            hooks.on(event, self._handler(event))
        return self

    def _handler(self, event: HookEvent) -> Any:
        def hook(**payload: Any) -> None:
            task = getattr(payload.get("task"), "value", payload.get("task"))
            backend = getattr(payload.get("backend"), "name", None)
            if event is HookEvent.ON_ERROR:
                self._log.warning("%s task=%s error=%r", event.value, task, payload.get("error"))
            elif event is HookEvent.AFTER_INFER:
                self._log.log(
                    self._level,
                    "%s task=%s backend=%s latency_ms=%.1f",
                    event.value,
                    task,
                    backend,
                    getattr(payload.get("result"), "latency_ms", float("nan")),
                )
            else:
                self._log.log(self._level, "%s task=%s backend=%s", event.value, task, backend)

        return hook


@dataclass(frozen=True)
class Timing:
    """One measured inference: wall time from ``BEFORE_INFER`` to ``AFTER_INFER``."""

    task: str
    backend: str
    seconds: float


class TimingHook:
    """Measures pipeline inference time (including backend loading on first use).

    Complements ``Result.latency_ms`` (which the backend reports itself) with
    an outside-in measurement. Thread-safe. Records are kept in memory;
    call :meth:`clear` periodically in long-running processes.
    """

    def __init__(self) -> None:
        self.records: list[Timing] = []
        self._lock = threading.Lock()
        self._local = threading.local()

    def attach(self, hooks: Hooks) -> TimingHook:
        hooks.on(HookEvent.BEFORE_INFER, self._before)
        hooks.on(HookEvent.AFTER_INFER, self._after)
        hooks.on(HookEvent.ON_ERROR, self._abandon)
        return self

    def _before(self, **_: Any) -> None:
        self._local.start = time.perf_counter()

    def _after(self, **payload: Any) -> None:
        start = getattr(self._local, "start", None)
        if start is None:
            return
        self._local.start = None
        timing = Timing(
            task=str(getattr(payload.get("task"), "value", payload.get("task"))),
            backend=str(getattr(payload.get("backend"), "name", "?")),
            seconds=time.perf_counter() - start,
        )
        with self._lock:
            self.records.append(timing)

    def _abandon(self, **_: Any) -> None:
        self._local.start = None

    def clear(self) -> None:
        with self._lock:
            self.records.clear()

    def summary(self) -> dict[str, dict[str, float]]:
        """Per-task ``{"count", "mean_s", "max_s"}``."""
        with self._lock:
            records = list(self.records)
        out: dict[str, dict[str, float]] = {}
        for task in sorted({r.task for r in records}):
            secs = [r.seconds for r in records if r.task == task]
            out[task] = {
                "count": float(len(secs)),
                "mean_s": sum(secs) / len(secs),
                "max_s": max(secs),
            }
        return out
