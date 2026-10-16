"""Built-in :class:`~horasis.orchestration.kairos.Kairos` hooks."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

from horasis.orchestration.kairos import Kairos, Moment


class LoggingHook:
    """Logs each pipeline moment to a :mod:`logging` logger.

    Args:
        logger: Logger to use; defaults to ``horasis.pipeline``.
        level: Level for normal moments (errors always log at WARNING).

    Usage::

        LoggingHook().attach(client.kairos)
    """

    def __init__(self, logger: logging.Logger | None = None, level: int = logging.DEBUG) -> None:
        self._log = logger or logging.getLogger("horasis.pipeline")
        self._level = level

    def attach(self, kairos: Kairos) -> LoggingHook:
        for moment in Moment:
            kairos.on(moment, self._handler(moment))
        return self

    def _handler(self, moment: Moment) -> Any:
        def hook(**payload: Any) -> None:
            praxis = getattr(payload.get("praxis"), "value", payload.get("praxis"))
            backend = getattr(payload.get("backend"), "name", None)
            if moment is Moment.ON_ERROR:
                self._log.warning("%s praxis=%s error=%r", moment.value, praxis, payload.get("error"))
            elif moment is Moment.AFTER_INFER:
                self._log.log(
                    self._level,
                    "%s praxis=%s backend=%s latency_ms=%.1f",
                    moment.value,
                    praxis,
                    backend,
                    getattr(payload.get("result"), "latency_ms", float("nan")),
                )
            else:
                self._log.log(self._level, "%s praxis=%s backend=%s", moment.value, praxis, backend)

        return hook


@dataclass(frozen=True)
class Timing:
    """One measured inference: wall time from ``BEFORE_INFER`` to ``AFTER_INFER``."""

    praxis: str
    backend: str
    seconds: float


class TimingHook:
    """Measures pipeline inference time (including backend loading on first use).

    Complements ``Theoria.latency_ms`` (which the backend reports itself) with
    an outside-in measurement. Thread-safe. Records are kept in memory;
    call :meth:`clear` periodically in long-running processes.
    """

    def __init__(self) -> None:
        self.records: list[Timing] = []
        self._lock = threading.Lock()
        self._local = threading.local()

    def attach(self, kairos: Kairos) -> TimingHook:
        kairos.on(Moment.BEFORE_INFER, self._before)
        kairos.on(Moment.AFTER_INFER, self._after)
        kairos.on(Moment.ON_ERROR, self._abandon)
        return self

    def _before(self, **_: Any) -> None:
        self._local.start = time.perf_counter()

    def _after(self, **payload: Any) -> None:
        start = getattr(self._local, "start", None)
        if start is None:
            return
        self._local.start = None
        timing = Timing(
            praxis=str(getattr(payload.get("praxis"), "value", payload.get("praxis"))),
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
        """Per-praxis ``{"count", "mean_s", "max_s"}``."""
        with self._lock:
            records = list(self.records)
        out: dict[str, dict[str, float]] = {}
        for praxis in sorted({r.praxis for r in records}):
            secs = [r.seconds for r in records if r.praxis == praxis]
            out[praxis] = {
                "count": float(len(secs)),
                "mean_s": sum(secs) / len(secs),
                "max_s": max(secs),
            }
        return out
