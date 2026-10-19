from __future__ import annotations

import threading
from typing import Any, Protocol


class Telemetry(Protocol):
    """Protocol for telemetry sinks.

    Implementations must be cheap and non-blocking; `horasis` calls these
    hooks synchronously on the request path.
    """

    def on_call_start(self, *, task: str, principal_id: str) -> None:
        """Fired when a public API call begins, after access checks pass."""

    def on_call_end(
        self,
        *,
        task: str,
        principal_id: str,
        backend: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Fired when a public API call completes, successfully or not."""

    def on_error(self, *, task: str, principal_id: str, error: BaseException) -> None:
        """Fired whenever a `HorasisError` (or subclass) is raised on the request path."""


class NullTelemetry:
    """No-op :class:`Telemetry` implementation; the default when telemetry is off."""

    def on_call_start(self, *, task: str, principal_id: str) -> None:
        return None

    def on_call_end(
        self,
        *,
        task: str,
        principal_id: str,
        backend: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        return None

    def on_error(self, *, task: str, principal_id: str, error: BaseException) -> None:
        return None


class LoggingTelemetry:
    """A minimal :class:`Telemetry` implementation that writes to `logging`."""

    def __init__(self, logger_name: str = "horasis.telemetry") -> None:
        import logging

        self._log = logging.getLogger(logger_name)

    def on_call_start(self, *, task: str, principal_id: str) -> None:
        self._log.debug("call_start task=%s principal=%s", task, principal_id)

    def on_call_end(
        self,
        *,
        task: str,
        principal_id: str,
        backend: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        self._log.info(
            "call_end task=%s principal=%s backend=%s latency_ms=%.2f success=%s",
            task,
            principal_id,
            backend,
            latency_ms,
            success,
        )

    def on_error(self, *, task: str, principal_id: str, error: BaseException) -> None:
        self._log.warning("call_error task=%s principal=%s error=%s", task, principal_id, error)


class OTelTelemetry:
    """Emits OpenTelemetry spans and metrics (extra: ``horasis[otel]``).

    One span per public API call (``horasis.<task>``), a ``horasis.calls``
    counter, a ``horasis.errors`` counter and a ``horasis.latency_ms``
    histogram, tagged by task and backend. Configure the OpenTelemetry SDK
    and exporters as usual; this class only uses the API.

    **No-op when OpenTelemetry is not installed**: construction succeeds,
    ``enabled`` is ``False`` and every hook returns immediately, so you can
    wire it in unconditionally.

    Args:
        tracer, meter: Pre-built OpenTelemetry objects (mainly for tests);
            default to ``trace.get_tracer("horasis")`` / ``metrics.get_meter("horasis")``.
    """

    def __init__(self, *, tracer: Any = None, meter: Any = None) -> None:
        self.enabled = False
        self._local = threading.local()
        if tracer is None or meter is None:
            try:
                from opentelemetry import metrics, trace
            except ImportError:
                return
            tracer = tracer or trace.get_tracer("horasis")
            meter = meter or metrics.get_meter("horasis")
        self._tracer = tracer
        self._calls = meter.create_counter("horasis.calls", description="API calls")
        self._errors = meter.create_counter("horasis.errors", description="Failed API calls")
        self._latency = meter.create_histogram(
            "horasis.latency_ms", unit="ms", description="API call latency"
        )
        self.enabled = True

    def _spans(self) -> list[Any]:
        spans: list[Any] | None = getattr(self._local, "spans", None)
        if spans is None:
            spans = self._local.spans = []
        return spans

    def on_call_start(self, *, task: str, principal_id: str) -> None:
        if not self.enabled:
            return
        span = self._tracer.start_span(
            f"horasis.{task}",
            attributes={"horasis.task": task, "horasis.principal_id": principal_id},
        )
        self._spans().append(span)

    def on_call_end(
        self,
        *,
        task: str,
        principal_id: str,
        backend: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        if not self.enabled:
            return
        attrs = {"horasis.task": task, "horasis.backend": backend, "horasis.success": success}
        spans = self._spans()
        if spans:
            span = spans.pop()
            span.set_attribute("horasis.backend", backend)
            span.set_attribute("horasis.latency_ms", latency_ms)
            span.end()
        self._calls.add(1, attrs)
        self._latency.record(latency_ms, attrs)

    def on_error(self, *, task: str, principal_id: str, error: BaseException) -> None:
        if not self.enabled:
            return
        spans = self._spans()
        if spans:
            span = spans.pop()
            span.record_exception(error)
            try:
                from opentelemetry.trace import Status, StatusCode

                span.set_status(Status(StatusCode.ERROR, str(error)))
            except ImportError:
                pass
            span.end()
        attrs = {"horasis.task": task, "horasis.error": type(error).__name__}
        self._calls.add(1, {**attrs, "horasis.success": False})
        self._errors.add(1, attrs)
