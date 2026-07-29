from __future__ import annotations

from typing import Protocol


class Aisthesis(Protocol):
    """Protocol for telemetry sinks.

    Implementations must be cheap and non-blocking; `horasis` calls these
    hooks synchronously on the request path.
    """

    def on_call_start(self, *, praxis: str, principal_id: str) -> None:
        """Fired when a public API call begins, after access checks pass."""
        ...

    def on_call_end(
        self,
        *,
        praxis: str,
        principal_id: str,
        backend: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        """Fired when a public API call completes, successfully or not."""
        ...

    def on_error(self, *, praxis: str, principal_id: str, error: BaseException) -> None:
        """Fired whenever a `Sphalma` (or subclass) is raised on the request path."""
        ...


class NullAisthesis:
    """No-op :class:`Aisthesis` implementation; the default when telemetry is off."""

    def on_call_start(self, *, praxis: str, principal_id: str) -> None:
        return None

    def on_call_end(
        self,
        *,
        praxis: str,
        principal_id: str,
        backend: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        return None

    def on_error(self, *, praxis: str, principal_id: str, error: BaseException) -> None:
        return None


class LoggingAisthesis:
    """A minimal :class:`Aisthesis` implementation that writes to `logging`."""

    def __init__(self, logger_name: str = "horasis.telemetry") -> None:
        import logging

        self._log = logging.getLogger(logger_name)

    def on_call_start(self, *, praxis: str, principal_id: str) -> None:
        self._log.debug("call_start praxis=%s principal=%s", praxis, principal_id)

    def on_call_end(
        self,
        *,
        praxis: str,
        principal_id: str,
        backend: str,
        latency_ms: float,
        success: bool,
    ) -> None:
        self._log.info(
            "call_end praxis=%s principal=%s backend=%s latency_ms=%.2f success=%s",
            praxis,
            principal_id,
            backend,
            latency_ms,
            success,
        )

    def on_error(self, *, praxis: str, principal_id: str, error: BaseException) -> None:
        self._log.warning("call_error praxis=%s principal=%s error=%s", praxis, principal_id, error)
