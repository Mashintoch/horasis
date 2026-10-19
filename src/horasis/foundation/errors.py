from __future__ import annotations


class HorasisError(Exception):
    """Base class for every exception raised by `horasis`.

    Attributes:
        message: Human-readable description of the failure.
        context: Optional structured metadata useful for debugging or
            logging (never includes secrets such as API keys).
    """

    def __init__(self, message: str, *, context: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict[str, object] = context or {}

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"


class ConfigurationError(HorasisError):
    """Raised when configuration is invalid."""


class ValidationError(HorasisError):
    """Raised when caller-supplied input fails validation before reaching a backend."""


class AccessDeniedError(HorasisError):
    """Raised when a principal may not perform an action."""


class QuotaExceededError(AccessDeniedError):
    """Raised when a quota window is exhausted.

    Attributes:
        limit: The limit that was hit.
        used: Usage that the rejected call would have reached.
        window_seconds: Length of the quota window.
        retry_after: Seconds until the window resets (``None`` if unknown).
        scope: Which limit: ``"quota"`` or ``"task:<name>"``.
    """

    def __init__(
        self,
        message: str,
        *,
        limit: int,
        used: int,
        window_seconds: int,
        retry_after: float | None = None,
        scope: str = "quota",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.limit = limit
        self.used = used
        self.window_seconds = window_seconds
        self.retry_after = retry_after
        self.scope = scope


class RateLimitedError(AccessDeniedError):
    """Raised when a per-minute token-bucket rate limit rejects a call.

    Attributes:
        limit: Calls per minute allowed by the rule that was hit.
        retry_after: Seconds until enough tokens refill (``None`` if the call
            costs more than the bucket can ever hold).
        scope: ``"rate"`` or ``"rate:<task>"``.
    """

    def __init__(
        self,
        message: str,
        *,
        limit: int,
        retry_after: float | None,
        scope: str = "rate",
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.limit = limit
        self.retry_after = retry_after
        self.scope = scope


class BackendUnavailableError(HorasisError):
    """Raised when a backend cannot serve a request."""


class NoBackendError(BackendUnavailableError):
    """Raised when no registered backend supports a task.

    ``context`` carries ``task``, the ``registered`` backend names and the
    ``considered`` names that were checked.
    """


class ModelDownloadError(HorasisError):
    """Raised when a model file cannot be downloaded or fails its checksum."""


class AdapterError(HorasisError):
    """Raised when a backend fails during inference."""


class RemoteError(AdapterError):
    """Raised when a remote horasis server fails or is unreachable.

    Attributes:
        status_code: HTTP status, or ``None`` for transport failures (timeouts,
            connection errors).
        retryable: Whether retrying the same request might succeed.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.status_code = status_code
        self.retryable = retryable


class VisualizationUnavailableError(HorasisError):
    """Raised when an optional drawing or video dependency is missing."""


class PluginError(HorasisError):
    """Raised when a third-party plugin fails to load or register."""


class InvalidApiKeyError(AccessDeniedError):
    """Raised when an API key cannot be authenticated."""


class ExpiredApiKeyError(InvalidApiKeyError):
    """Raised when an API key has expired."""


class RevokedApiKeyError(InvalidApiKeyError):
    """Raised when an API key has been revoked."""
