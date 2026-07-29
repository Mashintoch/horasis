from __future__ import annotations


class Sphalma(Exception):
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


HorasisError = Sphalma


class ConfigurationError(Sphalma):
    """Raised when :class:`~horasis.foundation.kanon.Kanon` configuration is invalid."""


class ValidationError(Sphalma):
    """Raised when caller-supplied input fails validation before reaching a backend."""


class AccessDeniedError(Sphalma):
    """Raised by the access layer when a principal lacks permission for an action."""


class QuotaExceededError(AccessDeniedError):
    """Raised by :class:`~horasis.access.metron.Metron` when a quota is exhausted."""

    def __init__(
        self,
        message: str,
        *,
        limit: int,
        used: int,
        window_seconds: int,
        context: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, context=context)
        self.limit = limit
        self.used = used
        self.window_seconds = window_seconds


class BackendUnavailableError(Sphalma):
    """Raised when no backend can be resolved for a requested task."""


class AdapterError(Sphalma):
    """Raised when a backend adapter fails to execute inference."""


class VisualizationUnavailableError(Sphalma):
    """Raised when `horasis.viz` is used without its optional dependency installed."""


class PluginError(Sphalma):
    """Raised when a third-party plugin fails to load or register."""


class InvalidApiKeyError(AccessDeniedError):
    """Raised when an API key cannot be authenticated."""


class ExpiredApiKeyError(InvalidApiKeyError):
    """Raised when an API key existed but has passed its expiry time."""


class RevokedApiKeyError(InvalidApiKeyError):
    """Raised when an API key was explicitly revoked, as opposed to simply
    never having existed or having expired.
    """
