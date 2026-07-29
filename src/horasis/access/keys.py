"""API key validation interface."""

from __future__ import annotations

import time
from typing import Protocol

from horasis.access.klimax import Klimax
from horasis.access.prosopon import Prosopon
from horasis.foundation.sphalma import ExpiredApiKeyError, InvalidApiKeyError, RevokedApiKeyError


class ApiKeyValidator(Protocol):
    """Resolves a raw API key string into an authenticated :class:`Prosopon`."""

    def validate(self, api_key: str) -> Prosopon:
        """Return the :class:`Prosopon` for `api_key`.

        Raises:
            InvalidApiKeyError: if the key is missing, malformed, or unknown.
            ExpiredApiKeyError: if the key existed but has passed its expiry.
            RevokedApiKeyError: if the key was explicitly revoked.
        """
        ...


class InMemoryApiKeyValidator:
    """A simple validator backed by a dict; intended for tests and demos."""

    def __init__(self) -> None:

        self._keys: dict[str, tuple[Prosopon, float | None, bool]] = {}

    def register(
        self,
        api_key: str,
        principal: Prosopon,
        *,
        expires_at: float | None = None,
    ) -> None:
        """Register `api_key`, optionally with an expiry (Unix epoch seconds)."""
        self._keys[api_key] = (principal, expires_at, False)

    def register_simple(
        self,
        api_key: str,
        *,
        principal_id: str,
        tier: Klimax,
        expires_at: float | None = None,
    ) -> None:
        self.register(
            api_key, Prosopon(principal_id=principal_id, tier=tier), expires_at=expires_at
        )

    def revoke(self, api_key: str) -> None:
        """Mark `api_key` as explicitly revoked without deleting its record.

        Distinguishes "this key was cut off on purpose" from "this key
        never existed" in the exception raised on the next `validate()` call.
        """
        if api_key in self._keys:
            principal, expires_at, _ = self._keys[api_key]
            self._keys[api_key] = (principal, expires_at, True)

    def validate(self, api_key: str) -> Prosopon:
        if not api_key:
            raise InvalidApiKeyError("API key is empty")

        entry = self._keys.get(api_key)
        if entry is None:
            raise InvalidApiKeyError("API key is unknown")

        principal, expires_at, revoked = entry
        if revoked:
            raise RevokedApiKeyError(
                "API key has been revoked", context={"principal_id": principal.principal_id}
            )
        if expires_at is not None and time.time() >= expires_at:
            raise ExpiredApiKeyError(
                "API key has expired", context={"principal_id": principal.principal_id}
            )
        return principal
