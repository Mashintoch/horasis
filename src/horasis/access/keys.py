"""API key validation interface."""

from __future__ import annotationsimport hashlibimport hmacimport osimport secretsimport timefrom dataclasses import dataclassfrom typing import Protocolfrom horasis.access.klimax import Klimaxfrom horasis.access.prosopon import Prosoponfrom horasis.foundation.sphalma import ExpiredApiKeyError, InvalidApiKeyError, RevokedApiKeyErrorclass ApiKeyValidator(Protocol):
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
    """A simple validator backed by a dict; intended for tests and demos.

    Keys are held in plaintext. For anything real use :class:`HashedApiKeyValidator`.
    """

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
        scopes: list[str] | None = None,
    ) -> None:
        self.register(
            api_key,
            Prosopon(principal_id=principal_id, tier=tier, scopes=scopes),
            expires_at=expires_at,
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


@dataclass
class _KeyRecord:
    salt: bytes
    digest: bytes
    principal: Prosopon
    expires_at: float | None
    revoked: bool = False


def _digest(salt: bytes, secret: str) -> bytes:
    return hashlib.sha256(salt + secret.encode()).digest()


class HashedApiKeyValidator:
    """Validator that never stores a key, only a per-key salted SHA-256 hash.

    Keys look like ``hrs_<key_id>.<secret>``. The public ``key_id`` finds the
    record; the 256-bit random ``secret`` is checked against its salted hash
    with a constant-time comparison. (A fast hash is appropriate because the
    secret is high-entropy random, not a human password.) A key is shown once,
    at :meth:`issue` time, and cannot be recovered afterwards.

    Supports expiry, revocation and scopes (which praxeis the key may call).
    Records live in memory; use :meth:`export_records` / :meth:`load_records`
    to persist them in your own store.
    """

    PREFIX = "hrs_"

    def __init__(self) -> None:
        self._records: dict[str, _KeyRecord] = {}

    def issue(
        self,
        *,
        principal_id: str,
        tier: Klimax,
        expires_at: float | None = None,
        scopes: list[str] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> str:
        """Create a key and return it. This is the only time it is available."""
        key_id = secrets.token_hex(6)
        secret = secrets.token_urlsafe(32)
        salt = os.urandom(16)
        principal = Prosopon(
            principal_id=principal_id, tier=tier, scopes=scopes, metadata=metadata or {}
        )
        self._records[key_id] = _KeyRecord(salt, _digest(salt, secret), principal, expires_at)
        return f"{self.PREFIX}{key_id}.{secret}"

    @staticmethod
    def _split(api_key: str) -> tuple[str, str] | None:
        if not api_key.startswith(HashedApiKeyValidator.PREFIX) or "." not in api_key:
            return None
        key_id, _, secret = api_key[len(HashedApiKeyValidator.PREFIX) :].partition(".")
        return (key_id, secret) if key_id and secret else None

    def revoke(self, api_key_or_id: str) -> None:
        """Revoke by full key or by ``key_id``."""
        parts = self._split(api_key_or_id)
        record = self._records.get(parts[0] if parts else api_key_or_id)
        if record is not None:
            record.revoked = True

    def validate(self, api_key: str) -> Prosopon:
        parts = self._split(api_key) if api_key else None
        if parts is None:
            raise InvalidApiKeyError("API key is malformed or empty")
        key_id, secret = parts
        record = self._records.get(key_id)
        # Hash even for unknown ids so timing doesn't reveal which ids exist.
        salt = record.salt if record else bytes(16)
        expected = record.digest if record else bytes(32)
        ok = hmac.compare_digest(_digest(salt, secret), expected)
        if record is None or not ok:
            raise InvalidApiKeyError("API key is unknown")
        if record.revoked:
            raise RevokedApiKeyError(
                "API key has been revoked", context={"principal_id": record.principal.principal_id}
            )
        if record.expires_at is not None and time.time() >= record.expires_at:
            raise ExpiredApiKeyError(
                "API key has expired", context={"principal_id": record.principal.principal_id}
            )
        return record.principal

    def export_records(self) -> dict[str, dict[str, object]]:
        """JSON-friendly snapshot (hashes only, never keys) for persistence."""
        return {
            kid: {
                "salt": r.salt.hex(),
                "digest": r.digest.hex(),
                "principal": r.principal.model_dump(mode="json"),
                "expires_at": r.expires_at,
                "revoked": r.revoked,
            }
            for kid, r in self._records.items()
        }

    def load_records(self, records: dict[str, dict[str, object]]) -> None:
        """Restore a snapshot made by :meth:`export_records`."""
        for kid, r in records.items():
            self._records[kid] = _KeyRecord(
                salt=bytes.fromhex(str(r["salt"])),
                digest=bytes.fromhex(str(r["digest"])),
                principal=Prosopon.model_validate(r["principal"]),
                expires_at=r["expires_at"],  # type: ignore[arg-type]
                revoked=bool(r["revoked"]),
            )
