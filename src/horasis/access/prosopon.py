from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from horasis.access.klimax import Klimax


class Prosopon(BaseModel):
    """An authenticated caller.

    Attributes:
        principal_id: Stable identifier for the caller (never the raw key).
        tier: The caller's :class:`Klimax` tier.
        metadata: Free-form data attached by the key-validation backend
            (e.g. organization name, plan expiry).
        scopes: Praxis values this caller's key may invoke, or ``None`` for
            "everything the tier allows". Scopes only narrow the tier.
    """

    model_config = ConfigDict(frozen=True)

    principal_id: str
    tier: Klimax
    metadata: dict[str, str] = Field(default_factory=dict)
    scopes: list[str] | None = None


class AnonymousProsopon(Prosopon):
    """Sentinel principal used only in tests / local development."""

    principal_id: str = "anonymous"
    tier: Klimax = Klimax.FREE
