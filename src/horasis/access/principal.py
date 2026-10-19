from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from horasis.access.tiers import Tier


class Principal(BaseModel):
    """An authenticated caller.

    Attributes:
        principal_id: Stable identifier for the caller (never the raw key).
        tier: The caller's :class:`Tier` tier.
        metadata: Free-form data attached by the key-validation backend
            (e.g. organization name, plan expiry).
        scopes: Task values this caller's key may invoke, or ``None`` for
            "everything the tier allows". Scopes only narrow the tier.
    """

    model_config = ConfigDict(frozen=True)

    principal_id: str
    tier: Tier
    metadata: dict[str, str] = Field(default_factory=dict)
    scopes: list[str] | None = None


class AnonymousPrincipal(Principal):
    """Sentinel principal used only in tests / local development."""

    principal_id: str = "anonymous"
    tier: Tier = Tier.FREE
