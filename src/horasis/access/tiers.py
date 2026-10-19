"""Tier -- access tiers."""

from __future__ import annotations

from enum import Enum


class Tier(str, Enum):
    """A caller's subscription tier."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

    @classmethod
    def from_value(cls, value: str | Tier) -> Tier:
        if isinstance(value, Tier):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"Unknown tier: {value!r}") from exc
