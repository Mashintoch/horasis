"""Klimax -- access tiers."""

from __future__ import annotations

from enum import Enum


class Klimax(str, Enum):
    """A caller's subscription tier."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"

    @classmethod
    def from_value(cls, value: str | Klimax) -> Klimax:
        if isinstance(value, Klimax):
            return value
        try:
            return cls(value)
        except ValueError as exc:
            raise ValueError(f"Unknown Klimax tier: {value!r}") from exc
