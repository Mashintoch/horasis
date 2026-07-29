from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from horasis.foundation.sphalma import ConfigurationError


class QuotaRule(BaseModel):
    """A single rate/volume limit: at most ``limit`` calls per ``window_seconds``."""

    model_config = ConfigDict(frozen=True)

    limit: int = Field(gt=0)
    window_seconds: int = Field(gt=0)


class TierRule(BaseModel):
    """Configuration for one :class:`~horasis.access.klimax.Klimax` tier."""

    model_config = ConfigDict(frozen=True)

    name: str
    quota: QuotaRule
    allowed_praxeis: list[str] = Field(
        default_factory=lambda: ["detect", "classify", "ocr", "face", "moderate"],
        description="Praxis values this tier is permitted to invoke.",
    )


class Kanon(BaseModel):
    """Top-level runtime configuration for `horasis`.

    Attributes:
        default_backend: Name registered in the backend registry to use
            when a task does not specify one explicitly.
        tiers: Mapping of tier name -> :class:`TierRule`.
        telemetry_enabled: Whether :class:`~horasis.foundation.aisthesis.Aisthesis`
            hooks should fire.
        extra: Free-form namespace for plugin-specific settings.
    """

    model_config = ConfigDict(frozen=True)

    default_backend: str = "ultralytics"
    tiers: dict[str, TierRule] = Field(default_factory=dict)
    telemetry_enabled: bool = True
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Kanon:
        """Load a :class:`Kanon` from a YAML file.

        Raises:
            ConfigurationError: if the file is missing or malformed.
        """
        p = Path(path)
        if not p.is_file():
            raise ConfigurationError(f"Kanon config file not found: {p}")
        try:
            raw = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Kanon config file is not valid YAML: {p}") from exc
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Kanon:
        try:
            return cls.model_validate(raw)
        except Exception as exc:  # pydantic.ValidationError, narrowed for callers
            raise ConfigurationError(f"Invalid Kanon configuration: {exc}") from exc

    @classmethod
    def default(cls) -> Kanon:
        """A sensible built-in configuration, used when no file is supplied."""
        return cls(
            default_backend="ultralytics",
            telemetry_enabled=True,
            tiers={
                "free": TierRule(name="free", quota=QuotaRule(limit=100, window_seconds=86400)),
                "pro": TierRule(name="pro", quota=QuotaRule(limit=10_000, window_seconds=86400)),
                "enterprise": TierRule(
                    name="enterprise",
                    quota=QuotaRule(limit=1_000_000, window_seconds=86400),
                ),
            },
        )

    def tier(self, name: str) -> TierRule:
        try:
            return self.tiers[name]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown tier: {name!r}") from exc
