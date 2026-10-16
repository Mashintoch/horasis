from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from horasis.foundation.sphalma import ConfigurationError


class QuotaRule(BaseModel):
    """A single rate/volume limit: at most ``limit`` calls per ``window_seconds``."""

    model_config = ConfigDict(frozen=True)

    limit: int = Field(gt=0)
    window_seconds: int = Field(gt=0)


class RateRule(BaseModel):
    """Token-bucket rate limit: ``per_minute`` sustained, ``burst`` capacity."""

    model_config = ConfigDict(frozen=True)

    per_minute: int = Field(gt=0)
    burst: int | None = Field(default=None, gt=0, description="Bucket size; defaults to per_minute.")

    @property
    def capacity(self) -> int:
        return self.burst or self.per_minute


DEFAULT_PRAXIS_COST: dict[str, int] = {
    "detect": 1,
    "classify": 1,
    "face": 1,
    "ocr": 2,
    "moderate": 2,
    "segment": 2,
    "pose": 2,
}


class TierRule(BaseModel):
    """Configuration for one :class:`~horasis.access.klimax.Klimax` tier."""

    model_config = ConfigDict(frozen=True)

    name: str
    quota: QuotaRule
    allowed_praxeis: list[str] = Field(
        default_factory=lambda: [
            "detect",
            "classify",
            "ocr",
            "face",
            "moderate",
            "segment",
            "pose",
        ],
        description="Praxis values this tier is permitted to invoke.",
    )
    rate: RateRule | None = Field(default=None, description="Per-minute limit across all praxeis.")
    praxis_rates: dict[str, RateRule] = Field(
        default_factory=dict, description="Extra per-minute limits for individual praxeis."
    )
    praxis_quotas: dict[str, QuotaRule] = Field(
        default_factory=dict, description="Extra windowed quotas for individual praxeis."
    )


class InputPolicy(BaseModel):
    """Rules `Physis` applies when loading caller-supplied images.

    Attributes:
        allow_urls: URL inputs are fetched only when this is True (off by
            default so a caller-controlled string can never trigger a
            network request unless the operator opted in).
        max_bytes: Largest accepted encoded image, in bytes.
        max_pixels: Largest accepted decoded image (width * height); guards
            against decompression bombs.
        allowed_formats: Pillow format names that may be decoded.
        url_timeout_seconds: Network timeout used when fetching a URL.
        apply_exif_orientation: Rotate/flip pixels per the EXIF orientation tag.
    """

    model_config = ConfigDict(frozen=True)

    allow_urls: bool = False
    max_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    max_pixels: int = Field(default=100_000_000, gt=0)
    allowed_formats: list[str] = Field(
        default_factory=lambda: ["JPEG", "PNG", "WEBP", "BMP", "TIFF", "GIF"]
    )
    url_timeout_seconds: float = Field(default=10.0, gt=0)
    apply_exif_orientation: bool = True


class Kanon(BaseModel):
    """Top-level runtime configuration for `horasis`.

    Attributes:
        default_backend: Name registered in the backend registry to use
            when a task does not specify one explicitly.
        tiers: Mapping of tier name -> :class:`TierRule`.
        telemetry_enabled: Whether :class:`~horasis.foundation.aisthesis.Aisthesis`
            hooks should fire.
        device: Compute device for backends that can choose: ``auto`` (let
            each backend decide; GPU if present), ``cpu``, ``cuda`` or ``mps``.
        max_loaded_models: Cap on backends holding models in memory at once;
            the least recently used is unloaded (and reloaded lazily on next use).
            ``None`` means no cap.
        inputs: Image-loading policy (size limits, formats, URL opt-in).
        backend_order: Per-praxis backend preference, e.g.
            ``{"ocr": ["easyocr", "tesseract"]}``. The first registered
            backend in the list that supports the praxis wins.
        fallback_order: Backends tried (in order) after ``backend_order`` and
            ``default_backend`` found nothing, before any other registered one.
        moderation_threshold: Score at or above which a moderation category
            is flagged, unless overridden in ``moderation_thresholds``.
        moderation_thresholds: Per-category flag thresholds.
        praxis_cost: Units of quota/rate each praxis consumes per call.
        moderation_safe_categories: Categories that describe benign content
            (e.g. ``"normal"``); never flagged regardless of score.
        extra: Free-form namespace for plugin-specific settings.
    """

    model_config = ConfigDict(frozen=True)

    default_backend: str = "ultralytics"
    tiers: dict[str, TierRule] = Field(default_factory=dict)
    telemetry_enabled: bool = True
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    max_loaded_models: int | None = Field(default=None, gt=0)
    inputs: InputPolicy = Field(default_factory=InputPolicy)
    backend_order: dict[str, list[str]] = Field(default_factory=dict)
    fallback_order: list[str] = Field(default_factory=list)
    moderation_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    moderation_thresholds: dict[str, float] = Field(default_factory=dict)
    praxis_cost: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_PRAXIS_COST))
    moderation_safe_categories: list[str] = Field(
        default_factory=lambda: ["normal", "safe", "sfw", "neutral", "drawings"]
    )
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
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
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
                "free": TierRule(
                    name="free",
                    quota=QuotaRule(limit=100, window_seconds=86400),
                    rate=RateRule(per_minute=20, burst=10),
                ),
                "pro": TierRule(
                    name="pro",
                    quota=QuotaRule(limit=10_000, window_seconds=86400),
                    rate=RateRule(per_minute=300, burst=60),
                ),
                "enterprise": TierRule(
                    name="enterprise",
                    quota=QuotaRule(limit=1_000_000, window_seconds=86400),
                ),
            },
        )

    def cost_for(self, praxis: str) -> int:
        """Units one call to `praxis` consumes (OCR/moderation cost more)."""
        # pydantic fields look like FieldInfo to pylint, hence the no-member pragma
        return max(1, self.praxis_cost.get(praxis, 1))  # pylint: disable=no-member

    def moderation_threshold_for(self, category: str) -> float:
        """Flag threshold for a moderation `category` (per-category override wins)."""
        return self.moderation_thresholds.get(  # pylint: disable=no-member
            category, self.moderation_threshold
        )

    def tier(self, name: str) -> TierRule:
        try:
            return self.tiers[name]
        except KeyError as exc:
            raise ConfigurationError(f"Unknown tier: {name!r}") from exc
