from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from horasis.foundation.errors import ConfigurationError


class QuotaRule(BaseModel):
    """Windowed quota: at most ``limit`` units per ``window_seconds``."""

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


DEFAULT_TASK_COST: dict[str, int] = {
    "detect": 1,
    "classify": 1,
    "face": 1,
    "ocr": 2,
    "moderate": 2,
    "segment": 2,
    "pose": 2,
}


class TierRule(BaseModel):
    """Limits and allowed tasks for one tier."""

    model_config = ConfigDict(frozen=True)

    name: str
    quota: QuotaRule
    allowed_tasks: list[str] = Field(
        default_factory=lambda: [
            "detect",
            "classify",
            "ocr",
            "face",
            "moderate",
            "segment",
            "pose",
        ],
        description="Task values this tier is permitted to invoke.",
    )
    rate: RateRule | None = Field(default=None, description="Per-minute limit across all tasks.")
    task_rates: dict[str, RateRule] = Field(
        default_factory=dict, description="Extra per-minute limits for individual tasks."
    )
    task_quotas: dict[str, QuotaRule] = Field(
        default_factory=dict, description="Extra windowed quotas for individual tasks."
    )


class InputPolicy(BaseModel):
    """Limits applied when loading caller-supplied images.

    Attributes:
        allow_urls: Fetch URL inputs. Off by default, so a caller-supplied
            string cannot trigger a network request unless the operator opts in.
        max_bytes: Largest accepted encoded image, in bytes.
        max_pixels: Largest accepted decoded image (width * height); guards
            against decompression bombs.
        allowed_formats: Pillow format names that may be decoded.
        url_timeout_seconds: Connect and per-read timeout used when fetching a URL.
        url_allow_hosts: If non-empty, only these hostnames may be fetched.
            Resolved addresses must still be public.
        url_deny_hosts: Hostnames never fetched (wins over the allow list).
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
    url_allow_hosts: list[str] = Field(default_factory=list)
    url_deny_hosts: list[str] = Field(default_factory=list)
    apply_exif_orientation: bool = True


class Config(BaseModel):
    """Runtime configuration.

    Attributes:
        default_backend: Name registered in the backend registry to use
            when a task does not specify one explicitly.
        tiers: Mapping of tier name -> :class:`TierRule`.
        telemetry_enabled: Fire :class:`~horasis.foundation.telemetry.Telemetry` hooks.
        device: Compute device for backends that can choose: ``auto`` (let
            each backend decide; GPU if present), ``cpu``, ``cuda`` or ``mps``.
        max_loaded_models: Cap on backends holding models in memory at once;
            the least recently used is unloaded (and reloaded lazily on next use).
            ``None`` means no cap.
        inputs: Image-loading policy (size limits, formats, URL opt-in).
        backend_order: Per-task backend preference, e.g.
            ``{"ocr": ["easyocr", "tesseract"]}``. The first registered
            backend in the list that supports the task wins.
        fallback_order: Backends tried (in order) after ``backend_order`` and
            ``default_backend`` found nothing, before any other registered one.
        moderation_threshold: Score at or above which a moderation category
            is flagged, unless overridden in ``moderation_thresholds``.
        moderation_thresholds: Per-category flag thresholds.
        task_cost: Units of quota/rate each task consumes per call.
        moderation_safe_categories: Categories that describe benign content
            (e.g. ``"normal"``); never flagged regardless of score.
        auto_download_models: Let backends download missing model files (such as
            the YuNet face model) on first use. Off by default; otherwise they
            raise and tell you to run ``horasis models pull``.
        rate_limit_store: Where per-minute token buckets live: ``memory`` (per
            process) or ``redis`` (shared by all workers; needs `redis_url`).
        redis_url: Redis URL used when `rate_limit_store` is ``redis`` and
            `Client.build` is not given a limiter/quota store to reuse.
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
    task_cost: dict[str, int] = Field(default_factory=lambda: dict(DEFAULT_TASK_COST))
    moderation_safe_categories: list[str] = Field(
        default_factory=lambda: ["normal", "safe", "sfw", "neutral", "drawings"]
    )
    auto_download_models: bool = False
    rate_limit_store: Literal["memory", "redis"] = "memory"
    redis_url: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a :class:`Config` from a YAML file.

        Raises:
            ConfigurationError: if the file is missing or malformed.
        """
        p = Path(path)
        if not p.is_file():
            raise ConfigurationError(f"config file not found: {p}")
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"config file is not valid YAML: {p}") from exc
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        try:
            return cls.model_validate(raw)
        except Exception as exc:  # pydantic.ValidationError, narrowed for callers
            raise ConfigurationError(f"invalid configuration: {exc}") from exc

    @classmethod
    def default(cls) -> Config:
        """Return the built-in configuration used when no file is given."""
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

    def cost_for(self, task: str) -> int:
        """Return the quota units one call to `task` consumes."""
        # pylint sees pydantic fields as FieldInfo
        return max(1, self.task_cost.get(task, 1))  # pylint: disable=no-member

    def moderation_threshold_for(self, category: str) -> float:
        """Return the flag threshold for a moderation `category`."""
        return self.moderation_thresholds.get(  # pylint: disable=no-member
            category, self.moderation_threshold
        )

    def tier(self, name: str) -> TierRule:
        try:
            return self.tiers[name]
        except KeyError as exc:
            raise ConfigurationError(f"unknown tier {name!r}") from exc
