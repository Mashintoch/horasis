from __future__ import annotations

import pytest
from pydantic import ValidationError

from horasis.foundation.config import Config, QuotaRule, TierRule
from horasis.foundation.errors import ConfigurationError


def test_default_config_has_three_tiers() -> None:
    config = Config.default()
    assert set(config.tiers) == {"free", "pro", "enterprise"}
    assert config.tier("pro").quota.limit == 10_000


def test_tier_lookup_raises_for_unknown_tier() -> None:
    config = Config.default()
    with pytest.raises(ConfigurationError):
        config.tier("nonexistent")


def test_from_dict_builds_config() -> None:
    config = Config.from_dict(
        {
            "default_backend": "ultralytics",
            "tiers": {
                "free": {
                    "name": "free",
                    "quota": {"limit": 5, "window_seconds": 60},
                }
            },
        }
    )
    assert config.default_backend == "ultralytics"
    assert config.tier("free").quota.limit == 5


def test_from_dict_raises_configuration_error_on_bad_input() -> None:
    with pytest.raises(ConfigurationError):
        Config.from_dict({"tiers": {"free": {"name": "free", "quota": {"limit": -1}}}})


def test_from_yaml_missing_file_raises() -> None:
    with pytest.raises(ConfigurationError):
        Config.from_yaml("/nonexistent/path/config.yaml")


def test_from_yaml_loads_shipped_default_config() -> None:
    from pathlib import Path

    config_path = (
        Path(__file__).resolve().parents[2] / "src" / "horasis" / "config" / "default.yaml"
    )
    config = Config.from_yaml(config_path)
    assert config.default_backend == "ultralytics"
    assert "free" in config.tiers


def test_config_is_immutable() -> None:
    config = Config.default()
    with pytest.raises(ValidationError):
        config.default_backend = "onnx"  # type: ignore[misc]


def test_quota_rule_requires_positive_values() -> None:
    with pytest.raises(ValidationError):
        QuotaRule(limit=0, window_seconds=60)
    with pytest.raises(ValidationError):
        QuotaRule(limit=10, window_seconds=0)


def test_tier_rule_default_allowed_tasks() -> None:
    rule = TierRule(name="x", quota=QuotaRule(limit=1, window_seconds=1))
    assert "detect" in rule.allowed_tasks
    assert "moderate" in rule.allowed_tasks


def test_shipped_default_yaml_loads_with_safe_input_defaults() -> None:
    from importlib.resources import files

    path = files("horasis") / "config" / "default.yaml"
    config = Config.from_yaml(str(path))
    assert config.inputs.allow_urls is False
    assert set(config.tiers) == {"free", "pro", "enterprise"}


def test_moderation_threshold_per_category_override() -> None:
    config = Config(moderation_threshold=0.4, moderation_thresholds={"nsfw": 0.9})
    assert config.moderation_threshold_for("nsfw") == 0.9
    assert config.moderation_threshold_for("violence") == 0.4


def test_input_policy_rejects_nonsense_limits() -> None:
    from horasis.foundation.config import InputPolicy

    with pytest.raises(ValidationError):
        InputPolicy(max_bytes=0)
