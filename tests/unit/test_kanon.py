from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from horasis.foundation.kanon import Kanon, QuotaRule, TierRule
from horasis.foundation.sphalma import ConfigurationError


def test_default_kanon_has_three_tiers() -> None:
    kanon = Kanon.default()
    assert set(kanon.tiers) == {"free", "pro", "enterprise"}
    assert kanon.tier("pro").quota.limit == 10_000


def test_tier_lookup_raises_for_unknown_tier() -> None:
    kanon = Kanon.default()
    with pytest.raises(ConfigurationError):
        kanon.tier("nonexistent")


def test_from_dict_builds_kanon() -> None:
    kanon = Kanon.from_dict(
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
    assert kanon.default_backend == "ultralytics"
    assert kanon.tier("free").quota.limit == 5


def test_from_dict_raises_configuration_error_on_bad_input() -> None:
    with pytest.raises(ConfigurationError):
        Kanon.from_dict({"tiers": {"free": {"name": "free", "quota": {"limit": -1}}}})


def test_from_yaml_missing_file_raises() -> None:
    with pytest.raises(ConfigurationError):
        Kanon.from_yaml("/nonexistent/path/kanon.yaml")


def test_from_yaml_loads_shipped_default_config() -> None:
    from pathlib import Path

    config_path = (
        Path(__file__).resolve().parents[2] / "src" / "horasis" / "config" / "default.yaml"
    )
    kanon = Kanon.from_yaml(config_path)
    assert kanon.default_backend == "ultralytics"
    assert "free" in kanon.tiers


def test_kanon_is_immutable() -> None:
    kanon = Kanon.default()
    with pytest.raises(FrozenInstanceError):
        kanon.default_backend = "onnx"  # type: ignore[misc]


def test_quota_rule_requires_positive_values() -> None:
    with pytest.raises(ValidationError):
        QuotaRule(limit=0, window_seconds=60)
    with pytest.raises(ValidationError):
        QuotaRule(limit=10, window_seconds=0)


def test_tier_rule_default_allowed_praxeis() -> None:
    rule = TierRule(name="x", quota=QuotaRule(limit=1, window_seconds=1))
    assert "detect" in rule.allowed_praxeis
    assert "moderate" in rule.allowed_praxeis
