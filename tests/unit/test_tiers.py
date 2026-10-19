from __future__ import annotations

import pytest

from horasis.access.tiers import Tier


def test_tier_values() -> None:
    assert Tier.FREE.value == "free"
    assert Tier.PRO.value == "pro"
    assert Tier.ENTERPRISE.value == "enterprise"


def test_from_value_accepts_string() -> None:
    assert Tier.from_value("pro") is Tier.PRO


def test_from_value_accepts_enum_instance() -> None:
    assert Tier.from_value(Tier.ENTERPRISE) is Tier.ENTERPRISE


def test_from_value_rejects_unknown_string() -> None:
    with pytest.raises(ValueError):
        Tier.from_value("legendary")
