from __future__ import annotations

import pytest

from horasis.access.klimax import Klimax


def test_klimax_values() -> None:
    assert Klimax.FREE.value == "free"
    assert Klimax.PRO.value == "pro"
    assert Klimax.ENTERPRISE.value == "enterprise"


def test_from_value_accepts_string() -> None:
    assert Klimax.from_value("pro") is Klimax.PRO


def test_from_value_accepts_enum_instance() -> None:
    assert Klimax.from_value(Klimax.ENTERPRISE) is Klimax.ENTERPRISE


def test_from_value_rejects_unknown_string() -> None:
    with pytest.raises(ValueError):
        Klimax.from_value("legendary")
