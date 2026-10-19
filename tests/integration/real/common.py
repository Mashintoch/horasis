"""Shared helpers for the real-model integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from horasis import Client, models
from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.tiers import Tier
from horasis.backend.base import Backend
from horasis.foundation.imaging import to_image
from horasis.foundation.schemas import BoundingBox, Confidence

KEY = "integration-key"


def make_client(*backends: Backend) -> Client:
    """A `Client` with the real Ultralytics adapter plus `backends` and one enterprise key."""
    validator = InMemoryApiKeyValidator()
    validator.register_simple(KEY, principal_id="integration", tier=Tier.ENTERPRISE)
    return Client.build(validator=validator, backends=list(backends))


def weights(name: str) -> str:
    """Cache path for a weights file, so Ultralytics downloads it there, not into the cwd."""
    path = models.cache_dir() / "weights" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def sample_image(name: str) -> Path:
    """An image shipped inside the Ultralytics package (``bus.jpg`` or ``zidane.jpg``)."""
    pytest.importorskip("ultralytics", reason="needs the ultralytics package for its sample images")
    from ultralytics.utils import ASSETS

    path = Path(ASSETS) / name
    if not path.is_file():
        pytest.skip(f"{name} is not shipped with this ultralytics version")
    return path


def image_size(path: Path) -> tuple[int, int]:
    img = to_image(path)
    return img.width, img.height


def assert_confidence(value: Confidence) -> None:
    assert 0.0 <= value.value <= 1.0


def assert_box_in_bounds(box: BoundingBox, size: tuple[int, int], tolerance: float = 1.5) -> None:
    width, height = size
    assert -tolerance <= box.x_min < box.x_max <= width + tolerance, box
    assert -tolerance <= box.y_min < box.y_max <= height + tolerance, box


def assert_is_name(label: str) -> None:
    """A label must be a class name, not a raw index (a classic label-mapping bug)."""
    assert label and not label.strip().isdigit(), f"label looks like a raw index: {label!r}"
