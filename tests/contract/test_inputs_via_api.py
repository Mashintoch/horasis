from __future__ import annotations

import numpy as np
import pytest
from PIL import Image as PILImage

from horasis import Client, Config, InputPolicy
from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.tiers import Tier
from horasis.backend.registry import BackendRegistry
from horasis.foundation.config import QuotaRule, TierRule
from horasis.foundation.errors import NoBackendError, ValidationError
from horasis.foundation.schemas import DetectionResult, Image, Task
from tests.conftest import FakeBackend, png_bytes


def _client(*, inputs: InputPolicy | None = None, backends: list[FakeBackend] | None = None) -> Client:
    config = Config(
        default_backend="fake",
        tiers={"pro": TierRule(name="pro", quota=QuotaRule(limit=100, window_seconds=60))},
        inputs=inputs or InputPolicy(),
    )
    validator = InMemoryApiKeyValidator()
    validator.register_simple("k", principal_id="u", tier=Tier.PRO)
    registry = BackendRegistry()
    for b in backends or [FakeBackend()]:
        registry.register(b)
    return Client(config=config, registry=registry, validator=validator)


def test_pil_and_numpy_inputs_reach_the_backend() -> None:
    client = _client()
    for image in (PILImage.new("RGB", (6, 4)), np.zeros((4, 6, 3), dtype=np.uint8)):
        assert isinstance(client.detection.detect(image, api_key="k"), DetectionResult)


def test_urls_are_rejected_unless_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    from horasis.foundation import imaging

    monkeypatch.setattr(imaging, "fetch_url", lambda url, **_: png_bytes())
    with pytest.raises(ValidationError, match="disabled"):
        _client().detection.detect("https://example.com/a.png", api_key="k")
    opted_in = _client(inputs=InputPolicy(allow_urls=True))
    assert isinstance(
        opted_in.detection.detect("https://example.com/a.png", api_key="k"), DetectionResult
    )


def test_policy_limits_apply_at_the_api() -> None:
    client = _client(inputs=InputPolicy(max_pixels=10))
    with pytest.raises(ValidationError, match="pixels"):
        client.detection.detect(Image.from_bytes(png_bytes(8, 6)), api_key="k")


def test_garbage_bytes_are_a_validation_error_not_a_backend_error() -> None:
    with pytest.raises(ValidationError):
        _client().detection.detect(b"junk", api_key="k")


def test_unsupported_task_gives_clear_error() -> None:
    client = _client(backends=[FakeBackend("det", frozenset({Task.DETECT}))])
    with pytest.raises(NoBackendError, match="'ocr'"):
        client.ocr.extract_text(png_bytes(), api_key="k")


def test_build_registers_extra_backends() -> None:
    extra = FakeBackend("extra")
    client = Client.build(backends=[extra])
    assert {"ultralytics", "extra"} <= set(client.registry.names())
