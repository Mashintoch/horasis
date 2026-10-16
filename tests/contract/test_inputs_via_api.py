from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from horasis import InputPolicy, Kanon, Skopos
from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.klimax import Klimax
from horasis.backend.taxis import Taxis
from horasis.foundation.kanon import QuotaRule, TierRule
from horasis.foundation.schemas import DetectionResult, Physis, Praxis
from horasis.foundation.sphalma import NoBackendError, ValidationError
from tests.conftest import FakeMechane, png_bytes


def _client(*, inputs: InputPolicy | None = None, backends: list[FakeMechane] | None = None) -> Skopos:
    kanon = Kanon(
        default_backend="fake",
        tiers={"pro": TierRule(name="pro", quota=QuotaRule(limit=100, window_seconds=60))},
        inputs=inputs or InputPolicy(),
    )
    validator = InMemoryApiKeyValidator()
    validator.register_simple("k", principal_id="u", tier=Klimax.PRO)
    taxis = Taxis()
    for b in backends or [FakeMechane()]:
        taxis.register(b)
    return Skopos(kanon=kanon, taxis=taxis, validator=validator)


def test_pil_and_numpy_inputs_reach_the_backend() -> None:
    client = _client()
    for image in (Image.new("RGB", (6, 4)), np.zeros((4, 6, 3), dtype=np.uint8)):
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
        client.detection.detect(Physis.from_bytes(png_bytes(8, 6)), api_key="k")


def test_garbage_bytes_are_a_validation_error_not_a_backend_error() -> None:
    with pytest.raises(ValidationError):
        _client().detection.detect(b"junk", api_key="k")


def test_unsupported_praxis_gives_clear_error() -> None:
    client = _client(backends=[FakeMechane("det", frozenset({Praxis.DETECT}))])
    with pytest.raises(NoBackendError, match="'ocr'"):
        client.ocr.extract_text(png_bytes(), api_key="k")


def test_build_registers_extra_backends() -> None:
    extra = FakeMechane("extra")
    client = Skopos.build(backends=[extra])
    assert {"ultralytics", "extra"} <= set(client.taxis.names())
