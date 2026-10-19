from __future__ import annotations

import pytest

from horasis import Client
from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.tiers import Tier
from horasis.backend.registry import BackendRegistry
from horasis.backend.remote_adapter import RemoteAdapter
from horasis.backend.ultralytics_adapter import UltralyticsAdapter
from horasis.foundation.config import Config
from horasis.foundation.schemas import DetectionResult

from .common import KEY, assert_box_in_bounds, assert_confidence, image_size, sample_image, weights

pytestmark = pytest.mark.integration
pytest.importorskip("ultralytics")
pytest.importorskip("fastapi", reason="needs horasis[server]")
pytest.importorskip("httpx", reason="needs horasis[remote]")


def _validator(principal: str) -> InMemoryApiKeyValidator:
    validator = InMemoryApiKeyValidator()
    validator.register_simple(KEY, principal_id=principal, tier=Tier.ENTERPRISE)
    return validator


def test_remote_detect_through_in_process_server_with_real_backend() -> None:
    from fastapi.testclient import TestClient

    from horasis.server import create_app

    server_registry = BackendRegistry()
    server_registry.register(UltralyticsAdapter(detect_model=weights("yolo11n.pt")))
    server = Client(config=Config.default(), registry=server_registry, validator=_validator("server"))
    http = TestClient(create_app(server))

    caller_registry = BackendRegistry()
    caller_registry.register(
        RemoteAdapter(endpoint_url="http://testserver", api_key=KEY, client=http, max_retries=0)
    )
    caller = Client(
        config=Config.default().model_copy(update={"default_backend": "remote"}),
        registry=caller_registry,
        validator=_validator("caller"),
    )

    path = sample_image("bus.jpg")
    result = caller.detection.detect(path, api_key=KEY)
    assert isinstance(result, DetectionResult)
    assert {"bus", "person"} <= {d.label for d in result.detections}
    size = image_size(path)
    for det in result.detections:
        assert_confidence(det.confidence)
        assert_box_in_bounds(det.box, size)
