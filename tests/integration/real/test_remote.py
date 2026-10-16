from __future__ import annotations

import pytest

from horasis import Skopos
from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.klimax import Klimax
from horasis.backend.remote_adapter import RemoteAdapter
from horasis.backend.taxis import Taxis
from horasis.backend.ultralytics_adapter import UltralyticsAdapter
from horasis.foundation.kanon import Kanon
from horasis.foundation.schemas import DetectionResult

from .common import KEY, assert_box_in_bounds, assert_confidence, image_size, sample_image, weights

pytestmark = pytest.mark.integration
pytest.importorskip("ultralytics")
pytest.importorskip("fastapi", reason="needs horasis[server]")
pytest.importorskip("httpx", reason="needs horasis[remote]")


def _validator(principal: str) -> InMemoryApiKeyValidator:
    validator = InMemoryApiKeyValidator()
    validator.register_simple(KEY, principal_id=principal, tier=Klimax.ENTERPRISE)
    return validator


def test_remote_detect_through_in_process_server_with_real_backend() -> None:
    from fastapi.testclient import TestClient

    from horasis.server import create_app

    server_taxis = Taxis()
    server_taxis.register(UltralyticsAdapter(detect_model=weights("yolo11n.pt")))
    server = Skopos(kanon=Kanon.default(), taxis=server_taxis, validator=_validator("server"))
    http = TestClient(create_app(server))

    caller_taxis = Taxis()
    caller_taxis.register(
        RemoteAdapter(endpoint_url="http://testserver", api_key=KEY, client=http, max_retries=0)
    )
    caller = Skopos(
        kanon=Kanon.default().model_copy(update={"default_backend": "remote"}),
        taxis=caller_taxis,
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
