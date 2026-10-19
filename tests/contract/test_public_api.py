"""Public API contract, independent of the backend serving the request."""

from __future__ import annotations

import pytest

from horasis.access.keys import InMemoryApiKeyValidator
from horasis.api.client import (
    Client,
    ContentModerator,
    FaceDetector,
    ImageClassifier,
    ObjectDetector,
    TextExtractor,
)
from horasis.backend.registry import BackendRegistry
from horasis.foundation.config import Config
from horasis.foundation.errors import ValidationError
from horasis.foundation.schemas import (
    ClassificationResult,
    DetectionResult,
    FaceResult,
    ModerationResult,
    OcrResult,
    Task,
)
from tests.conftest import png_bytes


@pytest.fixture
def client(config: Config, registry: BackendRegistry, validator: InMemoryApiKeyValidator) -> Client:
    return Client(config=config, registry=registry, validator=validator)


def test_client_exposes_five_capabilities(client: Client) -> None:
    assert isinstance(client.detection, ObjectDetector)
    assert isinstance(client.classification, ImageClassifier)
    assert isinstance(client.ocr, TextExtractor)
    assert isinstance(client.face, FaceDetector)
    assert isinstance(client.moderation, ContentModerator)


def test_detect_returns_detection_result(client: Client, sample_image) -> None:
    result = client.detection.detect(sample_image, api_key="pro-key")
    assert isinstance(result, DetectionResult)
    assert result.task is Task.DETECT


def test_classify_returns_classification_result(client: Client, sample_image) -> None:
    result = client.classification.classify(sample_image, api_key="pro-key")
    assert isinstance(result, ClassificationResult)


def test_ocr_returns_ocr_result(client: Client, sample_image) -> None:
    result = client.ocr.extract_text(sample_image, api_key="pro-key")
    assert isinstance(result, OcrResult)


def test_face_returns_face_result(client: Client, sample_image) -> None:
    result = client.face.detect_faces(sample_image, api_key="pro-key")
    assert isinstance(result, FaceResult)


def test_moderation_returns_moderation_result(client: Client, sample_image) -> None:
    result = client.moderation.moderate(sample_image, api_key="pro-key")
    assert isinstance(result, ModerationResult)


def test_free_tier_cannot_reach_ocr(client: Client, sample_image) -> None:
    # the `config` fixture only allows detect/classify for the free tier
    from horasis.foundation.errors import AccessDeniedError

    with pytest.raises(AccessDeniedError):
        client.ocr.extract_text(sample_image, api_key="free-key")


def test_public_api_never_leaks_backend_types(client: Client, sample_image) -> None:
    result = client.detection.detect(sample_image, api_key="pro-key")
    assert not hasattr(result, "infer")  # not a Backend
    assert result.backend == "fake"  # backend identity is just a string field


def test_capability_accepts_path_bytes_and_image(client: Client, tmp_path, sample_image) -> None:
    img_path = tmp_path / "image.bin"
    img_path.write_bytes(png_bytes())

    for image_input in (str(img_path), img_path, png_bytes(), sample_image):
        result = client.detection.detect(image_input, api_key="pro-key")
        assert isinstance(result, DetectionResult)


def test_capability_rejects_unsupported_image_type(client: Client) -> None:
    with pytest.raises(ValidationError):
        client.detection.detect(12345, api_key="pro-key")  # type: ignore[arg-type]
