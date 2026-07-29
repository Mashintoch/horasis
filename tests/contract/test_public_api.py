"""Contract tests: assert the public API boundary's shape and guarantees,
independent of which backend actually serves a request.
"""

from __future__ import annotations

import pytest

from horasis.access.keys import InMemoryApiKeyValidator
from horasis.api.skopos import (
    ContentModerator,
    FaceDetector,
    ImageClassifier,
    ObjectDetector,
    Skopos,
    TextExtractor,
)
from horasis.backend.taxis import Taxis
from horasis.foundation.kanon import Kanon
from horasis.foundation.schemas import (
    ClassificationResult,
    DetectionResult,
    FaceResult,
    ModerationResult,
    OcrResult,
    Praxis,
)
from horasis.foundation.sphalma import ValidationError


@pytest.fixture
def skopos(kanon: Kanon, taxis: Taxis, validator: InMemoryApiKeyValidator) -> Skopos:
    return Skopos(kanon=kanon, taxis=taxis, validator=validator)


def test_skopos_exposes_five_capabilities(skopos: Skopos) -> None:
    assert isinstance(skopos.detection, ObjectDetector)
    assert isinstance(skopos.classification, ImageClassifier)
    assert isinstance(skopos.ocr, TextExtractor)
    assert isinstance(skopos.face, FaceDetector)
    assert isinstance(skopos.moderation, ContentModerator)


def test_detect_returns_detection_result(skopos: Skopos, sample_image) -> None:
    result = skopos.detection.detect(sample_image, api_key="pro-key")
    assert isinstance(result, DetectionResult)
    assert result.praxis is Praxis.DETECT


def test_classify_returns_classification_result(skopos: Skopos, sample_image) -> None:
    result = skopos.classification.classify(sample_image, api_key="pro-key")
    assert isinstance(result, ClassificationResult)


def test_ocr_returns_ocr_result(skopos: Skopos, sample_image) -> None:
    result = skopos.ocr.extract_text(sample_image, api_key="pro-key")
    assert isinstance(result, OcrResult)


def test_face_returns_face_result(skopos: Skopos, sample_image) -> None:
    result = skopos.face.detect_faces(sample_image, api_key="pro-key")
    assert isinstance(result, FaceResult)


def test_moderation_returns_moderation_result(skopos: Skopos, sample_image) -> None:
    result = skopos.moderation.moderate(sample_image, api_key="pro-key")
    assert isinstance(result, ModerationResult)


def test_free_tier_cannot_reach_ocr(skopos: Skopos, sample_image) -> None:
    # the `kanon` fixture only allows detect/classify for the free tier
    from horasis.foundation.sphalma import AccessDeniedError

    with pytest.raises(AccessDeniedError):
        skopos.ocr.extract_text(sample_image, api_key="free-key")


def test_public_api_never_leaks_backend_types(skopos: Skopos, sample_image) -> None:
    """Callers get `Theoria` subclasses back, never a raw `Mechane` or backend object."""
    result = skopos.detection.detect(sample_image, api_key="pro-key")
    assert not hasattr(result, "infer")  # not a Mechane
    assert result.backend == "fake"  # backend identity is just a string field


def test_capability_accepts_path_bytes_and_physis(skopos: Skopos, tmp_path, sample_image) -> None:
    img_path = tmp_path / "image.bin"
    img_path.write_bytes(b"fake-bytes")

    for image_input in (str(img_path), img_path, b"raw-bytes", sample_image):
        result = skopos.detection.detect(image_input, api_key="pro-key")
        assert isinstance(result, DetectionResult)


def test_capability_rejects_unsupported_image_type(skopos: Skopos) -> None:
    with pytest.raises(ValidationError):
        skopos.detection.detect(12345, api_key="pro-key")  # type: ignore[arg-type]
