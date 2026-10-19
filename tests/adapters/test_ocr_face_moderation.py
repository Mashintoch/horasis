from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from horasis.backend._common import order_into_lines
from horasis.backend.easyocr_adapter import EasyOCRAdapter
from horasis.backend.face_adapter import MODEL_ENV_VAR, YuNetFaceAdapter
from horasis.backend.moderation_adapter import TransformersModerationAdapter
from horasis.backend.registry import BackendRegistry
from horasis.foundation.config import Config
from horasis.foundation.errors import AdapterError, BackendUnavailableError
from horasis.foundation.schemas import (
    BoundingBox,
    FaceResult,
    Image,
    ModerationResult,
    OcrResult,
    Task,
)
from horasis.orchestration.pipeline import Pipeline
from tests.conftest import png_bytes

IMAGE = Image.from_bytes(png_bytes(100, 60))


def _quad(x: float, y: float, w: float = 30, h: float = 10) -> list[list[float]]:
    return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]


class FakeReader:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.seen: Any = None

    def readtext(self, image: Any) -> list[Any]:
        self.seen = image
        return self.rows


# OCR


def test_easyocr_returns_regions_with_polygon_and_lines() -> None:
    reader = FakeReader(
        [
            (_quad(60, 40), "world", 0.8),  # second line, listed first
            (_quad(50, 5), "Hello", 0.9),  # first line, right word
            (_quad(10, 6), "Oh", 0.99),  # first line, left word (slightly lower)
            (_quad(5, 5), "   ", 0.99),  # blank: dropped
        ]
    )
    result = EasyOCRAdapter(reader=reader).infer(Task.OCR, IMAGE)
    assert isinstance(result, OcrResult)
    assert result.backend == "easyocr"
    assert [g.text for g in result.regions] == ["Oh", "Hello", "world"]
    assert [g.line_index for g in result.regions] == [0, 0, 1]
    first = result.regions[0]
    assert first.polygon == [(10.0, 6.0), (40.0, 6.0), (40.0, 16.0), (10.0, 16.0)]
    assert first.box == BoundingBox(x_min=10, y_min=6, x_max=40, y_max=16)
    assert [line.text for line in result.lines] == ["Oh Hello", "world"]
    assert result.text == "Oh Hello\nworld"
    assert isinstance(reader.seen, np.ndarray) and reader.seen.shape == (60, 100, 3)


def test_easyocr_min_confidence_and_clamping() -> None:
    reader = FakeReader([(_quad(0, 0), "low", 0.1), (_quad(0, 20), "ok", 1.0000001)])
    result = EasyOCRAdapter(reader=reader, min_confidence=0.5).infer(Task.OCR, IMAGE)
    assert [g.text for g in result.regions] == ["ok"]
    assert float(result.regions[0].confidence) == 1.0


def test_easyocr_wraps_engine_failures_and_rejects_other_tasks() -> None:
    class Boom:
        def readtext(self, image: Any) -> None:
            raise RuntimeError("kaboom")

    with pytest.raises(AdapterError, match="kaboom"):
        EasyOCRAdapter(reader=Boom()).infer(Task.OCR, IMAGE)
    with pytest.raises(BackendUnavailableError):
        EasyOCRAdapter(reader=FakeReader([])).infer(Task.DETECT, IMAGE)


def test_easyocr_missing_dependency_is_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "easyocr", None)
    with pytest.raises(BackendUnavailableError, match="horasis\\[ocr\\]"):
        EasyOCRAdapter().infer(Task.OCR, IMAGE)


def test_tesseract_adapter_maps_words_and_native_lines() -> None:
    from horasis.backend.tesseract_adapter import TesseractAdapter

    data = {
        "text": ["", "Hello", "there", "bye", "noise"],
        "conf": ["-1", "96.5", "88", "70", "-1"],
        "block_num": [1, 1, 1, 2, 2],
        "par_num": [1, 1, 1, 1, 1],
        "line_num": [0, 1, 1, 1, 1],
        "left": [0, 5, 50, 5, 0],
        "top": [0, 5, 5, 30, 0],
        "width": [0, 40, 40, 20, 0],
        "height": [0, 10, 10, 10, 0],
    }
    engine = SimpleNamespace(image_to_data=lambda img, lang, output_type: data)
    result = TesseractAdapter(engine=engine).infer(Task.OCR, IMAGE)
    assert [g.text for g in result.regions] == ["Hello", "there", "bye"]
    assert [g.line_index for g in result.regions] == [0, 0, 1]
    assert float(result.regions[0].confidence) == pytest.approx(0.965)
    assert result.regions[0].polygon is not None and len(result.regions[0].polygon) == 4
    assert result.text == "Hello there\nbye"


def test_tesseract_missing_binary_is_backend_unavailable() -> None:
    from horasis.backend.tesseract_adapter import TesseractAdapter

    class TesseractNotFoundError(Exception):
        pass

    def boom(*args: Any, **kwargs: Any) -> None:
        raise TesseractNotFoundError("tesseract is not installed")

    with pytest.raises(BackendUnavailableError, match="binary"):
        TesseractAdapter(engine=SimpleNamespace(image_to_data=boom)).infer(Task.OCR, IMAGE)


def test_order_into_lines_handles_empty_and_single() -> None:
    assert order_into_lines([], lambda b: b) == []
    box = BoundingBox(x_min=0, y_min=0, x_max=1, y_max=1)
    assert order_into_lines([box], lambda b: b) == [(0, box)]


# Faces


class FakeYuNet:
    def __init__(self, rows: Any) -> None:
        self.rows = rows
        self.input_size: Any = None
        self.seen: Any = None

    def setInputSize(self, size: tuple[int, int]) -> None:
        self.input_size = size

    def detect(self, bgr: Any) -> tuple[int, Any]:
        self.seen = bgr
        return 1, self.rows


def _row(x: float, y: float, w: float, h: float, score: float) -> list[float]:
    landmarks = [x + 1, y + 1, x + 2, y + 1, x + 1.5, y + 2, x + 1, y + 3, x + 2, y + 3]
    return [x, y, w, h, *landmarks, score]


def test_face_adapter_returns_boxes_landmarks_and_sets_input_size() -> None:
    det = FakeYuNet(np.array([_row(10, 12, 20, 24, 0.93)], dtype=np.float32))
    result = YuNetFaceAdapter(detector=det).infer(Task.FACE, IMAGE)
    assert isinstance(result, FaceResult)
    assert det.input_size == (100, 60)
    (face,) = result.faces
    assert face.label == "face"
    assert float(face.confidence) == pytest.approx(0.93, abs=1e-6)
    assert (face.box.x_min, face.box.y_min, face.box.x_max, face.box.y_max) == (10, 12, 30, 36)
    assert face.landmarks is not None
    assert set(face.landmarks) == {"right_eye", "left_eye", "nose", "mouth_right", "mouth_left"}
    assert face.landmarks["nose"] == pytest.approx((11.5, 14.0))


def test_face_adapter_passes_bgr_to_opencv() -> None:
    det = FakeYuNet(None)
    rgb_red = Image.from_bytes(png_bytes(4, 4, color=(255, 0, 0)))
    YuNetFaceAdapter(detector=det).infer(Task.FACE, rgb_red)
    assert tuple(det.seen[0, 0]) == (0, 0, 255)  # BGR order


def test_face_adapter_no_faces_and_optional_landmarks() -> None:
    assert YuNetFaceAdapter(detector=FakeYuNet(None)).infer(Task.FACE, IMAGE).faces == []
    det = FakeYuNet(np.array([_row(1, 1, 5, 5, 0.9)], dtype=np.float32))
    result = YuNetFaceAdapter(detector=det, landmarks=False).infer(Task.FACE, IMAGE)
    assert result.faces[0].landmarks is None


def test_face_adapter_missing_model_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    with pytest.raises(BackendUnavailableError, match="YuNet"):
        YuNetFaceAdapter().infer(Task.FACE, IMAGE)
    with pytest.raises(BackendUnavailableError, match="YuNet"):
        YuNetFaceAdapter(model_path="/no/such/model.onnx").infer(Task.FACE, IMAGE)


# Moderation


def _classifier(scores: dict[str, float]) -> Any:
    def run(image: Any, top_k: Any = None) -> list[dict[str, Any]]:
        return [{"label": k, "score": v} for k, v in scores.items()]

    return run


def test_moderation_default_thresholds_and_safe_categories() -> None:
    adapter = TransformersModerationAdapter(classifier=_classifier({"normal": 0.97, "nsfw": 0.03}))
    result = adapter.infer(Task.MODERATE, IMAGE)
    assert isinstance(result, ModerationResult)
    assert not result.is_flagged
    flagged = TransformersModerationAdapter(
        classifier=_classifier({"normal": 0.2, "nsfw": 0.8})
    ).infer(Task.MODERATE, IMAGE)
    assert flagged.is_flagged
    assert {lb.category: lb.flagged for lb in flagged.labels} == {"normal": False, "nsfw": True}


def test_moderation_threshold_callable_is_used_standalone() -> None:
    adapter = TransformersModerationAdapter(
        classifier=_classifier({"nsfw": 0.6}), threshold_for=lambda c: 0.9
    )
    assert not adapter.infer(Task.MODERATE, IMAGE).is_flagged  # type: ignore[attr-defined]


def test_config_thresholds_are_authoritative_through_pipeline() -> None:
    backend = TransformersModerationAdapter(classifier=_classifier({"normal": 0.3, "nsfw": 0.7}))
    registry = BackendRegistry()
    registry.register(backend)

    def flagged_with(config: Config) -> dict[str, bool]:
        pipeline = Pipeline(config=config, registry=registry)
        result = pipeline.run(Task.MODERATE, IMAGE)
        assert isinstance(result, ModerationResult)
        return {lb.category: lb.flagged for lb in result.labels}

    strict = Config(default_backend="moderation", moderation_threshold=0.5)
    lenient = Config(default_backend="moderation", moderation_thresholds={"nsfw": 0.9})
    no_safe_list = Config(
        default_backend="moderation", moderation_safe_categories=[], moderation_threshold=0.2
    )
    assert flagged_with(strict) == {"normal": False, "nsfw": True}
    assert flagged_with(lenient) == {"normal": False, "nsfw": False}
    assert flagged_with(no_safe_list)["normal"] is True  # only the safe list protected it


def test_moderation_errors() -> None:
    def boom(image: Any, top_k: Any = None) -> None:
        raise RuntimeError("model exploded")

    with pytest.raises(AdapterError, match="exploded"):
        TransformersModerationAdapter(classifier=boom).infer(Task.MODERATE, IMAGE)
    with pytest.raises(BackendUnavailableError):
        TransformersModerationAdapter(classifier=boom).infer(Task.OCR, IMAGE)


def test_moderation_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "transformers", None)
    with pytest.raises(BackendUnavailableError, match="horasis\\[moderation\\]"):
        TransformersModerationAdapter().infer(Task.MODERATE, IMAGE)
