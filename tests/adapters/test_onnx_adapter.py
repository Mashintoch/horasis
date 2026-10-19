from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from horasis.backend.onnx_adapter import (
    OnnxAdapter,
    center_crop_resize,
    letterbox,
    load_labels,
    nms,
)
from horasis.foundation.errors import (
    AdapterError,
    BackendUnavailableError,
    ConfigurationError,
)
from horasis.foundation.schemas import (
    ClassificationResult,
    DetectionResult,
    Image,
    Task,
)
from tests.conftest import png_bytes

# A 128x64 (WxH) image fed to a 64x64 model: scale 0.5, pad_x 0, pad_y 16.
IMAGE = Image.from_bytes(png_bytes(128, 64))


class FakeSession:
    def __init__(self, output: np.ndarray, shape: list[Any] | None = None) -> None:
        self._output = output
        self._shape = shape if shape is not None else [1, 3, 64, 64]
        self.feeds: dict[str, np.ndarray] = {}

    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name="images", shape=self._shape, type="tensor(float)")]

    def run(self, names: Any, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds = feeds
        return [self._output]


def _yolov8_output(columns: list[tuple[float, float, float, float, list[float]]]) -> np.ndarray:
    """Build a (1, 4+nc, N) tensor from (cx, cy, w, h, class_scores) columns."""
    arr = np.array([[cx, cy, w, h, *scores] for cx, cy, w, h, scores in columns], dtype=np.float32)
    return arr.T[None]


def _adapter(output: np.ndarray, **kwargs: Any) -> OnnxAdapter:
    kwargs.setdefault("output_format", "yolov8")
    return OnnxAdapter(
        model_path="m.onnx",
        labels=["cat", "dog"],
        session=FakeSession(output),
        **kwargs,
    )


def test_letterbox_geometry() -> None:
    from PIL import Image as PILImage

    tensor, scale, pad = letterbox(PILImage.new("RGB", (128, 64), (255, 255, 255)), (64, 64))
    assert tensor.shape == (3, 64, 64) and tensor.dtype == np.float32
    assert scale == 0.5 and pad == (0.0, 16.0)
    assert tensor[0, 32, 32] == 1.0  # image area
    assert tensor[0, 2, 32] == pytest.approx(114 / 255)  # padding


def test_center_crop_resize_shape() -> None:
    from PIL import Image as PILImage

    assert center_crop_resize(PILImage.new("RGB", (100, 50)), (32, 32)).shape == (3, 32, 32)


def test_nms_suppresses_overlaps_keeps_best() -> None:
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    assert nms(boxes, np.array([0.7, 0.9, 0.8], dtype=np.float32), 0.5) == [1, 2]


def test_detect_decodes_yolov8_with_nms_and_unletterbox() -> None:
    out = _yolov8_output(
        [
            (32, 32, 32, 16, [0.9, 0.0]),  # cat -> orig (32, 16, 96, 48)
            (33, 32, 32, 16, [0.8, 0.0]),  # same class, overlaps: suppressed
            (33, 32, 32, 16, [0.0, 0.6]),  # other class, same place: kept
            (10, 10, 4, 4, [0.1, 0.1]),  # below threshold
        ]
    )
    adapter = _adapter(out)
    result = adapter.infer(Task.DETECT, IMAGE)
    assert isinstance(result, DetectionResult)
    assert [d.label for d in result.detections] == ["cat", "dog"]
    box = result.detections[0].box
    assert (box.x_min, box.y_min, box.x_max, box.y_max) == (32, 16, 96, 48)
    assert float(result.detections[0].confidence) == pytest.approx(0.9)
    feed = adapter._session.feeds["images"]
    assert feed.shape == (1, 3, 64, 64) and feed.dtype == np.float32


def test_detect_clips_boxes_to_image_and_drops_empty() -> None:
    out = _yolov8_output(
        [
            (0, 32, 40, 16, [0.9, 0.0]),  # hangs off the left edge
            (-50, -50, 2, 2, [0.9, 0.0]),  # entirely outside: degenerate after clip
        ]
    )
    result = _adapter(out).infer(Task.DETECT, IMAGE)
    assert len(result.detections) == 1  # type: ignore[attr-defined]
    assert result.detections[0].box.x_min == 0  # type: ignore[attr-defined]


def test_detect_threshold_and_max_detections_options() -> None:
    out = _yolov8_output([(10 + 20 * i, 32, 8, 8, [0.5 + i / 100, 0.0]) for i in range(3)])
    adapter = _adapter(out, max_detections=2)
    assert len(adapter.infer(Task.DETECT, IMAGE).detections) == 2  # type: ignore[attr-defined]
    strict = adapter.infer(Task.DETECT, IMAGE, conf_threshold=0.515)
    assert len(strict.detections) == 1  # type: ignore[attr-defined]


def test_detect_yolov5_and_end_to_end_layouts() -> None:
    # YOLOv5: (1, N, 5+nc) with objectness; score = obj * class score.
    v5 = np.array([[[32, 32, 32, 16, 0.5, 0.0, 1.0], [5, 5, 2, 2, 0.1, 1.0, 0.0]]], dtype=np.float32)
    result = _adapter(v5, output_format="yolov5").infer(Task.DETECT, IMAGE)
    assert [(d.label, round(float(d.confidence), 2)) for d in result.detections] == [("dog", 0.5)]  # type: ignore[attr-defined]
    # End-to-end (1, N, 6): [x1, y1, x2, y2, score, cls] in letterboxed space.
    e2e = np.array([[[16, 24, 48, 40, 0.9, 1.0], [0, 0, 5, 5, 0.1, 0.0]]], dtype=np.float32)
    result = _adapter(e2e, output_format="nms").infer(Task.DETECT, IMAGE)
    (det,) = result.detections  # type: ignore[attr-defined]
    assert det.label == "dog" and (det.box.x_min, det.box.x_max) == (32, 96)


def test_auto_format_uses_class_count_then_shape() -> None:
    cols = [(32, 32, 32, 16, [0.9, 0.0])]
    few = _yolov8_output(cols)  # (1, 6, 1): fewer anchors than channels
    sized = _adapter(few, output_format="auto")  # 2 labels -> 4 + 2 == 6 channels
    assert [d.label for d in sized.infer(Task.DETECT, IMAGE).detections] == ["cat"]  # type: ignore[attr-defined]
    many = np.zeros((1, 85, 2000), dtype=np.float32)  # no labels: longer axis wins
    unlabeled = OnnxAdapter(model_path="m.onnx", session=FakeSession(many), labels=[])
    assert unlabeled.infer(Task.DETECT, IMAGE).detections == []  # type: ignore[attr-defined]


def test_auto_format_picks_nms_for_six_column_rows() -> None:
    rows = np.zeros((1, 300, 6), dtype=np.float32)
    rows[0, 0] = [16, 24, 48, 40, 0.9, 0]
    result = _adapter(rows, output_format="auto").infer(Task.DETECT, IMAGE)
    assert [d.label for d in result.detections] == ["cat"]  # type: ignore[attr-defined]


def test_classify_applies_softmax_and_imagenet_norm() -> None:
    adapter = OnnxAdapter(
        model_path="c.onnx",
        task="classify",
        labels=["a", "b", "c"],
        session=FakeSession(np.array([[1.0, 3.0, 2.0]], dtype=np.float32), [1, 3, 32, 32]),
        top_k=2,
    )
    result = adapter.infer(Task.CLASSIFY, IMAGE)
    assert isinstance(result, ClassificationResult)
    assert [e.label for e in result.entries] == ["b", "c"]
    assert float(result.entries[0].confidence) == pytest.approx(0.6652, abs=1e-3)
    feed = adapter._session.feeds["images"]
    assert feed.shape == (1, 3, 32, 32)
    assert feed.min() < 0  # ImageNet normalization pushes values negative


def test_classify_keeps_existing_probabilities() -> None:
    adapter = OnnxAdapter(
        model_path="c.onnx",
        task="classify",
        labels=["a", "b"],
        session=FakeSession(np.array([[0.25, 0.75]], dtype=np.float32), [1, 3, 8, 8]),
    )
    assert float(adapter.infer(Task.CLASSIFY, IMAGE).entries[0].confidence) == 0.75  # type: ignore[attr-defined]


def test_task_scopes_supported_tasks() -> None:
    det = _adapter(np.zeros((1, 6, 4), dtype=np.float32))
    assert det.supports(Task.DETECT) and not det.supports(Task.CLASSIFY)
    with pytest.raises(BackendUnavailableError):
        det.infer(Task.CLASSIFY, IMAGE)
    with pytest.raises(ConfigurationError):
        OnnxAdapter(model_path="x.onnx", task="segment")  # type: ignore[arg-type]


def test_dynamic_input_uses_configured_size() -> None:
    session = FakeSession(np.zeros((1, 6, 4), dtype=np.float32), [1, 3, "h", "w"])
    OnnxAdapter(
        model_path="m.onnx", session=session, input_size=(48, 96), output_format="yolov8"
    ).infer(
        Task.DETECT, IMAGE
    )
    assert session.feeds["images"].shape == (1, 3, 48, 96)


def test_non_nchw_model_is_a_configuration_error() -> None:
    session = FakeSession(np.zeros((1, 6, 4), dtype=np.float32), [1, 64, 64, 3])
    with pytest.raises(ConfigurationError, match="NCHW"):
        OnnxAdapter(model_path="m.onnx", session=session).infer(Task.DETECT, IMAGE)


def test_runtime_failures_are_wrapped() -> None:
    class Broken(FakeSession):
        def run(self, names: Any, feeds: Any) -> Any:
            raise RuntimeError("ort exploded")

    with pytest.raises(AdapterError, match="ort exploded"):
        OnnxAdapter(model_path="m.onnx", session=Broken(np.zeros(1))).infer(Task.DETECT, IMAGE)


# Labels sidecar


def test_labels_txt_sidecar_discovered(tmp_path: Path) -> None:
    (tmp_path / "m.labels.txt").write_text("person\n\nbicycle\n", encoding="utf-8")
    assert load_labels(tmp_path / "m.onnx") == ["person", "bicycle"]


def test_labels_json_list_and_dict(tmp_path: Path) -> None:
    (tmp_path / "a.labels.json").write_text(json.dumps(["x", "y"]))
    assert load_labels(tmp_path / "a.onnx") == ["x", "y"]
    (tmp_path / "b.labels.json").write_text(json.dumps({"0": "zero", "2": "two"}))
    assert load_labels(tmp_path / "b.onnx") == ["zero", "1", "two"]


def test_labels_missing_or_bad(tmp_path: Path) -> None:
    assert load_labels(tmp_path / "none.onnx") == []
    with pytest.raises(ConfigurationError, match="not found"):
        load_labels(tmp_path / "m.onnx", tmp_path / "ghost.txt")
    bad = tmp_path / "bad.labels.json"
    bad.write_text("{not json")
    with pytest.raises(ConfigurationError, match="parse"):
        load_labels(tmp_path / "bad.onnx")


def test_adapter_uses_sidecar_and_falls_back_to_ids(tmp_path: Path) -> None:
    model = tmp_path / "m.onnx"
    (tmp_path / "m.labels.txt").write_text("cat\n", encoding="utf-8")
    out = _yolov8_output([(32, 32, 32, 16, [0.9, 0.0]), (32, 32, 8, 8, [0.0, 0.8])])
    adapter = OnnxAdapter(model_path=model, session=FakeSession(out), output_format="yolov8")
    assert [d.label for d in adapter.infer(Task.DETECT, IMAGE).detections] == ["cat", "1"]  # type: ignore[attr-defined]


# Providers and session creation


@pytest.fixture
def fake_ort(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    created: list[list[str]] = []

    class Session:
        def __init__(self, path: str, providers: list[str]) -> None:
            created.append(providers)

    ort = SimpleNamespace(
        available=["CPUExecutionProvider"],
        created=created,
        InferenceSession=Session,
    )
    ort.get_available_providers = lambda: ort.available
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    return ort


def _model_file(tmp_path: Path) -> Path:
    path = tmp_path / "m.onnx"
    path.write_bytes(b"")
    return path


def test_provider_selection(fake_ort: SimpleNamespace, tmp_path: Path) -> None:
    model = _model_file(tmp_path)
    OnnxAdapter(model_path=model).warm_up()
    assert fake_ort.created[-1] == ["CPUExecutionProvider"]

    fake_ort.available = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    OnnxAdapter(model_path=model).warm_up()
    assert fake_ort.created[-1] == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    OnnxAdapter(model_path=model, device="cpu").warm_up()
    assert fake_ort.created[-1] == ["CPUExecutionProvider"]
    OnnxAdapter(model_path=model, providers=["TensorrtExecutionProvider"]).warm_up()
    assert fake_ort.created[-1] == ["TensorrtExecutionProvider"]


def test_explicit_cuda_without_cuda_fails_clearly(fake_ort: SimpleNamespace, tmp_path: Path) -> None:
    with pytest.raises(BackendUnavailableError, match="CUDA"):
        OnnxAdapter(model_path=_model_file(tmp_path), device="cuda").warm_up()


def test_missing_model_file_and_missing_runtime(
    fake_ort: SimpleNamespace, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(BackendUnavailableError, match="not found"):
        OnnxAdapter(model_path=tmp_path / "ghost.onnx").warm_up()
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(BackendUnavailableError, match="horasis\\[onnx\\]"):
        OnnxAdapter(model_path=_model_file(tmp_path)).warm_up()
