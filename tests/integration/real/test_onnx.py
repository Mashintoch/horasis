from __future__ import annotations

import os
from pathlib import Path

import pytest

from horasis.backend.onnx_adapter import OnnxAdapter
from horasis.foundation.schemas import DetectionResult

from .common import (
    KEY,
    assert_box_in_bounds,
    assert_confidence,
    assert_is_name,
    image_size,
    make_client,
    sample_image,
    weights,
)

pytestmark = pytest.mark.integration
pytest.importorskip("onnxruntime", reason="needs onnxruntime (pip install 'horasis[onnx]')")


def _model() -> tuple[Path, list[str]]:
    """Return YOLO11n as ONNX: `$HORASIS_TEST_ONNX_MODEL`, else export the Ultralytics weights."""
    pytest.importorskip("ultralytics")
    from ultralytics import YOLO

    model = YOLO(weights("yolo11n.pt"))
    names = [str(model.names[i]) for i in sorted(model.names)]
    override = os.environ.get("HORASIS_TEST_ONNX_MODEL")
    if override:
        return Path(override), names
    pytest.importorskip("onnx", reason="exporting to ONNX needs the 'onnx' package")
    return Path(model.export(format="onnx", imgsz=640, verbose=False)), names


def test_onnx_detects_bus() -> None:
    path, names = _model()
    client = make_client(OnnxAdapter(model_path=path, task="detect", labels=names, input_size=640))
    image = sample_image("bus.jpg")
    result = client.detection.detect(image, api_key=KEY, backend="onnx")
    assert isinstance(result, DetectionResult)
    assert {"bus", "person"} <= {d.label for d in result.detections}
    size = image_size(image)
    for det in result.detections:
        assert_is_name(det.label)
        assert_confidence(det.confidence)
        assert_box_in_bounds(det.box, size)
