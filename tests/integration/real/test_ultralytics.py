from __future__ import annotations

import pytest

from horasis import Skopos
from horasis.backend.ultralytics_adapter import UltralyticsAdapter
from horasis.foundation.schemas import (
    ClassificationResult,
    DetectionResult,
    PoseResult,
    SegmentationResult,
)

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
pytest.importorskip("ultralytics")


@pytest.fixture(scope="module")
def client() -> Skopos:
    skopos = make_client()
    skopos.taxis.register(
        UltralyticsAdapter(
            detect_model=weights("yolo11n.pt"),
            classify_model=weights("yolo11n-cls.pt"),
            segment_model=weights("yolo11n-seg.pt"),
            pose_model=weights("yolo11n-pose.pt"),
        )
    )
    return skopos


def test_detect_bus(client: Skopos) -> None:
    path = sample_image("bus.jpg")
    result = client.detection.detect(path, api_key=KEY)
    assert isinstance(result, DetectionResult)
    assert {"bus", "person"} <= {d.label for d in result.detections}
    size = image_size(path)
    for det in result.detections:
        assert_is_name(det.label)
        assert_confidence(det.confidence)
        assert_box_in_bounds(det.box, size)


def test_classify(client: Skopos) -> None:
    result = client.classification.classify(sample_image("bus.jpg"), api_key=KEY)
    assert isinstance(result, ClassificationResult)
    assert result.entries
    for entry in result.entries:
        assert_is_name(entry.label)
        assert_confidence(entry.confidence)
    scores = [e.confidence.value for e in result.entries]
    assert scores == sorted(scores, reverse=True)


def test_segment_bus(client: Skopos) -> None:
    path = sample_image("bus.jpg")
    result = client.segmentation.segment(path, api_key=KEY)
    assert isinstance(result, SegmentationResult)
    assert "bus" in {s.label for s in result.segments}
    width, height = image_size(path)
    for seg in result.segments:
        assert_is_name(seg.label)
        assert_confidence(seg.confidence)
        assert_box_in_bounds(seg.box, (width, height))
        assert len(seg.polygon) >= 3
        for x, y in seg.polygon:
            assert -1 <= x <= width + 1 and -1 <= y <= height + 1


def test_pose_zidane(client: Skopos) -> None:
    path = sample_image("zidane.jpg")
    result = client.pose.estimate(path, api_key=KEY)
    assert isinstance(result, PoseResult)
    assert result.poses
    size = image_size(path)
    for pose in result.poses:
        assert_box_in_bounds(pose.box, size)
        assert len(pose.keypoints) == 17
        assert {"nose", "left_shoulder", "right_knee"} <= {k.name for k in pose.keypoints}
        for kp in pose.keypoints:
            assert_confidence(kp.confidence)
