from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from PIL import Image

from horasis import Skopos
from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.klimax import Klimax
from horasis.backend.taxis import Taxis
from horasis.backend.ultralytics_adapter import UltralyticsAdapter
from horasis.foundation.kanon import Kanon, QuotaRule, TierRule
from horasis.foundation.schemas import (
    COCO_KEYPOINTS,
    Physis,
    PoseResult,
    Praxis,
    SegmentationResult,
)
from tests.conftest import FakeMechane, png_bytes

IMAGE = Physis.from_bytes(png_bytes(100, 80))


def _box(cls: int, conf: float, xyxy: list[float]) -> Any:
    return SimpleNamespace(
        cls=[cls], conf=[conf], xyxy=[np.array(xyxy, dtype=np.float32)]
    )


def _install_fake_yolo(monkeypatch: pytest.MonkeyPatch, adapter: UltralyticsAdapter, raw: Any):
    loaded: list[str] = []
    calls: list[dict[str, Any]] = []

    class FakeYOLO:
        def __init__(self, path: str) -> None:
            loaded.append(path)

        def __call__(self, source: Any, **kwargs: Any) -> list[Any]:
            calls.append({"source": source, **kwargs})
            return [raw]

    monkeypatch.setattr(adapter, "_load_ultralytics", lambda: FakeYOLO)
    return loaded, calls


def test_segmentation_returns_boxes_and_polygons(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = SimpleNamespace(
        names={0: "person", 1: "dog"},
        boxes=[_box(0, 0.9, [1, 2, 30, 40]), _box(1, 0.6, [50, 5, 90, 60])],
        masks=SimpleNamespace(
            xy=[np.array([[1, 2], [30, 2], [30, 40]]), np.array([[50, 5], [90, 5], [90, 60], [50, 60]])]
        ),
    )
    adapter = UltralyticsAdapter()
    loaded, calls = _install_fake_yolo(monkeypatch, adapter, raw)
    result = adapter.infer(Praxis.SEGMENT, IMAGE, conf=0.4)
    assert isinstance(result, SegmentationResult)
    assert loaded == ["yolo11n-seg.pt"]
    assert isinstance(calls[0]["source"], Image.Image) and calls[0]["conf"] == 0.4
    person, dog = result.segments
    assert person.label == "person" and len(person.polygon) == 3
    assert dog.polygon[2] == (90.0, 60.0) and dog.box.x_max == 90
    assert result.filter(labels=["dog"]).segments == [dog]


def test_segmentation_without_masks_and_model_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = SimpleNamespace(names={}, boxes=[], masks=None)
    adapter = UltralyticsAdapter(segment_model="custom-seg.pt")
    loaded, _ = _install_fake_yolo(monkeypatch, adapter, raw)
    assert adapter.infer(Praxis.SEGMENT, IMAGE).segments == []  # type: ignore[attr-defined]
    adapter.infer(Praxis.SEGMENT, IMAGE)
    assert loaded == ["custom-seg.pt"]
    assert adapter.is_loaded()
    adapter.unload()
    assert not adapter.is_loaded()


def test_pose_returns_named_coco_keypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    xy = np.arange(34, dtype=np.float32).reshape(1, 17, 2)
    conf = np.full((1, 17), 0.8, dtype=np.float32)
    conf[0, 0] = 0.25
    raw = SimpleNamespace(
        names={0: "person"},
        boxes=[_box(0, 0.95, [10, 10, 60, 70])],
        keypoints=SimpleNamespace(xy=xy, conf=conf),
    )
    adapter = UltralyticsAdapter()
    loaded, _ = _install_fake_yolo(monkeypatch, adapter, raw)
    result = adapter.infer(Praxis.POSE, IMAGE)
    assert isinstance(result, PoseResult) and loaded == ["yolo11n-pose.pt"]
    (person,) = result.poses
    assert [k.name for k in person.keypoints] == list(COCO_KEYPOINTS)
    assert (person.keypoints[1].x, person.keypoints[1].y) == (2.0, 3.0)
    assert float(person.keypoints[0].confidence) == pytest.approx(0.25)


def test_pose_with_unusual_keypoint_count_and_no_keypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = SimpleNamespace(
        names={0: "hand"},
        boxes=[_box(0, 0.9, [0, 0, 5, 5])],
        keypoints=SimpleNamespace(xy=np.zeros((1, 21, 2)), conf=None),
    )
    adapter = UltralyticsAdapter()
    _install_fake_yolo(monkeypatch, adapter, raw)
    pose = adapter.infer(Praxis.POSE, IMAGE).poses[0]  # type: ignore[attr-defined]
    assert pose.keypoints[3].name == "kp3" and float(pose.keypoints[3].confidence) == 1.0
    empty = SimpleNamespace(names={0: "p"}, boxes=[_box(0, 0.9, [0, 0, 5, 5])], keypoints=None)
    adapter2 = UltralyticsAdapter()
    _install_fake_yolo(monkeypatch, adapter2, empty)
    assert adapter2.infer(Praxis.POSE, IMAGE).poses[0].keypoints == []  # type: ignore[attr-defined]


def test_new_praxeis_declared() -> None:
    adapter = UltralyticsAdapter()
    assert adapter.supports(Praxis.SEGMENT) and adapter.supports(Praxis.POSE)
    assert not adapter.supports_video(Praxis.SEGMENT)


def test_draw_segmentation_and_pose() -> None:
    from horasis.foundation.schemas import (
        BoundingBox,
        Doxa,
        Keypoint,
        Pose,
        Segment,
    )

    box = BoundingBox(x_min=5, y_min=5, x_max=40, y_max=40)
    seg = SegmentationResult(
        backend="b",
        latency_ms=0,
        segments=[
            Segment(label="a", confidence=Doxa(value=0.9), box=box, polygon=[(5, 5), (40, 5), (20, 40)]),
            Segment(label="b", confidence=Doxa(value=0.9), box=box),  # no polygon: box fallback
        ],
    )
    blank = Image.new("RGB", (50, 50))
    assert np.asarray(seg.draw(blank)).sum() > 0
    kps = [
        Keypoint(name=n, x=5 + i, y=5 + 2 * i, confidence=Doxa(value=0.9))
        for i, n in enumerate(COCO_KEYPOINTS)
    ]
    pose = PoseResult(
        backend="b", latency_ms=0, poses=[Pose(label="person", confidence=Doxa(value=0.9), box=box, keypoints=kps)]
    )
    assert np.asarray(pose.draw(blank)).sum() > 0
    assert np.asarray(blank).sum() == 0


def _client(backend: FakeMechane, **kw: Any) -> Skopos:
    kanon = Kanon(
        default_backend=backend.name,
        tiers={"pro": TierRule(name="pro", quota=QuotaRule(limit=50, window_seconds=60))},
    )
    validator = InMemoryApiKeyValidator()
    validator.register_simple("k", principal_id="u", tier=Klimax.PRO)
    taxis = Taxis()
    taxis.register(backend)
    return Skopos(kanon=kanon, taxis=taxis, validator=validator, **kw)


def test_public_api_for_segmentation_and_pose_with_options_passthrough() -> None:
    seen: dict[str, Any] = {}

    class Recorder(FakeMechane):
        def infer(self, praxis: Praxis, image: Physis, **options: object) -> Any:
            seen[praxis.value] = options
            if praxis is Praxis.SEGMENT:
                return SegmentationResult(backend=self.name, latency_ms=0)
            if praxis is Praxis.POSE:
                return PoseResult(backend=self.name, latency_ms=0)
            return super().infer(praxis, image, **options)

    client = _client(Recorder("rec", frozenset(Praxis)))
    assert isinstance(
        client.segmentation.segment(png_bytes(), api_key="k", options={"conf": 0.7}),
        SegmentationResult,
    )
    assert isinstance(client.pose.estimate(png_bytes(), api_key="k"), PoseResult)
    client.detection.detect(png_bytes(), api_key="k", options={"conf": 0.2})
    client.ocr.extract_text(png_bytes(), api_key="k", options={"min_confidence": 0.5})
    assert seen["segment"] == {"conf": 0.7}
    assert seen["detect"] == {"estimate_demographics": False, "conf": 0.2}
    assert seen["ocr"] == {"min_confidence": 0.5}
    items = client.segmentation.segment_batch([png_bytes(), b"bad"], api_key="k")
    assert [i.ok for i in items] == [True, False]
    assert client.pose.estimate_batch([png_bytes()], api_key="k")[0].ok


def test_default_costs_and_tiers_cover_new_praxeis() -> None:
    kanon = Kanon.default()
    assert kanon.cost_for("segment") == 2 and kanon.cost_for("pose") == 2
    assert "segment" in kanon.tier("pro").allowed_praxeis
