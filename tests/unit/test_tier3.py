from __future__ import annotations

import asyncio
import json
import logging
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image as PILImage

from horasis import AsyncClient, Client
from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.tiers import Tier
from horasis.backend.registry import BackendRegistry
from horasis.foundation import frames
from horasis.foundation.config import Config, QuotaRule, TierRule
from horasis.foundation.errors import (
    AdapterError,
    BackendUnavailableError,
    InvalidApiKeyError,
    NoBackendError,
    QuotaExceededError,
    ValidationError,
)
from horasis.foundation.frames import iter_frames
from horasis.foundation.schemas import (
    BoundingBox,
    ClassificationEntry,
    ClassificationResult,
    Confidence,
    Detection,
    DetectionResult,
    FrameDetections,
    Image,
    ModerationLabel,
    ModerationResult,
    OcrResult,
    Task,
    TextRegion,
    VideoDetectionResult,
)
from horasis.foundation.telemetry import OTelTelemetry
from horasis.orchestration.builtin_hooks import LoggingHook, TimingHook
from horasis.orchestration.hooks import HookEvent, Hooks
from horasis.orchestration.pipeline import Pipeline
from tests.conftest import FakeBackend, png_bytes


def det(label: str, conf: float, x: float = 0) -> Detection:
    return Detection(
        label=label,
        confidence=Confidence(value=conf),
        box=BoundingBox(x_min=x, y_min=2, x_max=x + 6, y_max=8),
    )


def result_with(*dets: Detection) -> DetectionResult:
    return DetectionResult(backend="fake", latency_ms=1.0, detections=list(dets))


# Result utilities


def test_to_dict_and_to_json_round_trip() -> None:
    r = result_with(det("cat", 0.9))
    assert r.to_dict()["detections"][0]["label"] == "cat"
    assert json.loads(r.to_json())["backend"] == "fake"
    assert DetectionResult.model_validate_json(r.to_json()) == r
    assert "\n" in r.to_json(indent=2)


def test_filter_by_confidence_and_labels() -> None:
    r = result_with(det("cat", 0.9), det("dog", 0.4), det("cat", 0.3))
    assert [d.label for d in r.filter(min_conf=0.5).detections] == ["cat"]
    assert len(r.filter(labels=["cat"]).detections) == 2
    assert len(r.filter(min_conf=0.35, labels={"cat"}).detections) == 1
    assert len(r.detections) == 3  # original untouched
    assert r.filter().detections == r.detections


def test_filter_other_result_types_and_video() -> None:
    cls = ClassificationResult(
        backend="b",
        latency_ms=0,
        entries=[
            ClassificationEntry(label="a", confidence=Confidence(value=0.9)),
            ClassificationEntry(label="b", confidence=Confidence(value=0.1)),
        ],
    )
    assert [e.label for e in cls.filter(min_conf=0.5).entries] == ["a"]
    ocr = OcrResult(
        backend="b",
        latency_ms=0,
        regions=[TextRegion(text="hi", confidence=Confidence(value=0.9)), TextRegion(text="x", confidence=Confidence(value=0.2))],
    )
    assert [g.text for g in ocr.filter(labels=["hi"]).regions] == ["hi"]
    mod = ModerationResult(
        backend="b",
        latency_ms=0,
        labels=[ModerationLabel(category="nsfw", confidence=Confidence(value=0.7), flagged=True)],
    )
    assert mod.filter(min_conf=0.8).labels == []
    video = VideoDetectionResult(
        backend="b",
        latency_ms=0,
        frames=[FrameDetections(frame_index=0, detections=[det("cat", 0.9), det("dog", 0.1)])],
    )
    kept = video.filter(min_conf=0.5)
    assert len(kept.frames) == 1 and [d.label for d in kept.frames[0].detections] == ["cat"]


def test_coco_export() -> None:
    r = result_with(det("dog", 0.8, x=10), det("cat", 0.9))
    coco = r.to_coco(image_id=7, file_name="a.jpg", width=100, height=50)
    assert coco["images"] == [{"id": 7, "file_name": "a.jpg", "width": 100, "height": 50}]
    assert coco["categories"] == [{"id": 1, "name": "cat"}, {"id": 2, "name": "dog"}]
    dog = next(a for a in coco["annotations"] if a["category_id"] == 2)
    assert dog["bbox"] == [10, 2, 6, 6] and dog["area"] == 36 and dog["score"] == 0.8
    assert dog["image_id"] == 7 and dog["iscrowd"] == 0
    custom = r.to_coco(category_ids={"dog": 5})
    assert [a["category_id"] for a in custom["annotations"]] == [5]  # unknown labels skipped


def test_video_coco_export() -> None:
    video = VideoDetectionResult(
        backend="b",
        latency_ms=0,
        frames=[
            FrameDetections(frame_index=0, detections=[det("cat", 0.9)]),
            FrameDetections(frame_index=4, detections=[det("cat", 0.8)]),
        ],
    )
    coco = video.to_coco(width=10, height=10)
    assert [i["id"] for i in coco["images"]] == [1, 5]
    assert [a["image_id"] for a in coco["annotations"]] == [1, 5]
    assert len({a["id"] for a in coco["annotations"]}) == 2


def test_draw_returns_image_without_saving(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    source = PILImage.new("RGB", (40, 30), (0, 0, 0))
    drawn = result_with(det("cat", 0.9)).draw(source)
    assert isinstance(drawn, PILImage.Image) and drawn.size == (40, 30)
    assert np.asarray(drawn).sum() > 0  # something was drawn
    assert np.asarray(source).sum() == 0  # the input was not modified
    ocr = OcrResult(
        backend="b",
        latency_ms=0,
        regions=[TextRegion(text="hi", confidence=Confidence(value=0.9), polygon=[(1, 1), (20, 1), (20, 9), (1, 9)])],
    )
    assert np.asarray(ocr.draw(source)).sum() > 0
    cls = ClassificationResult(
        backend="b", latency_ms=0, entries=[ClassificationEntry(label="a", confidence=Confidence(value=0.9))]
    )
    assert np.asarray(cls.draw(source)).shape == (30, 40, 3)
    mod = ModerationResult(
        backend="b",
        latency_ms=0,
        labels=[ModerationLabel(category="nsfw", confidence=Confidence(value=0.9), flagged=True)],
    )
    assert isinstance(mod.draw(png_bytes()), PILImage.Image)
    with pytest.raises(ValidationError):
        VideoDetectionResult(backend="b", latency_ms=0).draw(source)
    assert list(tmp_path.iterdir()) == []  # no annotated.png side effect


# Hooks


def _pipeline(backend: FakeBackend | None = None, hooks: Hooks | None = None) -> Pipeline:
    registry = BackendRegistry()
    registry.register(backend or FakeBackend())
    return Pipeline(config=Config(default_backend="fake"), registry=registry, hooks=hooks)


IMG = Image.from_bytes(png_bytes())


def test_hook_payloads_and_order() -> None:
    hooks = Hooks()
    seen: list[tuple[HookEvent, dict[str, Any]]] = []
    for event in HookEvent:
        hooks.on(event, lambda m=event, **p: seen.append((m, p)))
    _pipeline(hooks=hooks).run(Task.DETECT, IMG)
    assert [m for m, _ in seen] == [
        HookEvent.BEFORE_RESOLVE,
        HookEvent.AFTER_RESOLVE,
        HookEvent.BEFORE_INFER,
        HookEvent.AFTER_INFER,
    ]
    assert seen[0][1] == {"task": Task.DETECT, "image": IMG}
    assert seen[1][1]["backend"].name == "fake"
    assert isinstance(seen[3][1]["result"], DetectionResult)


class Exploding(FakeBackend):
    def infer(self, task: Task, image: Image, **options: object) -> Any:
        raise AdapterError("boom")


def test_on_error_fires_for_backend_and_resolution_failures() -> None:
    hooks = Hooks()
    errors: list[BaseException] = []
    hooks.on(HookEvent.ON_ERROR, lambda **p: errors.append(p["error"]))
    with pytest.raises(AdapterError):
        _pipeline(Exploding(), hooks).run(Task.DETECT, IMG)
    with pytest.raises(BackendUnavailableError):
        _pipeline(FakeBackend(tasks=frozenset({Task.OCR})), hooks).run(Task.DETECT, IMG)
    assert [type(e) for e in errors] == [AdapterError, NoBackendError]


def test_before_hook_exception_propagates_and_error_hook_failure_is_swallowed() -> None:
    hooks = Hooks()
    hooks.on(HookEvent.BEFORE_INFER, lambda **_: (_ for _ in ()).throw(RuntimeError("hook bug")))
    hooks.on(HookEvent.ON_ERROR, lambda **_: (_ for _ in ()).throw(RuntimeError("error hook bug")))
    with pytest.raises(RuntimeError, match="^hook bug$"):  # not masked by the error hook
        _pipeline(hooks=hooks).run(Task.DETECT, IMG)


def test_hooks_can_be_removed() -> None:
    hooks = Hooks()
    calls: list[int] = []

    def hook(**_: Any) -> None:
        calls.append(1)

    hooks.on(HookEvent.AFTER_INFER, hook)
    hooks.off(HookEvent.AFTER_INFER, hook)
    _pipeline(hooks=hooks).run(Task.DETECT, IMG)
    assert calls == []


def test_logging_hook(caplog: pytest.LogCaptureFixture) -> None:
    hooks = Hooks()
    LoggingHook(level=logging.INFO).attach(hooks)
    with caplog.at_level(logging.INFO, logger="horasis.pipeline"):
        _pipeline(hooks=hooks).run(Task.DETECT, IMG)
        with pytest.raises(AdapterError):
            _pipeline(Exploding(), hooks).run(Task.DETECT, IMG)
    text = caplog.text
    assert "after_infer task=detect backend=fake latency_ms=" in text
    assert "on_error task=detect" in text and "boom" in text


def test_timing_hook_records_and_summarises() -> None:
    hooks = Hooks()
    timing = TimingHook().attach(hooks)

    class Slow(FakeBackend):
        def infer(self, task: Task, image: Image, **options: object) -> Any:
            time.sleep(0.02)
            return super().infer(task, image, **options)

    pipeline = _pipeline(Slow(), hooks)
    pipeline.run(Task.DETECT, IMG)
    pipeline.run(Task.OCR, IMG)
    assert [r.task for r in timing.records] == ["detect", "ocr"]
    assert all(r.backend == "fake" and r.seconds >= 0.02 for r in timing.records)
    assert timing.summary()["detect"]["count"] == 1
    with pytest.raises(AdapterError):
        _pipeline(Exploding(), hooks).run(Task.DETECT, IMG)
    assert len(timing.records) == 2  # failed run recorded nothing
    timing.clear()
    assert timing.records == []


# Client fixtures


def make_client(limit: int = 100, backends: list[FakeBackend] | None = None, **config_kw: Any) -> Client:
    config = Config(
        default_backend="fake",
        tiers={"pro": TierRule(name="pro", quota=QuotaRule(limit=limit, window_seconds=3600))},
        task_cost={"detect": 1, "classify": 1, "ocr": 1, "face": 1, "moderate": 1},
        **config_kw,
    )
    validator = InMemoryApiKeyValidator()
    validator.register_simple("k", principal_id="u", tier=Tier.PRO)
    registry = BackendRegistry()
    for b in backends or [FakeBackend()]:
        registry.register(b)
    return Client(config=config, registry=registry, validator=validator)


# Batching


def test_detect_batch_gives_per_item_results_and_errors() -> None:
    client = make_client()
    items = client.detection.detect_batch([png_bytes(), b"garbage", png_bytes(3, 3)], api_key="k")
    assert [i.index for i in items] == [0, 1, 2]
    assert [i.ok for i in items] == [True, False, True]
    assert isinstance(items[0].result, DetectionResult)
    assert isinstance(items[1].error, ValidationError)
    assert items[1].source == "<bytes>"
    assert items[0].unwrap() is items[0].result
    with pytest.raises(ValidationError):
        items[1].unwrap()


def test_batch_charges_quota_per_item_and_reports_exhaustion_per_item() -> None:
    client = make_client(limit=3)
    items = client.classification.classify_batch([png_bytes()] * 5, api_key="k")
    assert [i.ok for i in items] == [True, True, True, False, False]
    assert all(isinstance(i.error, QuotaExceededError) for i in items[3:])
    assert client.usage("k").overall.used == 3


def test_invalid_image_costs_no_quota() -> None:
    client = make_client()
    with pytest.raises(ValidationError):
        client.detection.detect(b"garbage", api_key="k")
    assert client.usage("k").overall.used == 0


def test_batch_charges_only_items_that_pass_validation() -> None:
    client = make_client()
    items = client.detection.detect_batch([png_bytes(), b"garbage", png_bytes()], api_key="k")
    assert [i.ok for i in items] == [True, False, True]
    assert client.usage("k").overall.used == 2


def test_every_capability_has_a_batch_method() -> None:
    client = make_client()
    imgs = [png_bytes(), png_bytes()]
    for method in (
        client.detection.detect_batch,
        client.classification.classify_batch,
        client.ocr.extract_text_batch,
        client.face.detect_faces_batch,
        client.moderation.moderate_batch,
    ):
        assert all(i.ok for i in method(imgs, api_key="k"))


def test_invalid_key_aborts_the_whole_batch() -> None:
    with pytest.raises(InvalidApiKeyError):
        make_client().detection.detect_batch([png_bytes()], api_key="wrong")


def test_batch_items_describe_paths(tmp_path: Path) -> None:
    f = tmp_path / "a.png"
    f.write_bytes(png_bytes())
    items = make_client().detection.detect_batch([f, tmp_path / "missing.png"], api_key="k")
    assert items[0].source == str(f) and items[1].source.endswith("missing.png")
    assert isinstance(items[1].error, ValidationError)


# Frames and streaming


class FakeCapture:
    def __init__(self, n: int, opened: bool = True) -> None:
        self.n, self.i, self.opened, self.released = n, 0, opened, False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, prop: int) -> float:
        return 10.0

    def read(self) -> tuple[bool, Any]:
        if self.i >= self.n:
            return False, None
        self.i += 1
        frame = np.zeros((4, 6, 3), dtype=np.uint8)
        frame[..., 0] = 255  # blue in BGR
        return True, frame

    def release(self) -> None:
        self.released = True


def test_iter_frames_sampling_timestamps_and_rgb_conversion() -> None:
    cap = FakeCapture(10)
    out = list(iter_frames("v.mp4", every_n_frames=3, capture_factory=lambda s: cap))
    assert [i for i, _, _ in out] == [0, 3, 6, 9]
    assert [round(t, 2) for _, t, _ in out] == [0.0, 0.3, 0.6, 0.9]
    assert tuple(out[0][2][0, 0]) == (0, 0, 255)  # BGR -> RGB
    assert cap.released


def test_iter_frames_max_frames_release_and_errors() -> None:
    cap = FakeCapture(100)
    gen = iter_frames(0, max_frames=2, capture_factory=lambda s: cap)
    assert len(list(gen)) == 2 and cap.released
    cap2 = FakeCapture(100)
    gen2 = iter_frames("v.mp4", capture_factory=lambda s: cap2)
    next(gen2)
    gen2.close()
    assert cap2.released  # early stop still releases the capture
    with pytest.raises(ValidationError, match="open"):
        list(iter_frames("x", capture_factory=lambda s: FakeCapture(0, opened=False)))
    with pytest.raises(ValidationError):
        list(iter_frames("x", every_n_frames=0))


def test_iter_frames_reads_a_real_video(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG video")
    for _ in range(6):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    out = list(iter_frames(path, every_n_frames=2))
    assert [i for i, _, _ in out] == [0, 2, 4] and out[0][2].shape == (24, 32, 3)


def test_client_stream_charges_per_processed_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_iter(source: Any, **kw: Any) -> Iterator[tuple[int, float, Any]]:
        for i in range(0, 20, kw["every_n_frames"]):
            yield i, i / 10, np.zeros((6, 8, 3), dtype=np.uint8)

    monkeypatch.setattr("horasis.api.client.iter_frames", fake_iter)
    client = make_client(limit=2)
    got: list[Any] = []
    with pytest.raises(QuotaExceededError):
        for fr in client.detection.stream("v.mp4", api_key="k", every_n_frames=5):
            got.append(fr.frame_index)
    assert got == [0, 5]  # two frames were paid for, the third hit the quota
    client = make_client(limit=100)
    frames_out = list(client.detection.stream("v.mp4", api_key="k", every_n_frames=5))
    assert [f.frame_index for f in frames_out] == [0, 5, 10, 15]
    assert frames_out[1].timestamp_s == 0.5 and isinstance(frames_out[1].result, DetectionResult)
    assert client.usage("k").overall.used == 4


def test_stream_urls_are_opt_in() -> None:
    with pytest.raises(ValidationError, match="disabled"):
        next(make_client().detection.stream("https://x.test/v.mp4", api_key="k"))


# Model lifecycle


class Loadable(FakeBackend):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.loaded = False
        self.unloads = 0

    def infer(self, task: Task, image: Image, **options: object) -> Any:
        self.loaded = True
        return super().infer(task, image, **options)

    def is_loaded(self) -> bool:
        return self.loaded

    def unload(self) -> None:
        self.loaded = False
        self.unloads += 1


def test_lru_cap_unloads_least_recently_used_backend() -> None:
    a, b, c = Loadable("a"), Loadable("b"), Loadable("c")
    client = make_client(backends=[a, b, c], max_loaded_models=2)
    for name in ("a", "b", "a", "c"):
        client.detection.detect(png_bytes(), api_key="k", backend=name)
    assert (a.loaded, b.loaded, c.loaded) == (True, False, True)  # b was LRU
    assert b.unloads == 1 and a.unloads == 0
    client.detection.detect(png_bytes(), api_key="k", backend="b")  # lazily reloads
    assert b.loaded and not a.loaded  # now a is the LRU


def test_no_cap_never_unloads() -> None:
    a, b = Loadable("a"), Loadable("b")
    client = make_client(backends=[a, b])
    client.detection.detect(png_bytes(), api_key="k", backend="a")
    client.detection.detect(png_bytes(), api_key="k", backend="b")
    assert a.loaded and b.loaded and a.unloads == b.unloads == 0


def test_device_setting_reaches_backends() -> None:
    a = Loadable("a")
    pinned = Loadable("pinned")
    pinned.device = "cuda"
    client = make_client(backends=[a, pinned], device="cpu")
    assert a.effective_device() == "cpu" and pinned.effective_device() == "cuda"
    late = Loadable("late")
    client.registry.register(late)  # registered after construction
    assert late.effective_device() == "cpu"
    with pytest.raises(ValueError):
        Config(device="tpu")  # type: ignore[arg-type]


def test_ultralytics_device_option() -> None:
    from horasis.backend.ultralytics_adapter import UltralyticsAdapter

    adapter = UltralyticsAdapter()
    assert adapter._with_device({"conf": 0.3}) == {"conf": 0.3}  # auto: leave it to Ultralytics
    adapter.default_device = "cuda"
    assert adapter._with_device({})["device"] == "cuda:0"
    assert adapter._with_device({"device": "cpu"}) == {"device": "cpu"}  # explicit option wins


def test_warmup_skips_unavailable_but_raises_when_named() -> None:
    class Missing(FakeBackend):
        def warm_up(self) -> None:
            raise BackendUnavailableError("extra not installed")

    good, bad = Loadable("good"), Missing("bad")
    client = make_client(backends=[good, bad])
    assert client.warmup() == ["good"]
    with pytest.raises(BackendUnavailableError):
        client.warmup(["bad"])
    assert client.warmup(["good"]) == ["good"]


# Async


def test_async_client_mirrors_sync() -> None:
    async def scenario() -> None:
        async with AsyncClient(make_client(limit=10)) as client:
            one = await client.detection.detect(png_bytes(), api_key="k")
            assert isinstance(one, DetectionResult)
            items = await client.classification.classify_batch(
                [png_bytes(), b"junk", png_bytes()], api_key="k"
            )
            assert [i.ok for i in items] == [True, False, True]
            assert (await client.ocr.extract_text(png_bytes(), api_key="k")).task is Task.OCR
            assert (await client.face.detect_faces(png_bytes(), api_key="k")).task is Task.FACE
            assert (await client.moderation.moderate(png_bytes(), api_key="k")).task is Task.MODERATE
            # detect 1 + classify batch 2 (the malformed item is not charged) + ocr/face/moderate 3
            assert (await client.usage("k")).overall.used == 6
            assert await client.warmup() == ["fake"]
            with pytest.raises(InvalidApiKeyError):
                await client.detection.detect(png_bytes(), api_key="nope")

    asyncio.run(scenario())


def test_async_calls_run_concurrently_off_the_loop() -> None:
    barrier = threading.Barrier(3, timeout=5)

    class Rendezvous(FakeBackend):
        def infer(self, task: Task, image: Image, **options: object) -> Any:
            barrier.wait()  # only passes if 3 inferences are in flight at once
            return super().infer(task, image, **options)

    async def scenario() -> None:
        async with AsyncClient(make_client(backends=[Rendezvous()]), max_workers=4) as client:
            results = await asyncio.gather(
                *(client.detection.detect(png_bytes(), api_key="k") for _ in range(3))
            )
            assert len(results) == 3

    asyncio.run(scenario())


def test_async_batch_runs_items_concurrently_and_stream_iterates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_iter(source: Any, **kw: Any) -> Iterator[tuple[int, float, Any]]:
        for i in range(3):
            yield i, float(i), np.zeros((4, 4, 3), dtype=np.uint8)

    monkeypatch.setattr("horasis.api.client.iter_frames", fake_iter)

    async def scenario() -> None:
        async with AsyncClient(make_client()) as client:
            items = await client.detection.detect_batch([png_bytes()] * 4, api_key="k")
            assert len(items) == 4 and all(i.ok for i in items)
            seen = [f.frame_index async for f in client.detection.stream("v.mp4", api_key="k")]
            assert seen == [0, 1, 2]

    asyncio.run(scenario())


# OpenTelemetry


class FakeSpan:
    def __init__(self, name: str, attributes: dict[str, Any]) -> None:
        self.name, self.attributes, self.ended, self.exceptions = name, dict(attributes), False, []

    def set_attribute(self, k: str, v: Any) -> None:
        self.attributes[k] = v

    def record_exception(self, e: BaseException) -> None:
        self.exceptions.append(e)

    def set_status(self, status: Any) -> None:
        self.attributes["status"] = status

    def end(self) -> None:
        self.ended = True


class FakeTracer:
    def __init__(self) -> None:
        self.spans: list[FakeSpan] = []

    def start_span(self, name: str, attributes: dict[str, Any]) -> FakeSpan:
        self.spans.append(FakeSpan(name, attributes))
        return self.spans[-1]


class FakeInstrument:
    def __init__(self) -> None:
        self.points: list[tuple[float, dict[str, Any]]] = []

    def add(self, v: float, attrs: dict[str, Any]) -> None:
        self.points.append((v, attrs))

    record = add


class FakeMeter:
    def __init__(self) -> None:
        self.made: dict[str, FakeInstrument] = {}

    def _make(self, name: str, **_: Any) -> FakeInstrument:
        return self.made.setdefault(name, FakeInstrument())

    create_counter = create_histogram = _make


def test_otel_emits_spans_and_metrics_through_the_client() -> None:
    tracer, meter = FakeTracer(), FakeMeter()
    sink = OTelTelemetry(tracer=tracer, meter=meter)
    assert sink.enabled
    config = Config(
        default_backend="fake",
        tiers={"pro": TierRule(name="pro", quota=QuotaRule(limit=9, window_seconds=60))},
    )
    validator = InMemoryApiKeyValidator()
    validator.register_simple("k", principal_id="u", tier=Tier.PRO)
    registry = BackendRegistry()
    registry.register(Exploding("bad"))
    registry.register(FakeBackend())
    client = Client(config=config, registry=registry, validator=validator, telemetry=sink)
    client.detection.detect(png_bytes(), api_key="k", backend="fake")
    with pytest.raises(AdapterError):
        client.detection.detect(png_bytes(), api_key="k", backend="bad")

    ok, failed = tracer.spans
    assert ok.name == "horasis.detect" and ok.ended and ok.attributes["horasis.backend"] == "fake"
    assert failed.ended and isinstance(failed.exceptions[0], AdapterError)
    assert len(meter.made["horasis.calls"].points) == 2
    assert len(meter.made["horasis.latency_ms"].points) == 1
    assert meter.made["horasis.errors"].points[0][1]["horasis.error"] == "AdapterError"


def test_otel_is_a_noop_without_the_package(monkeypatch: pytest.MonkeyPatch) -> None:
    for mod in ("opentelemetry", "opentelemetry.trace", "opentelemetry.metrics"):
        monkeypatch.setitem(sys.modules, mod, None)
    sink = OTelTelemetry()
    assert sink.enabled is False
    sink.on_call_start(task="detect", principal_id="u")
    sink.on_call_end(task="detect", principal_id="u", backend="b", latency_ms=1.0, success=True)
    sink.on_error(task="detect", principal_id="u", error=RuntimeError("x"))
    client = make_client()
    assert isinstance(client.detection.detect(png_bytes(), api_key="k"), DetectionResult)


def test_frames_module_requires_opencv_when_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(Exception, match="horasis\\[video\\]"):
        list(frames.iter_frames("v.mp4"))
