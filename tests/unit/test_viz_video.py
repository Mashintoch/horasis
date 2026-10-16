"""Tests for the viz helpers, video rendering, annotate() and the merged
Ultralytics demographics/video code. Everything is faked: no models, no network."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import pytest
from PIL import Image

from horasis import viz
from horasis.access.keys import ApiKeyValidator
from horasis.backend import ultralytics_adapter as ua
from horasis.foundation.schemas import (
    BoundingBox,
    Detection,
    DetectionResult,
    Doxa,
    Face,
    FaceResult,
    FrameDetections,
    Glyph,
    Kinesis,
    OcrResult,
    Physis,
    Praxis,
    VideoDetectionResult,
)
from horasis.foundation.sphalma import (
    BackendUnavailableError,
    ValidationError,
    VisualizationUnavailableError,
)
from tests.conftest import png_bytes

BOX = BoundingBox(x_min=4, y_min=14, x_max=30, y_max=40)


def det(label: str = "cat") -> Detection:
    return Detection(label=label, confidence=Doxa(value=0.9), box=BOX)


def black() -> Image.Image:
    return Image.new("RGB", (48, 48), (0, 0, 0))


class FakeWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


@pytest.fixture
def avi_path(tmp_path: Path) -> Path:
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    if not writer.isOpened():
        pytest.skip("this OpenCV build cannot write MJPG video")
    for _ in range(4):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    return path


# ---- image drawing -----------------------------------------------------------------


def _green(img: Image.Image) -> int:
    return int(np.asarray(img)[..., 1].astype(int).sum())


def test_text_color_is_used_for_box_labels() -> None:
    kwargs = {"show_labels": True, "show_confidence": True, "box_color": (90, 0, 0)}
    red = viz._draw_boxes(black(), [det()], text_color=(255, 0, 0), line_width=2, font_size=12, **kwargs)
    green = viz._draw_boxes(black(), [det()], text_color=(0, 255, 0), line_width=2, font_size=12, **kwargs)
    assert _green(green) > _green(red)  # only the caption text differs between the two


def test_text_color_is_used_for_ocr_labels() -> None:
    ocr = OcrResult(
        backend="b",
        latency_ms=0,
        glyphs=[Glyph(text="HELLO", confidence=Doxa(value=0.9), box=BOX)],
    )
    red = viz.render_ocr(black(), ocr, box_color=(90, 0, 0), text_color=(255, 0, 0))
    green = viz.render_ocr(black(), ocr, box_color=(90, 0, 0), text_color=(0, 255, 0))
    assert _green(green) > _green(red)


def test_render_detections_and_faces_save_and_return_image(tmp_path: Path) -> None:
    out = tmp_path / "d" / "out.png"
    img = viz.render_detections(png_bytes(48, 48), DetectionResult(backend="b", latency_ms=0, detections=[det()]), output_path=out)
    assert out.is_file() and np.asarray(img).sum() > np.asarray(black()).sum()
    faces = FaceResult(backend="b", latency_ms=0, faces=[Face(label="face", confidence=Doxa(value=0.8), box=BOX)])
    out2 = tmp_path / "faces.png"
    viz.render_faces(Physis.from_bytes(png_bytes(48, 48)), faces, output_path=out2, show_labels=False)
    assert out2.is_file()


def test_output_path_resolution(tmp_path: Path) -> None:
    path, kw = viz._resolve_image_output_path(tmp_path)  # existing directory
    assert path == tmp_path / "annotated.png" and kw == {}
    path, kw = viz._resolve_image_output_path(tmp_path / "noext")
    assert kw == {"format": "PNG"}
    default, _ = viz._resolve_image_output_path(None)
    assert default.name == "annotated.png"


def test_color_for_label_is_deterministic() -> None:
    assert viz._color_for_label("cat") == viz._color_for_label("cat")
    assert len(viz._color_for_label("dog")) == 3


def test_annotate_methods_delegate_and_save(tmp_path: Path) -> None:
    result = DetectionResult(backend="b", latency_ms=0, detections=[det()])
    out = tmp_path / "a.png"
    annotated = result.annotate(png_bytes(48, 48), output_path=out, show_confidence=False)
    assert out.is_file() and annotated.size == (48, 48)
    faces = FaceResult(backend="b", latency_ms=0, faces=[Face(label="face", confidence=Doxa(value=0.8), box=BOX)])
    out2 = tmp_path / "f.png"
    assert faces.annotate(png_bytes(48, 48), output_path=out2).size == (48, 48)
    assert out2.is_file()


def test_open_image_rejects_bad_input() -> None:
    with pytest.raises(ValidationError):
        viz._open_image(12345)


# ---- video rendering --------------------------------------------------------------------


def test_open_video_writer_picks_ffmpeg_only_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(viz, "_FfmpegVideoWriter", lambda cv2, path, fps: ("ffmpeg", path))
    monkeypatch.setattr(viz, "_Cv2VideoWriter", lambda cv2, path, fps, w, h: ("cv2", path))
    both = lambda name: object()
    monkeypatch.setattr(viz.importlib.util, "find_spec", both)
    assert viz._open_video_writer(None, "o.mp4", 10.0, 4, 4) == ("ffmpeg", "o.mp4")
    monkeypatch.setattr(viz.importlib.util, "find_spec", lambda name: None)
    assert viz._open_video_writer(None, "o.mp4", 10.0, 4, 4) == ("cv2", "o.mp4")


def test_cv2_video_writer_wraps_opencv() -> None:
    calls: list[Any] = []

    class FakeCv2:
        @staticmethod
        def VideoWriter_fourcc(*chars: str) -> str:
            return "".join(chars)

        class VideoWriter:
            def __init__(self, path: str, fourcc: str, fps: float, size: tuple[int, int]) -> None:
                calls.append(("open", path, fourcc, fps, size))

            def write(self, frame: Any) -> None:
                calls.append(("write", frame))

            def release(self) -> None:
                calls.append(("release",))

    writer = viz._Cv2VideoWriter(FakeCv2, "o.mp4", 5.0, 8, 6)
    writer.write("frame")
    writer.release()
    assert calls == [("open", "o.mp4", "mp4v", 5.0, (8, 6)), ("write", "frame"), ("release",)]


def test_capture_source_variants(tmp_path: Path) -> None:
    assert viz._video_capture_source(tmp_path / "v.mp4") == (str(tmp_path / "v.mp4"), None)
    assert viz._video_capture_source(Kinesis(source="u", url="http://x/v.mp4")) == ("http://x/v.mp4", None)
    source, temp = viz._video_capture_source(b"not really a video")
    try:
        assert temp == source and Path(source).read_bytes() == b"not really a video"
    finally:
        os.unlink(source)
    with pytest.raises(ValidationError):
        viz._video_capture_source(3.14)  # type: ignore[arg-type]


def test_render_video_annotates_only_listed_frames(
    avi_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = FakeWriter()
    seen: dict[str, Any] = {}

    def fake_open(cv2: Any, path: str, fps: float, width: int, height: int) -> FakeWriter:
        seen.update(path=path, fps=fps, size=(width, height))
        return writer

    monkeypatch.setattr(viz, "_open_video_writer", fake_open)
    result = VideoDetectionResult(
        backend="b",
        latency_ms=0,
        fps=7.0,
        frames=[
            FrameDetections(frame_index=0, detections=[det()]),
            FrameDetections(frame_index=2, detections=[det("dog")]),
        ],
    )
    out = viz.render_video(avi_path, result, output_path=tmp_path / "sub" / "out.mp4", text_color=(0, 255, 0))
    assert out == tmp_path / "sub" / "out.mp4" and out.parent.is_dir()
    assert seen == {"path": str(out), "fps": 7.0, "size": (32, 24)}
    assert len(writer.frames) == 4 and writer.released
    annotated = [f.sum() > 0 for f in writer.frames]
    assert annotated == [True, False, True, False]  # sampled frames drawn, others pass through


def test_video_annotate_method_delegates(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_render(video: Any, result: Any, **kwargs: Any) -> str:
        captured.update(video=video, result=result, **kwargs)
        return "done"

    monkeypatch.setattr(viz, "render_video", fake_render)
    result = VideoDetectionResult(backend="b", latency_ms=0)
    assert result.annotate("in.mp4", output_path="out.mp4", show_labels=False) == "done"
    assert captured["video"] == "in.mp4" and captured["show_labels"] is False


def test_render_video_cleans_up_when_video_cannot_be_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("cv2")
    made: list[str] = []
    real = viz._video_capture_source

    def spy(video: Any) -> tuple[Any, str | None]:
        source, temp = real(video)
        if temp:
            made.append(temp)
        return source, temp

    monkeypatch.setattr(viz, "_video_capture_source", spy)
    with pytest.raises(ValidationError, match="open"):
        viz.render_video(b"garbage bytes", VideoDetectionResult(backend="b", latency_ms=0), output_path=tmp_path / "o.mp4")
    assert made and not Path(made[0]).exists()


def test_load_cv2_failure_is_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "cv2", None)
    with pytest.raises(VisualizationUnavailableError, match=r"horasis\[viz\]"):
        viz._load_cv2()


def test_api_key_validator_protocol_stub_is_abstract() -> None:
    with pytest.raises(NotImplementedError):
        ApiKeyValidator.validate(object(), "key")  # type: ignore[arg-type]


# ---- merged Ultralytics code: demographics + video ---------------------------------------


def _box(cls: int, conf: float, xyxy: list[float]) -> Any:
    return SimpleNamespace(cls=[cls], conf=[conf], xyxy=[np.array(xyxy, dtype=np.float32)])


def _raw(boxes: list[Any], orig: np.ndarray | None = None) -> Any:
    frame = orig if orig is not None else np.full((24, 32, 3), 40, dtype=np.uint8)
    return SimpleNamespace(names={0: "person", 1: "dog"}, boxes=boxes, orig_img=frame)


def _install_yolo(monkeypatch: pytest.MonkeyPatch, adapter: ua.UltralyticsAdapter, make: Any) -> None:
    class FakeYOLO:
        def __init__(self, path: str) -> None:
            self.path = path

        def __call__(self, source: Any, **kwargs: Any) -> Any:
            return make(source, **kwargs)

    monkeypatch.setattr(adapter, "_load_ultralytics", lambda: FakeYOLO)


class FakeDeepFace:
    result: ClassVar[Any] = [{"age": 31, "dominant_gender": "Man", "gender": {"Man": 91.0, "Woman": 9.0}}]
    seen: ClassVar[list[Any]] = []

    @classmethod
    def analyze(cls, crop: Any, **kwargs: Any) -> Any:
        cls.seen.append(crop.shape)
        if isinstance(cls.result, Exception):
            raise cls.result
        return cls.result


def test_demographics_attach_estimates_to_person_boxes_only(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ua.UltralyticsAdapter()
    _install_yolo(monkeypatch, adapter, lambda src, **kw: [_raw([_box(0, 0.9, [2, 2, 20, 22]), _box(1, 0.8, [1, 1, 5, 5])])])
    monkeypatch.setattr(adapter, "_load_deepface", lambda: FakeDeepFace)
    FakeDeepFace.seen = []
    FakeDeepFace.result = [{"age": 31, "dominant_gender": "Man", "gender": {"Man": 91.0, "Woman": 9.0}}]
    result = adapter.infer(Praxis.DETECT, Physis.from_bytes(png_bytes(32, 24)), estimate_demographics=True)
    person, dog = result.detections  # type: ignore[attr-defined]
    assert person.face_estimate.estimated_age == 31
    assert person.face_estimate.estimated_gender == "Man"
    assert float(person.face_estimate.gender_confidence) == pytest.approx(0.91)
    assert dog.face_estimate is None and FakeDeepFace.seen == [(20, 18, 3)]


@pytest.mark.parametrize("analysis", [[], ValueError("no face"), RuntimeError("tf")])
def test_demographics_failures_yield_no_estimate(monkeypatch: pytest.MonkeyPatch, analysis: Any) -> None:
    adapter = ua.UltralyticsAdapter()
    monkeypatch.setattr(adapter, "_load_deepface", lambda: FakeDeepFace)
    FakeDeepFace.result = analysis
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    assert adapter._estimate_face_in_box(frame, [2, 2, 20, 20]) is None
    FakeDeepFace.result = [{"age": 1, "dominant_gender": "Man", "gender": {"Man": 50.0}}]
    assert adapter._estimate_face_in_box(frame, [30, 20, 10, 10]) is None  # degenerate box


def test_deepface_missing_is_backend_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "deepface", None)
    with pytest.raises(BackendUnavailableError, match=r"horasis\[demographics\]"):
        ua.UltralyticsAdapter()._load_deepface()


def test_detect_video_samples_frames_and_cleans_temp_file(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = ua.UltralyticsAdapter()
    calls: dict[str, Any] = {}

    def make(source: Any, **kw: Any) -> Any:
        calls.update(source=source, **kw)
        return iter([_raw([_box(0, 0.9, [1, 1, 9, 9])]), _raw([])])

    _install_yolo(monkeypatch, adapter, make)
    result = adapter.infer_video(Praxis.DETECT, Kinesis.from_bytes(b"video-bytes"), sample_every_n_frames=5)
    assert [f.frame_index for f in result.frames] == [0, 5]  # type: ignore[attr-defined]
    assert result.frame_count == 2 and calls["vid_stride"] == 5  # type: ignore[attr-defined]
    assert not Path(calls["source"]).exists()  # temp file for raw bytes was removed
    with pytest.raises(BackendUnavailableError):
        adapter.infer_video(Praxis.CLASSIFY, Kinesis.from_path("x.mp4"))


def test_detect_and_annotate_video_single_pass(
    avi_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = ua.UltralyticsAdapter()
    writer = FakeWriter()
    monkeypatch.setattr(ua, "_open_video_writer", lambda cv2, path, fps, w, h: writer)
    frames = [_raw([_box(0, 0.9, [2, 2, 20, 20])]), _raw([_box(0, 0.9, [2, 2, 20, 20])]), _raw([]), _raw([])]
    _install_yolo(monkeypatch, adapter, lambda src, **kw: iter(frames))
    result, out = adapter.infer_video_and_annotate(
        Praxis.DETECT,
        Kinesis.from_path(avi_path),
        output_path=tmp_path / "o" / "out.mp4",
        sample_every_n_frames=2,
        text_color=(0, 255, 0),
    )
    assert out == tmp_path / "o" / "out.mp4"
    assert [f.frame_index for f in result.frames] == [0, 2]  # type: ignore[attr-defined]
    assert len(writer.frames) == 4 and writer.released
    # frame 0 had detections and was drawn; frame 1 was skipped by sampling so it is untouched
    assert writer.frames[0].sum() > writer.frames[1].sum()
    with pytest.raises(BackendUnavailableError):
        adapter.infer_video_and_annotate(Praxis.CLASSIFY, Kinesis.from_path(avi_path), output_path="x")
