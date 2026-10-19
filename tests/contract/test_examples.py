"""Every example must compile, and each capability example's `run()` must work."""

from __future__ import annotations

import asyncio
import importlib
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from horasis import Client, Config, HashedApiKeyValidator, Tier
from horasis.backend.registry import BackendRegistry
from horasis.foundation.schemas import (
    BoundingBox,
    ClassificationEntry,
    ClassificationResult,
    Confidence,
    Detection,
    DetectionResult,
    Face,
    FaceResult,
    Image,
    Keypoint,
    ModerationLabel,
    ModerationResult,
    OcrResult,
    Pose,
    PoseResult,
    Result,
    Segment,
    SegmentationResult,
    Task,
    TextRegion,
)
from tests.conftest import FakeBackend, png_bytes

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
BOX = BoundingBox(x_min=1, y_min=1, x_max=9, y_max=9)
CONF = Confidence(value=0.9)


class Everything(FakeBackend):
    """Answers every task with a small, non-empty result."""

    def __init__(self, name: str = "fake") -> None:
        super().__init__(name, frozenset(Task))

    def infer(self, task: Task, image: Image, **options: object) -> Result:
        kw: dict[str, Any] = {"backend": self.name, "latency_ms": 1.0}
        det = Detection(label="cat", confidence=CONF, box=BOX)
        return {
            Task.DETECT: lambda: DetectionResult(**kw, detections=[det]),
            Task.CLASSIFY: lambda: ClassificationResult(
                **kw, entries=[ClassificationEntry(label="tabby", confidence=CONF)]
            ),
            Task.OCR: lambda: OcrResult(
                **kw, regions=[TextRegion(text="hi", confidence=CONF, box=BOX, line_index=0)]
            ),
            Task.FACE: lambda: FaceResult(
                **kw, faces=[Face(label="face", confidence=CONF, box=BOX, landmarks={"nose": (4.0, 4.0)})]
            ),
            Task.MODERATE: lambda: ModerationResult(
                **kw, labels=[ModerationLabel(category="nsfw", confidence=CONF, flagged=True)]
            ),
            Task.SEGMENT: lambda: SegmentationResult(
                **kw, segments=[Segment(label="cat", confidence=CONF, box=BOX, polygon=[(1, 1), (9, 1), (5, 9)])]
            ),
            Task.POSE: lambda: PoseResult(
                **kw,
                poses=[
                    Pose(
                        label="person",
                        confidence=CONF,
                        box=BOX,
                        keypoints=[Keypoint(name="nose", x=4, y=4, confidence=CONF)],
                    )
                ],
            ),
        }[task]()


@pytest.fixture
def example_client() -> tuple[Client, str]:
    validator = HashedApiKeyValidator()
    key = validator.issue(principal_id="t", tier=Tier.PRO)
    registry = BackendRegistry()
    registry.register(Everything("fake"))
    registry.register(Everything("remote"))
    return Client(config=Config.default().model_copy(update={"default_backend": "fake"}), registry=registry, validator=validator), key


@pytest.fixture
def image(tmp_path: Path) -> str:
    path = tmp_path / "pic.png"
    path.write_bytes(png_bytes(40, 30))
    return str(path)


def load(name: str) -> Any:
    sys.path.insert(0, str(EXAMPLES))
    try:
        sys.modules.pop(name, None)
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(EXAMPLES))


def test_every_example_compiles() -> None:
    files = sorted(EXAMPLES.glob("*.py"))
    assert len(files) >= 10
    for f in files:
        compile(f.read_text(encoding="utf-8"), str(f), "exec")


@pytest.mark.parametrize("name", ["classify", "ocr", "faces", "moderate"])
def test_simple_examples_run(
    name: str, example_client: tuple[Client, str], image: str, capsys: pytest.CaptureFixture[str]
) -> None:
    client, key = example_client
    load(name).run(client, key, image)
    assert capsys.readouterr().out.strip()


@pytest.mark.parametrize("name", ["detect", "segment", "pose"])
def test_drawing_examples_run_and_save(
    name: str, example_client: tuple[Client, str], image: str, tmp_path: Path
) -> None:
    client, key = example_client
    out = tmp_path / f"{name}.png"
    load(name).run(client, key, image, out=str(out))
    assert out.is_file()


def test_async_batch_example(
    example_client: tuple[Client, str], image: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    client, key = example_client
    items = asyncio.run(load("async_batch").run(client, key, [image, str(tmp_path / "gone.png")]))
    out = capsys.readouterr().out
    assert [i.ok for i in items] == [True, False]
    assert "FAILED" in out and "quota used:" in out


def test_stream_example(
    example_client: tuple[Client, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_frames(source: Any, **kw: Any) -> Iterator[tuple[int, float, Any]]:
        for i in range(3):
            yield i, i / 10, np.zeros((6, 8, 3), dtype=np.uint8)

    monkeypatch.setattr("horasis.api.client.iter_frames", fake_frames)
    client, key = example_client
    assert load("stream_video").run(client, key, "clip.mp4", every_n_frames=1) == 3


def test_remote_client_example(example_client: tuple[Client, str], image: str) -> None:
    client, key = example_client
    result = load("remote_client").run(client, key, image)
    assert result.backend == "remote"


def test_server_example_builds_an_app() -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("multipart")
    app = load("server").build_app("secret")
    assert any(getattr(r, "path", "") == "/v1/{task}" for r in app.routes)
