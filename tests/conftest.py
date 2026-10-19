from __future__ import annotations

import io

import pytest
from PIL import Image as PILImage

from horasis.access.guard import AccessContext
from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.tiers import Tier
from horasis.backend.base import Backend
from horasis.backend.registry import BackendRegistry
from horasis.foundation.config import Config, QuotaRule, TierRule
from horasis.foundation.schemas import (
    ClassificationResult,
    DetectionResult,
    FaceResult,
    Image,
    ModerationResult,
    OcrResult,
    Result,
    Task,
)
from horasis.orchestration.pipeline import Pipeline


@pytest.fixture(autouse=True)
def _isolated_model_cache(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep offline tests away from the user's real model cache (integration tests use it)."""
    if request.node.get_closest_marker("integration") is None:
        monkeypatch.setenv("HORASIS_CACHE_DIR", str(tmp_path_factory.mktemp("model-cache")))


def png_bytes(width: int = 8, height: int = 6, color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    """A tiny but real PNG, for tests that go through input validation."""
    buf = io.BytesIO()
    PILImage.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip `@pytest.mark.integration` tests unless selected with `-m integration`."""
    if "integration" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="integration test (real models); run with -m integration")
    for item in items:
        if item.get_closest_marker("integration") is not None:
            item.add_marker(skip)


class FakeBackend(Backend):
    """A deterministic, in-memory `Backend` used across the test suite."""

    def __init__(self, name: str = "fake", tasks: frozenset[Task] | None = None) -> None:
        self.name = name
        self.supported_tasks = tasks or frozenset(
            {Task.DETECT, Task.CLASSIFY, Task.OCR, Task.FACE, Task.MODERATE}
        )
        self.calls: list[Task] = []

    def infer(self, task: Task, image: Image, **options: object) -> Result:
        self.calls.append(task)
        result_cls = {
            Task.DETECT: DetectionResult,
            Task.CLASSIFY: ClassificationResult,
            Task.OCR: OcrResult,
            Task.FACE: FaceResult,
            Task.MODERATE: ModerationResult,
        }[task]
        return result_cls(backend=self.name, latency_ms=0.5)


@pytest.fixture
def config() -> Config:
    return Config(
        default_backend="fake",
        tiers={
            "free": TierRule(
                name="free",
                quota=QuotaRule(limit=2, window_seconds=3600),
                allowed_tasks=["detect", "classify"],
            ),
            "pro": TierRule(
                name="pro",
                quota=QuotaRule(limit=1000, window_seconds=3600),
                allowed_tasks=["detect", "classify", "ocr", "face", "moderate"],
            ),
        },
    )


@pytest.fixture
def validator() -> InMemoryApiKeyValidator:
    v = InMemoryApiKeyValidator()
    v.register_simple("free-key", principal_id="free-user", tier=Tier.FREE)
    v.register_simple("pro-key", principal_id="pro-user", tier=Tier.PRO)
    return v


@pytest.fixture
def fake_backend() -> FakeBackend:
    return FakeBackend()


@pytest.fixture
def registry(fake_backend: FakeBackend) -> BackendRegistry:
    t = BackendRegistry()
    t.register(fake_backend)
    return t


@pytest.fixture
def pipeline(config: Config, registry: BackendRegistry) -> Pipeline:
    return Pipeline(config=config, registry=registry)


@pytest.fixture
def access(config: Config, validator: InMemoryApiKeyValidator) -> AccessContext:
    return AccessContext.build(config=config, validator=validator)


@pytest.fixture
def sample_image() -> Image:
    return Image.from_bytes(png_bytes(), source="test-fixture")
