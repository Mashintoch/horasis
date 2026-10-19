from __future__ import annotations

from horasis.backend.registry import BackendRegistry
from horasis.foundation.config import Config
from horasis.foundation.schemas import DetectionResult, Task
from horasis.orchestration.hooks import HookEvent, Hooks
from horasis.orchestration.pipeline import Pipeline
from tests.conftest import FakeBackend, png_bytes


def test_pipeline_resolves_default_backend_and_returns_result(config: Config, registry: BackendRegistry, sample_image) -> None:
    pipeline = Pipeline(config=config, registry=registry)
    result = pipeline.run(Task.DETECT, sample_image)
    assert isinstance(result, DetectionResult)
    assert result.backend == "fake"


def test_pipeline_honors_explicit_backend_override(config: Config, sample_image) -> None:
    registry = BackendRegistry()
    registry.register(FakeBackend(name="primary"))
    registry.register(FakeBackend(name="secondary"))
    pipeline = Pipeline(config=config, registry=registry)

    result = pipeline.run(Task.DETECT, sample_image, backend="secondary")
    assert result.backend == "secondary"


def test_hooks_fire_in_order(config: Config, registry: BackendRegistry, sample_image) -> None:
    fired: list[HookEvent] = []
    hooks = Hooks()
    for event in HookEvent:
        hooks.on(event, lambda event=event, **_: fired.append(event))

    pipeline = Pipeline(config=config, registry=registry, hooks=hooks)
    pipeline.run(Task.DETECT, sample_image)

    assert fired == [
        HookEvent.BEFORE_RESOLVE,
        HookEvent.AFTER_RESOLVE,
        HookEvent.BEFORE_INFER,
        HookEvent.AFTER_INFER,
    ]


def test_backend_actually_receives_the_call(config: Config, sample_image) -> None:
    backend = FakeBackend(name="tracked")
    registry = BackendRegistry()
    registry.register(backend)
    pipeline = Pipeline(config=config, registry=registry)

    pipeline.run(Task.CLASSIFY, sample_image)
    assert backend.calls == [Task.CLASSIFY]


def test_end_to_end_client_call_through_client(config: Config, registry: BackendRegistry, validator) -> None:
    from horasis.api.client import Client
    from horasis.foundation.schemas import Image

    client = Client(config=config, registry=registry, validator=validator)
    result = client.detection.detect(Image.from_bytes(png_bytes()), api_key="pro-key")
    assert result.backend == "fake"
    assert result.task is Task.DETECT
