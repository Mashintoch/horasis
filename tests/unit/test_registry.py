from __future__ import annotations

import pytest

from horasis.backend.registry import BackendRegistry
from horasis.foundation.config import Config
from horasis.foundation.errors import BackendUnavailableError, NoBackendError
from horasis.foundation.schemas import Task
from horasis.orchestration.pipeline import Pipeline
from tests.conftest import FakeBackend, png_bytes


def _registry(*backends: FakeBackend) -> BackendRegistry:
    t = BackendRegistry()
    for b in backends:
        t.register(b)
    return t


def test_explicit_backend_must_support_task() -> None:
    t = _registry(FakeBackend("a", frozenset({Task.OCR})))
    with pytest.raises(BackendUnavailableError, match="does not support"):
        t.resolve(Task.DETECT, preferred="a", default="a")


def test_config_order_beats_default() -> None:
    t = _registry(FakeBackend("default"), FakeBackend("tess"), FakeBackend("easy"))
    chosen = t.resolve(Task.OCR, preferred=None, default="default", order=["easy", "tess"])
    assert chosen.name == "easy"


def test_order_skips_unregistered_and_unsupporting_backends() -> None:
    t = _registry(
        FakeBackend("ocr-only", frozenset({Task.OCR})),
        FakeBackend("generalist"),
    )
    chosen = t.resolve(
        Task.DETECT, preferred=None, default="missing", order=["ghost", "ocr-only", "generalist"]
    )
    assert chosen.name == "generalist"


def test_fallback_used_after_default_fails() -> None:
    t = _registry(
        FakeBackend("default", frozenset({Task.DETECT})),
        FakeBackend("b"),
        FakeBackend("c"),
    )
    chosen = t.resolve(Task.OCR, preferred=None, default="default", fallback=["c", "b"])
    assert chosen.name == "c"


def test_falls_back_to_any_registered_backend() -> None:
    t = _registry(FakeBackend("only"))
    assert t.resolve(Task.FACE, preferred=None, default="nope").name == "only"


def test_no_backend_error_is_clear_and_structured() -> None:
    t = _registry(FakeBackend("det", frozenset({Task.DETECT})))
    with pytest.raises(NoBackendError) as info:
        t.resolve(Task.OCR, preferred=None, default="det")
    err = info.value
    assert isinstance(err, BackendUnavailableError)
    assert "'ocr'" in err.message and "det" in err.message and "horasis[ocr]" in err.message
    assert err.context["registered"] == ["det"]


def test_pipeline_routes_by_config_backend_order_and_fallback() -> None:
    config = Config(
        default_backend="main",
        backend_order={"ocr": ["secondary"]},
        fallback_order=["tertiary"],
    )
    main, secondary, tertiary = FakeBackend("main"), FakeBackend("secondary"), FakeBackend("tertiary")
    pipeline = Pipeline(config=config, registry=_registry(main, secondary, tertiary))
    from horasis.foundation.schemas import Image

    image = Image.from_bytes(png_bytes())
    assert pipeline.run(Task.OCR, image).backend == "secondary"
    assert pipeline.run(Task.DETECT, image).backend == "main"  # no order for detect
