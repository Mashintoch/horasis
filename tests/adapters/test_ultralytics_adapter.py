from __future__ import annotations

import pytest

from horasis.backend.base import Backend
from horasis.backend.ultralytics_adapter import UltralyticsAdapter
from horasis.foundation.errors import BackendUnavailableError
from horasis.foundation.schemas import Image, Task
from tests.conftest import FakeBackend


@pytest.mark.parametrize("backend_factory", [UltralyticsAdapter, FakeBackend])
def test_every_adapter_conforms_to_backend(backend_factory) -> None:
    """Every backend, present or future, must satisfy the `Backend` contract."""
    backend = backend_factory()
    assert isinstance(backend, Backend)
    assert isinstance(backend.name, str) and backend.name
    assert isinstance(backend.supported_tasks, frozenset)
    assert all(isinstance(p, Task) for p in backend.supported_tasks)


def test_ultralytics_adapter_declares_detect_and_classify() -> None:
    adapter = UltralyticsAdapter()
    assert adapter.supports(Task.DETECT)
    assert adapter.supports(Task.CLASSIFY)
    assert not adapter.supports(Task.OCR)
    assert not adapter.supports(Task.FACE)
    assert not adapter.supports(Task.MODERATE)


def test_ultralytics_adapter_rejects_unsupported_task() -> None:
    adapter = UltralyticsAdapter()
    with pytest.raises(BackendUnavailableError):
        adapter.infer(Task.OCR, Image.from_bytes(b"x"))


def test_ultralytics_adapter_surfaces_missing_dependency() -> None:
    """Without the `ultralytics` extra installed, inference must fail loudly
    with `BackendUnavailableError`, not a bare `ImportError` or `AdapterError`.
    """
    adapter = UltralyticsAdapter()
    try:
        import ultralytics  # noqa: F401

        pytest.skip("ultralytics is installed in this environment; nothing to assert")
    except ImportError:
        pass

    with pytest.raises(BackendUnavailableError):
        adapter.infer(Task.DETECT, Image.from_bytes(b"x"))
