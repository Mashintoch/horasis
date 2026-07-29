from __future__ import annotations

import pytest

from horasis.backend.mechane import Mechane
from horasis.backend.ultralytics_adapter import UltralyticsAdapter
from horasis.foundation.schemas import Physis, Praxis
from horasis.foundation.sphalma import BackendUnavailableError
from tests.conftest import FakeMechane


@pytest.mark.parametrize("backend_factory", [UltralyticsAdapter, FakeMechane])
def test_every_adapter_conforms_to_mechane(backend_factory) -> None:
    """Every backend, present or future, must satisfy the `Mechane` contract."""
    backend = backend_factory()
    assert isinstance(backend, Mechane)
    assert isinstance(backend.name, str) and backend.name
    assert isinstance(backend.supported_praxeis, frozenset)
    assert all(isinstance(p, Praxis) for p in backend.supported_praxeis)


def test_ultralytics_adapter_declares_detect_and_classify() -> None:
    adapter = UltralyticsAdapter()
    assert adapter.supports(Praxis.DETECT)
    assert adapter.supports(Praxis.CLASSIFY)
    assert not adapter.supports(Praxis.OCR)
    assert not adapter.supports(Praxis.FACE)
    assert not adapter.supports(Praxis.MODERATE)


def test_ultralytics_adapter_rejects_unsupported_praxis() -> None:
    adapter = UltralyticsAdapter()
    with pytest.raises(BackendUnavailableError):
        adapter.infer(Praxis.OCR, Physis.from_bytes(b"x"))


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
        adapter.infer(Praxis.DETECT, Physis.from_bytes(b"x"))
