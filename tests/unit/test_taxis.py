from __future__ import annotations

import pytest

from horasis.backend.taxis import Taxis
from horasis.foundation.kanon import Kanon
from horasis.foundation.schemas import Praxis
from horasis.foundation.sphalma import BackendUnavailableError, NoBackendError
from horasis.orchestration.organon import Organon
from tests.conftest import FakeMechane, png_bytes


def _taxis(*backends: FakeMechane) -> Taxis:
    t = Taxis()
    for b in backends:
        t.register(b)
    return t


def test_explicit_backend_must_support_praxis() -> None:
    t = _taxis(FakeMechane("a", frozenset({Praxis.OCR})))
    with pytest.raises(BackendUnavailableError, match="does not support"):
        t.resolve(Praxis.DETECT, preferred="a", default="a")


def test_config_order_beats_default() -> None:
    t = _taxis(FakeMechane("default"), FakeMechane("tess"), FakeMechane("easy"))
    chosen = t.resolve(Praxis.OCR, preferred=None, default="default", order=["easy", "tess"])
    assert chosen.name == "easy"


def test_order_skips_unregistered_and_unsupporting_backends() -> None:
    t = _taxis(
        FakeMechane("ocr-only", frozenset({Praxis.OCR})),
        FakeMechane("generalist"),
    )
    chosen = t.resolve(
        Praxis.DETECT, preferred=None, default="missing", order=["ghost", "ocr-only", "generalist"]
    )
    assert chosen.name == "generalist"


def test_fallback_used_after_default_fails() -> None:
    t = _taxis(
        FakeMechane("default", frozenset({Praxis.DETECT})),
        FakeMechane("b"),
        FakeMechane("c"),
    )
    chosen = t.resolve(Praxis.OCR, preferred=None, default="default", fallback=["c", "b"])
    assert chosen.name == "c"


def test_falls_back_to_any_registered_backend() -> None:
    t = _taxis(FakeMechane("only"))
    assert t.resolve(Praxis.FACE, preferred=None, default="nope").name == "only"


def test_no_backend_error_is_clear_and_structured() -> None:
    t = _taxis(FakeMechane("det", frozenset({Praxis.DETECT})))
    with pytest.raises(NoBackendError) as info:
        t.resolve(Praxis.OCR, preferred=None, default="det")
    err = info.value
    assert isinstance(err, BackendUnavailableError)
    assert "'ocr'" in err.message and "det" in err.message and "horasis[ocr]" in err.message
    assert err.context["registered"] == ["det"]


def test_organon_routes_by_kanon_backend_order_and_fallback() -> None:
    kanon = Kanon(
        default_backend="main",
        backend_order={"ocr": ["secondary"]},
        fallback_order=["tertiary"],
    )
    main, secondary, tertiary = FakeMechane("main"), FakeMechane("secondary"), FakeMechane("tertiary")
    organon = Organon(kanon=kanon, taxis=_taxis(main, secondary, tertiary))
    from horasis.foundation.schemas import Physis

    image = Physis.from_bytes(png_bytes())
    assert organon.run(Praxis.OCR, image).backend == "secondary"
    assert organon.run(Praxis.DETECT, image).backend == "main"  # no order for detect
