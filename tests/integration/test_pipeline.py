from __future__ import annotations

from horasis.backend.taxis import Taxis
from horasis.foundation.kanon import Kanon
from horasis.foundation.schemas import DetectionResult, Praxis
from horasis.orchestration.kairos import Kairos, Moment
from horasis.orchestration.organon import Organon
from tests.conftest import FakeMechane


def test_organon_resolves_default_backend_and_returns_result(kanon: Kanon, taxis: Taxis, sample_image) -> None:
    organon = Organon(kanon=kanon, taxis=taxis)
    result = organon.run(Praxis.DETECT, sample_image)
    assert isinstance(result, DetectionResult)
    assert result.backend == "fake"


def test_organon_honors_explicit_backend_override(kanon: Kanon, sample_image) -> None:
    taxis = Taxis()
    taxis.register(FakeMechane(name="primary"))
    taxis.register(FakeMechane(name="secondary"))
    organon = Organon(kanon=kanon, taxis=taxis)

    result = organon.run(Praxis.DETECT, sample_image, backend="secondary")
    assert result.backend == "secondary"


def test_kairos_hooks_fire_in_order(kanon: Kanon, taxis: Taxis, sample_image) -> None:
    fired: list[Moment] = []
    kairos = Kairos()
    for moment in Moment:
        kairos.on(moment, lambda moment=moment, **_: fired.append(moment))

    organon = Organon(kanon=kanon, taxis=taxis, kairos=kairos)
    organon.run(Praxis.DETECT, sample_image)

    assert fired == [
        Moment.BEFORE_RESOLVE,
        Moment.AFTER_RESOLVE,
        Moment.BEFORE_INFER,
        Moment.AFTER_INFER,
    ]


def test_backend_actually_receives_the_call(kanon: Kanon, sample_image) -> None:
    backend = FakeMechane(name="tracked")
    taxis = Taxis()
    taxis.register(backend)
    organon = Organon(kanon=kanon, taxis=taxis)

    organon.run(Praxis.CLASSIFY, sample_image)
    assert backend.calls == [Praxis.CLASSIFY]


def test_end_to_end_client_call_through_skopos(kanon: Kanon, taxis: Taxis, validator) -> None:
    from horasis.api.skopos import Skopos
    from horasis.foundation.schemas import Physis

    client = Skopos(kanon=kanon, taxis=taxis, validator=validator)
    result = client.detection.detect(Physis.from_bytes(b"img"), api_key="pro-key")
    assert result.backend == "fake"
    assert result.praxis is Praxis.DETECT
