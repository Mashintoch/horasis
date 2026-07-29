from __future__ import annotations

import pytest

from horasis.access.keys import InMemoryApiKeyValidator
from horasis.access.klimax import Klimax
from horasis.access.limen import AccessContext
from horasis.backend.mechane import Mechane
from horasis.backend.taxis import Taxis
from horasis.foundation.kanon import Kanon, QuotaRule, TierRule
from horasis.foundation.schemas import (
    ClassificationResult,
    DetectionResult,
    FaceResult,
    ModerationResult,
    OcrResult,
    Physis,
    Praxis,
    Theoria,
)
from horasis.orchestration.organon import Organon


class FakeMechane(Mechane):
    """A deterministic, in-memory `Mechane` used across the test suite."""

    def __init__(self, name: str = "fake", praxeis: frozenset[Praxis] | None = None) -> None:
        self.name = name
        self.supported_praxeis = praxeis or frozenset(
            {Praxis.DETECT, Praxis.CLASSIFY, Praxis.OCR, Praxis.FACE, Praxis.MODERATE}
        )
        self.calls: list[Praxis] = []

    def infer(self, praxis: Praxis, image: Physis, **options: object) -> Theoria:
        self.calls.append(praxis)
        result_cls = {
            Praxis.DETECT: DetectionResult,
            Praxis.CLASSIFY: ClassificationResult,
            Praxis.OCR: OcrResult,
            Praxis.FACE: FaceResult,
            Praxis.MODERATE: ModerationResult,
        }[praxis]
        return result_cls(backend=self.name, latency_ms=0.5)


@pytest.fixture
def kanon() -> Kanon:
    return Kanon(
        default_backend="fake",
        tiers={
            "free": TierRule(
                name="free",
                quota=QuotaRule(limit=2, window_seconds=3600),
                allowed_praxeis=["detect", "classify"],
            ),
            "pro": TierRule(
                name="pro",
                quota=QuotaRule(limit=1000, window_seconds=3600),
                allowed_praxeis=["detect", "classify", "ocr", "face", "moderate"],
            ),
        },
    )


@pytest.fixture
def validator() -> InMemoryApiKeyValidator:
    v = InMemoryApiKeyValidator()
    v.register_simple("free-key", principal_id="free-user", tier=Klimax.FREE)
    v.register_simple("pro-key", principal_id="pro-user", tier=Klimax.PRO)
    return v


@pytest.fixture
def fake_backend() -> FakeMechane:
    return FakeMechane()


@pytest.fixture
def taxis(fake_backend: FakeMechane) -> Taxis:
    t = Taxis()
    t.register(fake_backend)
    return t


@pytest.fixture
def organon(kanon: Kanon, taxis: Taxis) -> Organon:
    return Organon(kanon=kanon, taxis=taxis)


@pytest.fixture
def access(kanon: Kanon, validator: InMemoryApiKeyValidator) -> AccessContext:
    return AccessContext.build(kanon=kanon, validator=validator)


@pytest.fixture
def sample_image() -> Physis:
    return Physis.from_bytes(b"not-really-an-image", source="test-fixture")
