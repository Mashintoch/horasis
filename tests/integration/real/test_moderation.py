from __future__ import annotations

import pytest

from horasis.backend.moderation_adapter import TransformersModerationAdapter
from horasis.foundation.schemas import ModerationResult

from .common import KEY, assert_confidence, assert_is_name, make_client, sample_image

pytestmark = pytest.mark.integration
pytest.importorskip("transformers", reason="needs horasis[moderation]")
pytest.importorskip("torch", reason="needs horasis[moderation]")


def test_moderation_labels_are_named_and_benign_image_not_flagged() -> None:
    client = make_client(TransformersModerationAdapter())
    result = client.moderation.moderate(sample_image("bus.jpg"), api_key=KEY, backend="moderation")
    assert isinstance(result, ModerationResult)
    assert result.labels
    for label in result.labels:
        assert_is_name(label.category)
        assert_confidence(label.confidence)
    assert not any(label.flagged for label in result.labels)
