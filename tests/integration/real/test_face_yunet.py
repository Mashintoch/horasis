from __future__ import annotations

import pytest

from horasis import models
from horasis.backend.face_adapter import YuNetFaceAdapter
from horasis.foundation.errors import ModelDownloadError
from horasis.foundation.schemas import FaceResult

from .common import (
    KEY,
    assert_box_in_bounds,
    assert_confidence,
    image_size,
    make_client,
    sample_image,
)

pytestmark = pytest.mark.integration
pytest.importorskip("cv2", reason="needs opencv (pip install 'horasis[face]')")


def test_yunet_finds_faces_in_zidane() -> None:
    try:
        models.pull("yunet")  # checksum-verified; a cache hit when already downloaded
    except ModelDownloadError as exc:
        pytest.skip(f"YuNet model unavailable: {exc}")
    assert models.verify("yunet")
    path = sample_image("zidane.jpg")
    client = make_client(YuNetFaceAdapter())
    result = client.face.detect_faces(path, api_key=KEY, backend="yunet")
    assert isinstance(result, FaceResult)
    assert result.faces
    size = image_size(path)
    for face in result.faces:
        assert face.label == "face"
        assert_confidence(face.confidence)
        assert_box_in_bounds(face.box, size)
        assert face.landmarks is not None and len(face.landmarks) == 5
        for x, y in face.landmarks.values():
            assert 0 <= x <= size[0] and 0 <= y <= size[1]
