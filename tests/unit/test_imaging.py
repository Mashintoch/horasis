from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from horasis.foundation import imaging
from horasis.foundation.imaging import decode_image, encode_image, load_rgb, to_physis
from horasis.foundation.kanon import InputPolicy
from horasis.foundation.schemas import Physis
from horasis.foundation.sphalma import ValidationError
from tests.conftest import png_bytes


def _jpeg_with_orientation(orientation: int, size: tuple[int, int] = (8, 4)) -> bytes:
    img = Image.new("RGB", size, (10, 20, 30))
    exif = img.getexif()
    exif[0x0112] = orientation
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    return buf.getvalue()


def test_bytes_become_decoded_physis() -> None:
    physis = to_physis(png_bytes(8, 6))
    assert (physis.width, physis.height) == (8, 6)
    assert load_rgb(physis).size == (8, 6)


def test_path_and_str_path(tmp_path: Path) -> None:
    f = tmp_path / "a.png"
    f.write_bytes(png_bytes())
    assert to_physis(f).source == str(f)
    assert to_physis(str(f)).width == 8


def test_missing_file_is_validation_error(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not found"):
        to_physis(tmp_path / "nope.png")


def test_pil_and_numpy_inputs() -> None:
    assert to_physis(Image.new("L", (5, 3))).width == 5
    arr = np.zeros((3, 5, 3), dtype=np.uint8)
    assert to_physis(arr).height == 3
    assert Physis.from_array(arr).width == 5
    assert Physis.from_pil(Image.new("RGB", (2, 2))).width == 2


@pytest.mark.parametrize(
    "arr",
    [
        np.zeros((4, 4), dtype=np.uint8),
        np.zeros((4, 4, 1), dtype=np.uint8),
        np.zeros((4, 4, 4), dtype=np.uint8),
        np.full((4, 4, 3), 0.5, dtype=np.float32),
        np.full((4, 4, 3), 300, dtype=np.int32),
        np.ones((4, 4), dtype=bool),
    ],
)
def test_numpy_dtypes_and_layouts_are_accepted(arr: np.ndarray) -> None:
    assert to_physis(arr).pixels.mode == "RGB"


def test_float_array_in_unit_range_is_scaled() -> None:
    px = np.asarray(to_physis(np.full((2, 2, 3), 1.0, dtype=np.float32)).pixels)
    assert px.max() == 255


@pytest.mark.parametrize("shape", [(4,), (4, 4, 2), (2, 2, 2, 3)])
def test_bad_array_shapes_rejected(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValidationError):
        to_physis(np.zeros(shape, dtype=np.uint8))


def test_unsupported_type_rejected() -> None:
    with pytest.raises(ValidationError, match="Unsupported"):
        to_physis(12345)


def test_garbage_and_empty_bytes_rejected() -> None:
    with pytest.raises(ValidationError, match="decode"):
        decode_image(b"definitely not an image")
    with pytest.raises(ValidationError, match="empty"):
        decode_image(b"")


def test_truncated_image_rejected() -> None:
    with pytest.raises(ValidationError):
        decode_image(png_bytes(64, 64)[:-40])


def test_size_limits() -> None:
    with pytest.raises(ValidationError, match="bytes"):
        decode_image(png_bytes(), InputPolicy(max_bytes=10))
    with pytest.raises(ValidationError, match="pixels"):
        decode_image(png_bytes(10, 10), InputPolicy(max_pixels=50))
    with pytest.raises(ValidationError, match="pixels"):
        to_physis(np.zeros((10, 10, 3), dtype=np.uint8), InputPolicy(max_pixels=50))


def test_file_size_checked_before_reading(tmp_path: Path) -> None:
    f = tmp_path / "big.png"
    f.write_bytes(png_bytes())
    with pytest.raises(ValidationError, match="limit"):
        to_physis(f, InputPolicy(max_bytes=10))


def test_format_allow_list() -> None:
    with pytest.raises(ValidationError, match="not allowed"):
        decode_image(png_bytes(), InputPolicy(allowed_formats=["JPEG"]))


@pytest.mark.parametrize(
    ("orientation", "expected_size"),
    [(1, (8, 4)), (6, (4, 8)), (8, (4, 8)), (3, (8, 4))],
)
def test_exif_orientation_is_applied(orientation: int, expected_size: tuple[int, int]) -> None:
    assert decode_image(_jpeg_with_orientation(orientation)).size == expected_size


def test_exif_orientation_can_be_disabled() -> None:
    policy = InputPolicy(apply_exif_orientation=False)
    assert decode_image(_jpeg_with_orientation(6), policy).size == (8, 4)


def test_urls_are_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_fetch(url: str, *, timeout: float, max_bytes: int) -> bytes:
        calls.append(url)
        return png_bytes()

    monkeypatch.setattr(imaging, "fetch_url", fake_fetch)
    with pytest.raises(ValidationError, match="disabled"):
        to_physis("https://example.com/a.png")
    assert calls == []  # nothing was fetched

    physis = to_physis("https://example.com/a.png", InputPolicy(allow_urls=True))
    assert calls == ["https://example.com/a.png"]
    assert physis.width == 8


def test_physis_from_source_and_from_path_reject_urls_by_default() -> None:
    with pytest.raises(ValidationError):
        Physis.from_source("https://example.com/a.png")
    with pytest.raises(ValidationError):
        Physis.from_path("https://example.com/a.png")


def test_fetch_url_rejects_non_http() -> None:
    with pytest.raises(ValidationError, match="http"):
        imaging.fetch_url("file:///etc/passwd", timeout=1, max_bytes=10)


def test_load_rgb_decodes_lazily_built_physis() -> None:
    assert load_rgb(Physis.from_bytes(png_bytes(7, 5))).size == (7, 5)


def test_encode_image_roundtrip() -> None:
    physis = to_physis(png_bytes(9, 3))
    for fmt in ("JPEG", "PNG"):
        assert Image.open(io.BytesIO(encode_image(physis, fmt=fmt))).size == (9, 3)
