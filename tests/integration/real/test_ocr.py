from __future__ import annotations

import shutil

import pytest
from PIL import Image, ImageDraw, ImageFont

from horasis.backend.easyocr_adapter import EasyOCRAdapter
from horasis.backend.tesseract_adapter import TesseractAdapter
from horasis.foundation.schemas import OcrResult

from .common import KEY, assert_box_in_bounds, assert_confidence, make_client

pytestmark = pytest.mark.integration

TEXT = "HELLO WORLD 2024"
SIZE = (640, 160)


def render_text() -> Image.Image:
    """Deterministic OCR input: black text on white, drawn with Pillow's bundled font."""
    image = Image.new("RGB", SIZE, "white")
    ImageDraw.Draw(image).text((30, 40), TEXT, fill="black", font=ImageFont.load_default(size=56))
    return image


def check(result: OcrResult) -> None:
    assert isinstance(result, OcrResult)
    words = result.text.upper().replace("\n", " ").split()
    assert "HELLO" in words and "WORLD" in words, result.text
    for glyph in result.glyphs:
        assert_confidence(glyph.confidence)
        assert_box_in_bounds(glyph.box, SIZE)


def test_tesseract_reads_rendered_text() -> None:
    pytest.importorskip("pytesseract", reason="needs pytesseract (horasis[ocr])")
    if shutil.which("tesseract") is None:
        pytest.skip("the tesseract binary is not installed")
    client = make_client(TesseractAdapter())
    check(client.ocr.extract_text(render_text(), api_key=KEY, backend="tesseract"))


def test_easyocr_reads_rendered_text() -> None:
    pytest.importorskip("easyocr", reason="needs easyocr (horasis[ocr])")
    client = make_client(EasyOCRAdapter(gpu=False))
    check(client.ocr.extract_text(render_text(), api_key=KEY, backend="easyocr"))
