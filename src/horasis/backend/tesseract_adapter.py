"""OCR backend powered by Tesseract (extra: ``horasis[ocr]``; needs the binary)."""

from __future__ import annotations

import time
from typing import Any

from horasis.backend._common import box_to_polygon, doxa
from horasis.backend.mechane import Mechane
from horasis.foundation.imaging import load_rgb
from horasis.foundation.schemas import BoundingBox, Glyph, OcrResult, Physis, Praxis, Theoria
from horasis.foundation.sphalma import AdapterError, BackendUnavailableError, Sphalma


class TesseractAdapter(Mechane):
    """`Mechane` that reads text with Tesseract via ``pytesseract``.

    Tesseract reports words together with their block/paragraph/line
    numbers, which become ``Glyph.line_index`` directly. Tesseract only
    gives rectangles, so each polygon is the box's four corners.

    Args:
        lang: Tesseract language string, e.g. ``"eng"`` or ``"eng+fra"``.
        min_confidence: Drop words scoring below this (0..1).
        engine: A ``pytesseract``-compatible module (for tests). If omitted,
            ``pytesseract`` is imported on first use.
        name: Registry name; defaults to ``"tesseract"``.
    """

    supported_praxeis = frozenset({Praxis.OCR})

    def __init__(
        self,
        *,
        lang: str = "eng",
        min_confidence: float = 0.0,
        engine: Any = None,
        name: str = "tesseract",
    ) -> None:
        self.name = name
        self._lang = lang
        self._min_confidence = min_confidence
        self._engine: Any = engine

    def _load_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            import pytesseract
        except ImportError as exc:
            raise BackendUnavailableError(
                "The 'pytesseract' package is required for TesseractAdapter. "
                "Install it with: pip install 'horasis[ocr]' (and install the Tesseract binary)"
            ) from exc
        self._engine = pytesseract
        return pytesseract

    def infer(self, praxis: Praxis, image: Physis, **options: Any) -> Theoria:
        if not self.supports(praxis):
            raise BackendUnavailableError(
                f"TesseractAdapter does not support praxis {praxis.value!r}"
            )
        min_conf = float(options.get("min_confidence", self._min_confidence))
        start = time.perf_counter()
        try:
            engine = self._load_engine()
            data = engine.image_to_data(load_rgb(image), lang=self._lang, output_type="dict")
            glyphs: list[Glyph] = []
            line_ids: dict[tuple[int, int, int], int] = {}
            for i, text in enumerate(data["text"]):
                text = str(text).strip()
                conf = float(data["conf"][i])
                if not text or conf < 0 or conf / 100.0 < min_conf:
                    continue
                key = (int(data["block_num"][i]), int(data["par_num"][i]), int(data["line_num"][i]))
                line = line_ids.setdefault(key, len(line_ids))
                left, top = float(data["left"][i]), float(data["top"][i])
                box = BoundingBox(
                    x_min=left,
                    y_min=top,
                    x_max=left + float(data["width"][i]),
                    y_max=top + float(data["height"][i]),
                )
                glyphs.append(
                    Glyph(
                        text=text,
                        confidence=doxa(conf / 100.0),
                        box=box,
                        polygon=box_to_polygon(box),
                        line_index=line,
                    )
                )
        except Sphalma:
            raise
        except Exception as exc:
            if type(exc).__name__ == "TesseractNotFoundError":
                raise BackendUnavailableError(
                    "The Tesseract binary was not found; install it and ensure it is on PATH"
                ) from exc
            raise AdapterError(f"Tesseract inference failed: {exc}") from exc
        return OcrResult(
            backend=self.name, latency_ms=(time.perf_counter() - start) * 1000, glyphs=glyphs
        )
