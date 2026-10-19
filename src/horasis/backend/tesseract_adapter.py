"""OCR backend powered by Tesseract (extra: ``horasis[ocr]``; needs the binary)."""

from __future__ import annotations

import time
from typing import Any

from horasis.backend._common import box_to_polygon, to_confidence
from horasis.backend.base import Backend
from horasis.foundation.errors import AdapterError, BackendUnavailableError, HorasisError
from horasis.foundation.imaging import load_rgb
from horasis.foundation.schemas import BoundingBox, Image, OcrResult, Result, Task, TextRegion


class TesseractAdapter(Backend):
    """`Backend` that reads text with Tesseract via ``pytesseract``.

    Tesseract reports words together with their block/paragraph/line
    numbers, which become ``TextRegion.line_index`` directly. Tesseract only
    gives rectangles, so each polygon is the box's four corners.

    Args:
        lang: Tesseract language string, e.g. ``"eng"`` or ``"eng+fra"``.
        min_confidence: Drop words scoring below this (0..1).
        engine: A ``pytesseract``-compatible module (for tests). If omitted,
            ``pytesseract`` is imported on first use.
        name: Registry name; defaults to ``"tesseract"``.
    """

    supported_tasks = frozenset({Task.OCR})

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
                "TesseractAdapter needs the pytesseract package: pip install 'horasis[ocr]' "
                "(plus the Tesseract binary)"
            ) from exc
        self._engine = pytesseract
        return pytesseract

    def infer(self, task: Task, image: Image, **options: Any) -> Result:
        if not self.supports(task):
            raise BackendUnavailableError(
                f"TesseractAdapter does not support task {task.value!r}"
            )
        min_conf = float(options.get("min_confidence", self._min_confidence))
        start = time.perf_counter()
        try:
            engine = self._load_engine()
            data = engine.image_to_data(load_rgb(image), lang=self._lang, output_type="dict")
            regions: list[TextRegion] = []
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
                regions.append(
                    TextRegion(
                        text=text,
                        confidence=to_confidence(conf / 100.0),
                        box=box,
                        polygon=box_to_polygon(box),
                        line_index=line,
                    )
                )
        except HorasisError:
            raise
        except Exception as exc:
            if type(exc).__name__ == "TesseractNotFoundError":
                raise BackendUnavailableError(
                    "Tesseract binary not found on PATH"
                ) from exc
            raise AdapterError(f"Tesseract inference failed: {exc}") from exc
        return OcrResult(
            backend=self.name, latency_ms=(time.perf_counter() - start) * 1000, regions=regions
        )
