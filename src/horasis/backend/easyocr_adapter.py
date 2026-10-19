"""OCR backend powered by EasyOCR (extra: ``horasis[ocr]``)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from horasis.backend._common import order_into_lines, polygon_to_box, to_confidence
from horasis.backend.base import Backend
from horasis.foundation.errors import AdapterError, BackendUnavailableError, HorasisError
from horasis.foundation.imaging import load_rgb
from horasis.foundation.schemas import Image, OcrResult, Result, Task, TextRegion


class EasyOCRAdapter(Backend):
    """`Backend` that reads text with EasyOCR.

    Each text segment EasyOCR finds becomes a :class:`TextRegion` with its text,
    confidence, tight polygon and axis-aligned box. Segments are put in
    reading order and numbered into lines (``TextRegion.line_index``), so
    ``OcrResult.lines`` gives you line-level text.

    Args:
        languages: EasyOCR language codes, e.g. ``["en", "fr"]``.
        gpu: Use the GPU. ``None`` lets EasyOCR decide.
        min_confidence: Drop segments scoring below this.
        reader: A pre-built ``easyocr.Reader`` (or compatible object);
            mainly for tests. If omitted, one is created on first use,
            which may download model weights.
        name: Registry name; defaults to ``"easyocr"``.
    """

    supported_tasks = frozenset({Task.OCR})

    def __init__(
        self,
        *,
        languages: Sequence[str] = ("en",),
        gpu: bool | None = None,
        min_confidence: float = 0.0,
        reader: Any = None,
        name: str = "easyocr",
    ) -> None:
        self.name = name
        self._languages = list(languages)
        self._gpu = gpu
        self._min_confidence = min_confidence
        self._reader: Any = reader
        self._injected: Any = reader

    def _load_reader(self) -> Any:
        if self._reader is not None:
            return self._reader
        try:
            import easyocr
        except ImportError as exc:
            raise BackendUnavailableError(
                "The 'easyocr' package is required for EasyOCRAdapter. "
                "Install it with: pip install 'horasis[ocr]'"
            ) from exc
        gpu = self._gpu
        if gpu is None and self.effective_device() != "auto":
            gpu = self.effective_device() != "cpu"
        kwargs: dict[str, Any] = {} if gpu is None else {"gpu": gpu}
        self._reader = easyocr.Reader(self._languages, **kwargs)
        return self._reader

    def is_loaded(self) -> bool:
        return self._reader is not None and self._reader is not self._injected

    def unload(self) -> None:
        self._reader = self._injected

    def warm_up(self) -> None:
        self._load_reader()

    def infer(self, task: Task, image: Image, **options: Any) -> Result:
        if not self.supports(task):
            raise BackendUnavailableError(
                f"EasyOCRAdapter does not support task {task.value!r}"
            )
        min_conf = float(options.get("min_confidence", self._min_confidence))
        start = time.perf_counter()
        try:
            import numpy as np

            reader = self._load_reader()
            raw = reader.readtext(np.asarray(load_rgb(image)))
            segments = []
            for points, text, conf in raw:
                if not str(text).strip() or float(conf) < min_conf:
                    continue
                polygon = [(float(x), float(y)) for x, y in points]
                segments.append((str(text), float(conf), polygon))
            ordered = order_into_lines(segments, lambda seg: polygon_to_box(seg[2]))
            regions = [
                TextRegion(
                    text=text,
                    confidence=to_confidence(conf),
                    box=polygon_to_box(polygon),
                    polygon=polygon,
                    line_index=line,
                )
                for line, (text, conf, polygon) in ordered
            ]
        except HorasisError:
            raise
        except Exception as exc:
            raise AdapterError(f"EasyOCR inference failed: {exc}") from exc
        return OcrResult(
            backend=self.name, latency_ms=(time.perf_counter() - start) * 1000, regions=regions
        )
