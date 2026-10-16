"""Content-moderation backend using a transformers image classifier
(extra: ``horasis[moderation]``)."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from horasis.backend._common import doxa
from horasis.backend.mechane import Mechane
from horasis.foundation.imaging import load_rgb
from horasis.foundation.schemas import ModerationLabel, ModerationResult, Physis, Praxis, Theoria
from horasis.foundation.sphalma import AdapterError, BackendUnavailableError, Sphalma

DEFAULT_MODEL = "Falconsai/nsfw_image_detection"
DEFAULT_SAFE_CATEGORIES = frozenset({"normal", "safe", "sfw", "neutral", "drawings"})


class TransformersModerationAdapter(Mechane):
    """`Mechane` that scores an image against safety categories.

    Wraps a Hugging Face ``image-classification`` pipeline; every label the
    model emits becomes a category with its score.

    ``flagged`` is decided by thresholds. The :class:`~horasis.orchestration.organon.Organon`
    re-applies the thresholds from :class:`~horasis.foundation.kanon.Kanon`
    to every moderation result, so Kanon is the single source of truth when
    used through ``Skopos``. When used standalone, pass `threshold_for`
    (e.g. ``kanon.moderation_threshold_for``); the default is 0.5.

    Args:
        model: Hugging Face model id or local path. Weights are downloaded
            on first use unless already cached or local.
        threshold_for: Maps a category to its flag threshold.
        safe_categories: Categories that are never flagged (benign labels).
        device: Passed to the pipeline (``-1``/``"cpu"``, ``0``/``"cuda"``...).
        classifier: A pre-built callable returning
            ``[{"label": str, "score": float}, ...]`` (for tests).
        name: Registry name; defaults to ``"moderation"``.
    """

    supported_praxeis = frozenset({Praxis.MODERATE})

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        threshold_for: Callable[[str], float] | None = None,
        safe_categories: frozenset[str] = DEFAULT_SAFE_CATEGORIES,
        device: Any = None,
        classifier: Any = None,
        name: str = "moderation",
    ) -> None:
        self.name = name
        self._model = model
        self._threshold_for = threshold_for or (lambda _category: 0.5)
        self._safe = frozenset(c.lower() for c in safe_categories)
        self._device = device
        self._classifier: Any = classifier
        self._injected: Any = classifier

    def _load_classifier(self) -> Any:
        if self._classifier is not None:
            return self._classifier
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise BackendUnavailableError(
                "The 'transformers' package is required for the moderation backend. "
                "Install it with: pip install 'horasis[moderation]'"
            ) from exc
        device = self._device
        if device is None and self.effective_device() != "auto":
            device = self.effective_device()
        kwargs: dict[str, Any] = {} if device is None else {"device": device}
        self._classifier = pipeline("image-classification", model=self._model, **kwargs)
        return self._classifier

    def is_loaded(self) -> bool:
        return self._classifier is not None and self._classifier is not self._injected

    def unload(self) -> None:
        self._classifier = self._injected

    def warm_up(self) -> None:
        self._load_classifier()

    def infer(self, praxis: Praxis, image: Physis, **options: Any) -> Theoria:
        if not self.supports(praxis):
            raise BackendUnavailableError(
                f"TransformersModerationAdapter does not support praxis {praxis.value!r}"
            )
        start = time.perf_counter()
        try:
            classifier = self._load_classifier()
            raw = classifier(load_rgb(image), top_k=None)
            labels = [
                ModerationLabel(
                    category=str(item["label"]),
                    confidence=doxa(item["score"]),
                    flagged=self._is_flagged(str(item["label"]), float(item["score"])),
                )
                for item in raw
            ]
        except Sphalma:
            raise
        except Exception as exc:
            raise AdapterError(f"Moderation inference failed: {exc}") from exc
        return ModerationResult(
            backend=self.name, latency_ms=(time.perf_counter() - start) * 1000, labels=labels
        )

    def _is_flagged(self, category: str, score: float) -> bool:
        return category.lower() not in self._safe and score >= self._threshold_for(category)
