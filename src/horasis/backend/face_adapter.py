"""Face detection backend powered by OpenCV's YuNet (extra: ``horasis[face]``)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from horasis import models
from horasis.backend._common import to_confidence
from horasis.backend.base import Backend
from horasis.foundation.errors import (
    AdapterError,
    BackendUnavailableError,
    HorasisError,
    ModelDownloadError,
)
from horasis.foundation.imaging import load_rgb
from horasis.foundation.schemas import BoundingBox, Face, FaceResult, Image, Result, Task

MODEL_ENV_VAR = "HORASIS_YUNET_MODEL"
MODEL_NAME = "yunet"
LANDMARK_NAMES = ("right_eye", "left_eye", "nose", "mouth_right", "mouth_left")


class YuNetFaceAdapter(Backend):
    """`Backend` that finds faces with OpenCV's YuNet detector.

    YuNet is a ~230 KB ONNX model, so this is the lightest face backend.
    It only *locates* faces -- it never identifies or recognizes anyone.
    Each :class:`Face` has a box, a confidence and (optionally) 5 landmarks.

    The model file is found in this order: `model_path`, the
    ``HORASIS_YUNET_MODEL`` env var, then the horasis model cache. Fetch it
    with ``horasis models pull yunet``, or set ``Config(auto_download_models=True)``
    to download it on first use.

    Args:
        model_path: Path to the YuNet ``.onnx`` file (optional; see above).
        score_threshold: Minimum face confidence.
        nms_threshold: IoU threshold for suppressing overlapping faces.
        top_k: Candidates kept before NMS.
        landmarks: Include the 5-point landmarks on each face.
        detector: A pre-built detector exposing ``setInputSize`` and
            ``detect`` (for tests). If omitted, one is built on first use.
        name: Registry name; defaults to ``"yunet"``.
    """

    supported_tasks = frozenset({Task.FACE})

    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        landmarks: bool = True,
        detector: Any = None,
        name: str = "yunet",
    ) -> None:
        self.name = name
        self._model_path = model_path
        self._score_threshold = score_threshold
        self._nms_threshold = nms_threshold
        self._top_k = top_k
        self._landmarks = landmarks
        self._detector: Any = detector
        self._injected: Any = detector

    def _load_detector(self) -> Any:
        if self._detector is not None:
            return self._detector
        path = self._resolve_model_path()
        try:
            import cv2
        except ImportError as exc:
            raise BackendUnavailableError(
                "OpenCV is required for YuNetFaceAdapter. "
                "Install it with: pip install 'horasis[face]'"
            ) from exc
        self._detector = cv2.FaceDetectorYN.create(
            str(path), "", (320, 320), self._score_threshold, self._nms_threshold, self._top_k
        )
        return self._detector

    def _resolve_model_path(self) -> Path:
        explicit = self._model_path or os.environ.get(MODEL_ENV_VAR)
        if explicit:
            if not Path(explicit).is_file():
                raise BackendUnavailableError(f"YuNet model file not found: {explicit}")
            return Path(explicit)
        if models.is_cached(MODEL_NAME):
            return models.model_path(MODEL_NAME)
        if self.default_auto_download:
            try:
                return models.pull(MODEL_NAME)
            except ModelDownloadError as exc:
                raise BackendUnavailableError(f"Could not fetch the YuNet model: {exc}") from exc
        raise BackendUnavailableError(
            "The YuNet face model is not available. Run `horasis models pull yunet`, "
            f"pass model_path=..., set {MODEL_ENV_VAR}, or use Config(auto_download_models=True)."
        )

    def is_loaded(self) -> bool:
        return self._detector is not None and self._detector is not self._injected

    def unload(self) -> None:
        self._detector = self._injected

    def warm_up(self) -> None:
        self._load_detector()

    def infer(self, task: Task, image: Image, **options: Any) -> Result:
        if not self.supports(task):
            raise BackendUnavailableError(
                f"YuNetFaceAdapter does not support task {task.value!r}"
            )
        with_landmarks = bool(options.get("landmarks", self._landmarks))
        start = time.perf_counter()
        try:
            import numpy as np

            detector = self._load_detector()
            rgb = np.asarray(load_rgb(image))
            bgr = np.ascontiguousarray(rgb[..., ::-1])
            height, width = bgr.shape[:2]
            detector.setInputSize((width, height))
            _, rows = detector.detect(bgr)
            faces: list[Face] = []
            for row in [] if rows is None else rows:
                x, y, w, h = (float(v) for v in row[:4])
                marks = None
                if with_landmarks:
                    marks = {
                        n: (float(row[4 + 2 * i]), float(row[5 + 2 * i]))
                        for i, n in enumerate(LANDMARK_NAMES)
                    }
                faces.append(
                    Face(
                        label="face",
                        confidence=to_confidence(row[14]),
                        box=BoundingBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h),
                        landmarks=marks,
                    )
                )
        except HorasisError:
            raise
        except Exception as exc:
            raise AdapterError(f"Face detection failed: {exc}") from exc
        return FaceResult(
            backend=self.name, latency_ms=(time.perf_counter() - start) * 1000, faces=faces
        )
