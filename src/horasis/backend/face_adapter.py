"""Face detection backend powered by OpenCV's YuNet (extra: ``horasis[face]``)."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from horasis.backend._common import doxa
from horasis.backend.mechane import Mechane
from horasis.foundation.imaging import load_rgb
from horasis.foundation.schemas import BoundingBox, Face, FaceResult, Physis, Praxis, Theoria
from horasis.foundation.sphalma import AdapterError, BackendUnavailableError, Sphalma

MODEL_ENV_VAR = "HORASIS_YUNET_MODEL"
LANDMARK_NAMES = ("right_eye", "left_eye", "nose", "mouth_right", "mouth_left")


class YuNetFaceAdapter(Mechane):
    """`Mechane` that finds faces with OpenCV's YuNet detector.

    YuNet is a ~230 KB ONNX model, so this is the lightest face backend.
    It only *locates* faces -- it never identifies or recognizes anyone.
    Each :class:`Face` has a box, a confidence and (optionally) 5 landmarks.

    The model file is **not** downloaded for you. Get
    ``face_detection_yunet_2023mar.onnx`` from the OpenCV Zoo and pass its
    path as `model_path` or set the ``HORASIS_YUNET_MODEL`` env var.

    Args:
        model_path: Path to the YuNet ``.onnx`` file.
        score_threshold: Minimum face confidence.
        nms_threshold: IoU threshold for suppressing overlapping faces.
        top_k: Candidates kept before NMS.
        landmarks: Include the 5-point landmarks on each face.
        detector: A pre-built detector exposing ``setInputSize`` and
            ``detect`` (for tests). If omitted, one is built on first use.
        name: Registry name; defaults to ``"yunet"``.
    """

    supported_praxeis = frozenset({Praxis.FACE})

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
        path = self._model_path or os.environ.get(MODEL_ENV_VAR)
        if not path or not Path(path).is_file():
            raise BackendUnavailableError(
                "YuNetFaceAdapter needs the YuNet model file. Download "
                "face_detection_yunet_2023mar.onnx from the OpenCV Zoo and pass "
                f"model_path=... or set {MODEL_ENV_VAR}."
            )
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

    def is_loaded(self) -> bool:
        return self._detector is not None and self._detector is not self._injected

    def unload(self) -> None:
        self._detector = self._injected

    def warm_up(self) -> None:
        self._load_detector()

    def infer(self, praxis: Praxis, image: Physis, **options: Any) -> Theoria:
        if not self.supports(praxis):
            raise BackendUnavailableError(
                f"YuNetFaceAdapter does not support praxis {praxis.value!r}"
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
                        confidence=doxa(row[14]),
                        box=BoundingBox(x_min=x, y_min=y, x_max=x + w, y_max=y + h),
                        landmarks=marks,
                    )
                )
        except Sphalma:
            raise
        except Exception as exc:
            raise AdapterError(f"Face detection failed: {exc}") from exc
        return FaceResult(
            backend=self.name, latency_ms=(time.perf_counter() - start) * 1000, faces=faces
        )
