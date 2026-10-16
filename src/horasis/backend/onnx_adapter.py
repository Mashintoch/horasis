"""ONNX Runtime backend for detection and classification (extra: ``horasis[onnx]``)."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from horasis.backend._common import doxa
from horasis.backend.mechane import Mechane
from horasis.foundation.imaging import load_rgb
from horasis.foundation.schemas import (
    BoundingBox,
    ClassificationEntry,
    ClassificationResult,
    Detection,
    DetectionResult,
    Physis,
    Praxis,
    Theoria,
)
from horasis.foundation.sphalma import (
    AdapterError,
    BackendUnavailableError,
    ConfigurationError,
    Sphalma,
)

OutputFormat = Literal["auto", "yolov8", "yolov5", "nms"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_PAD_VALUE = 114


def load_labels(model_path: str | Path, labels_path: str | Path | None = None) -> list[str]:
    """Load class names for a model from a sidecar file.

    Looks at `labels_path`, else next to the model for ``<stem>.labels.json``
    then ``<stem>.labels.txt``. A ``.txt`` file has one label per line; a
    ``.json`` file is a list, or an object mapping index -> name.

    Returns:
        Labels indexed by class id, or an empty list when no file exists.

    Raises:
        ConfigurationError: if a labels file exists but cannot be parsed.
    """
    model = Path(model_path)
    candidates = (
        [Path(labels_path)]
        if labels_path is not None
        else [model.with_suffix(".labels.json"), model.with_suffix(".labels.txt")]
    )
    for path in candidates:
        if not path.is_file():
            if labels_path is not None:
                raise ConfigurationError(f"Labels file not found: {path}")
            continue
        try:
            if path.suffix == ".json":
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    size = max(int(k) for k in raw) + 1 if raw else 0
                    names = [str(i) for i in range(size)]
                    for k, v in raw.items():
                        names[int(k)] = str(v)
                    return names
                return [str(x) for x in raw]
            return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        except (ValueError, TypeError, OSError) as exc:
            raise ConfigurationError(f"Could not parse labels file {path}: {exc}") from exc
    return []


def letterbox(
    image: Image.Image, size: tuple[int, int]
) -> tuple[np.ndarray[Any, Any], float, tuple[float, float]]:
    """Resize keeping aspect ratio and pad to `size` ``(height, width)``.

    Returns:
        ``(chw_float01, scale, (pad_x, pad_y))`` -- a ``(3, H, W)`` array in
        ``0..1`` plus what is needed to map boxes back to the original image.
    """
    height, width = size
    scale = min(height / image.height, width / image.width)
    new_w = max(1, round(image.width * scale))
    new_h = max(1, round(image.height * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (width, height), (_PAD_VALUE,) * 3)
    pad_x, pad_y = (width - new_w) // 2, (height - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1), scale, (float(pad_x), float(pad_y))


def center_crop_resize(
    image: Image.Image, size: tuple[int, int], crop_pct: float = 0.875
) -> np.ndarray[Any, Any]:
    """Classification preprocessing: resize short side, centre-crop, scale to 0..1 (CHW)."""
    height, width = size
    short = round(min(height, width) / crop_pct)
    scale = short / min(image.width, image.height)
    resized = image.resize(
        (max(width, round(image.width * scale)), max(height, round(image.height * scale))),
        Image.Resampling.BILINEAR,
    )
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    cropped = resized.crop((left, top, left + width, top + height))
    return (np.asarray(cropped, dtype=np.float32) / 255.0).transpose(2, 0, 1)


def nms(
    boxes: np.ndarray[Any, Any], scores: np.ndarray[Any, Any], iou_threshold: float
) -> list[int]:
    """Greedy non-maximum suppression over ``xyxy`` boxes; returns kept indices."""
    order = scores.argsort()[::-1]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if rest.size == 0:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


def softmax(logits: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return np.asarray(exp / exp.sum())


class OnnxAdapter(Mechane):
    """`Mechane` that runs a detection *or* classification ONNX model.

    One instance serves one model/task (register two instances with
    different names to serve both). Detection supports the common YOLO
    export layouts -- YOLOv8/11 ``(1, 4+nc, N)``, YOLOv5 ``(1, N, 5+nc)``,
    and end-to-end ``(1, N, 6)`` with NMS already applied -- selected by
    `output_format` (``"auto"`` guesses from the shape). Input is NCHW float.

    Args:
        model_path: Path to the ``.onnx`` file (opened lazily).
        task: ``"detect"`` or ``"classify"``.
        labels_path: Sidecar class names; otherwise discovered next to the
            model (see :func:`load_labels`). Without labels, ids are used.
        labels: Class names given directly (takes precedence over files).
        input_size: ``(height, width)`` or a single int, used when the model
            has dynamic input dims. Default 640 (detect) / 224 (classify).
        device: ``"auto"``, ``"cpu"`` or ``"cuda"``. ``"cuda"`` raises if
            CUDA is unavailable; ``"auto"`` uses it when present.
        providers: Explicit ORT provider list; overrides `device`.
        conf_threshold, iou_threshold, max_detections: Detection filtering.
        output_format: Detection output layout (see above).
        mean, std: Per-channel normalization after scaling to 0..1.
            Defaults: none for detection, ImageNet for classification.
        top_k: Number of classification entries to return.
        session: A pre-built ``InferenceSession``-like object (for tests).
        name: Registry name; defaults to ``"onnx"``.
    """

    def __init__(
        self,
        *,
        model_path: str | Path,
        task: Literal["detect", "classify"] = "detect",
        labels_path: str | Path | None = None,
        labels: Sequence[str] | None = None,
        input_size: int | tuple[int, int] | None = None,
        device: Literal["auto", "cpu", "cuda"] = "auto",
        providers: Sequence[str] | None = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 300,
        output_format: OutputFormat = "auto",
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        top_k: int = 5,
        session: Any = None,
        name: str = "onnx",
    ) -> None:
        if task not in ("detect", "classify"):
            raise ConfigurationError(f"OnnxAdapter task must be 'detect' or 'classify', not {task!r}")
        self.name = name
        self.supported_praxeis = frozenset({Praxis.DETECT if task == "detect" else Praxis.CLASSIFY})
        self._task = task
        self._model_path = Path(model_path)
        self._labels_path = labels_path
        self._labels: list[str] | None = list(labels) if labels is not None else None
        default_side = 640 if task == "detect" else 224
        side = input_size if input_size is not None else default_side
        self._input_size = (side, side) if isinstance(side, int) else (int(side[0]), int(side[1]))
        self._device = device
        self._providers = list(providers) if providers else None
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._max_det = max_detections
        self._output_format = output_format
        if mean is None and task == "classify":
            mean, std = IMAGENET_MEAN, IMAGENET_STD
        self._mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1) if mean else None
        self._std = np.asarray(std or (1.0, 1.0, 1.0), dtype=np.float32).reshape(3, 1, 1)
        self._top_k = top_k
        self._session: Any = session

    def _resolve_providers(self) -> list[str]:
        if self._providers:
            return self._providers
        import onnxruntime as ort

        available = set(ort.get_available_providers())
        if self._device == "cpu":
            return ["CPUExecutionProvider"]
        if self._device == "cuda" and "CUDAExecutionProvider" not in available:
            raise BackendUnavailableError(
                "device='cuda' requested but CUDAExecutionProvider is unavailable "
                "(install onnxruntime-gpu and a matching CUDA runtime)"
            )
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return [p for p in wanted if p in available]

    def _load_session(self) -> Any:
        if self._session is not None:
            return self._session
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendUnavailableError(
                "The 'onnxruntime' package is required for OnnxAdapter. "
                "Install it with: pip install 'horasis[onnx]'"
            ) from exc
        if not self._model_path.is_file():
            raise BackendUnavailableError(f"ONNX model file not found: {self._model_path}")
        self._session = ort.InferenceSession(
            str(self._model_path), providers=self._resolve_providers()
        )
        return self._session

    def warm_up(self) -> None:
        self._load_session()

    @property
    def labels(self) -> list[str]:
        if self._labels is None:
            self._labels = load_labels(self._model_path, self._labels_path)
        return self._labels

    def _label(self, class_id: int) -> str:
        names = self.labels
        return names[class_id] if 0 <= class_id < len(names) else str(class_id)

    def _model_input_size(self, session: Any) -> tuple[int, int]:
        shape = session.get_inputs()[0].shape
        if len(shape) != 4 or shape[1] != 3:
            raise ConfigurationError(f"Expected an NCHW image input (1, 3, H, W); model has {shape}")
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int):
            return h, w
        return self._input_size

    def infer(self, praxis: Praxis, image: Physis, **options: Any) -> Theoria:
        if not self.supports(praxis):
            raise BackendUnavailableError(
                f"OnnxAdapter {self.name!r} serves {sorted(p.value for p in self.supported_praxeis)}, "
                f"not praxis {praxis.value!r}"
            )
        start = time.perf_counter()
        try:
            session = self._load_session()
            rgb = load_rgb(image)
            size = self._model_input_size(session)
            if self._task == "detect":
                tensor, scale, pad = letterbox(rgb, size)
            else:
                tensor, scale, pad = center_crop_resize(rgb, size), 1.0, (0.0, 0.0)
            if self._mean is not None:
                tensor = (tensor - self._mean) / self._std
            inp = session.get_inputs()[0]
            dtype = np.float16 if "float16" in str(getattr(inp, "type", "")) else np.float32
            outputs = session.run(None, {inp.name: tensor[None].astype(dtype)})
            result: Theoria
            if self._task == "detect":
                result = self._decode_detect(
                    np.asarray(outputs[0], dtype=np.float32), scale, pad, rgb.size, options
                )
            else:
                result = self._decode_classify(np.asarray(outputs[0], dtype=np.float32), options)
        except Sphalma:
            raise
        except Exception as exc:
            raise AdapterError(f"ONNX inference failed: {exc}") from exc
        return result.model_copy(update={"latency_ms": (time.perf_counter() - start) * 1000})

    def _decode_classify(
        self, output: np.ndarray[Any, Any], options: dict[str, Any]
    ) -> ClassificationResult:
        logits = output.reshape(-1)
        is_prob = bool(np.all(logits >= 0) and abs(float(logits.sum()) - 1.0) < 1e-3)
        probs = logits if is_prob else softmax(logits)
        top_k = int(options.get("top_k", self._top_k))
        idx = np.argsort(probs)[::-1][:top_k]
        entries = [
            ClassificationEntry(label=self._label(int(i)), confidence=doxa(float(probs[i])))
            for i in idx
        ]
        return ClassificationResult(backend=self.name, latency_ms=0.0, entries=entries)

    def _guess_format(self, shape: tuple[int, ...]) -> OutputFormat:
        """Infer the detection output layout from its 2-D shape.

        A known class count settles it exactly; otherwise anchors (thousands)
        outnumber channels (tens), so the longer axis is the anchor axis.
        """
        n_classes = len(self.labels)
        if n_classes and shape[0] == 4 + n_classes:
            return "yolov8"
        if shape[0] < shape[1]:
            return "yolov8"
        return "nms" if shape[1] == 6 else "yolov5"

    def _decode_detect(
        self,
        output: np.ndarray[Any, Any],
        scale: float,
        pad: tuple[float, float],
        orig_size: tuple[int, int],
        options: dict[str, Any],
    ) -> DetectionResult:
        conf_thr = float(options.get("conf_threshold", self._conf))
        iou_thr = float(options.get("iou_threshold", self._iou))
        pred = output[0] if output.ndim == 3 else output
        fmt = self._output_format
        if fmt == "auto":
            fmt = self._guess_format(pred.shape)

        if fmt == "yolov8":
            rows = pred.T
            xyxy = _cxcywh_to_xyxy(rows[:, :4])
            class_scores = rows[:, 4:]
            cls = class_scores.argmax(axis=1)
            conf = class_scores.max(axis=1)
        elif fmt == "yolov5":
            xyxy = _cxcywh_to_xyxy(pred[:, :4])
            class_scores = pred[:, 5:] * pred[:, 4:5]
            cls = class_scores.argmax(axis=1)
            conf = class_scores.max(axis=1)
        else:  # "nms": [x1, y1, x2, y2, score, class], already suppressed
            xyxy, conf, cls = pred[:, :4], pred[:, 4], pred[:, 5].astype(int)

        keep = conf >= conf_thr
        xyxy, conf, cls = xyxy[keep], conf[keep], cls[keep]
        if fmt != "nms" and len(conf):
            # Offset per class so boxes of different classes never suppress each other.
            offset = cls[:, None].astype(np.float32) * 8192.0
            kept = nms(xyxy + offset, conf, iou_thr)
            xyxy, conf, cls = xyxy[kept], conf[kept], cls[kept]
        order = np.argsort(conf)[::-1][: self._max_det]
        xyxy, conf, cls = xyxy[order], conf[order], cls[order]

        width, height = orig_size
        detections: list[Detection] = []
        for box, score, class_id in zip(xyxy, conf, cls, strict=True):
            x1 = float(np.clip((box[0] - pad[0]) / scale, 0, width))
            y1 = float(np.clip((box[1] - pad[1]) / scale, 0, height))
            x2 = float(np.clip((box[2] - pad[0]) / scale, 0, width))
            y2 = float(np.clip((box[3] - pad[1]) / scale, 0, height))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append(
                Detection(
                    label=self._label(int(class_id)),
                    confidence=doxa(float(score)),
                    box=BoundingBox(x_min=x1, y_min=y1, x_max=x2, y_max=y2),
                )
            )
        return DetectionResult(backend=self.name, latency_ms=0.0, detections=detections)


def _cxcywh_to_xyxy(boxes: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return out
