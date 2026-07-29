from __future__ import annotations

import time
from typing import Any

from horasis.backend.mechane import Mechane
from horasis.foundation.schemas import (
    BoundingBox,
    ClassificationEntry,
    ClassificationResult,
    Detection,
    DetectionResult,
    Doxa,
    FaceEstimate,
    FrameDetections,
    Kinesis,
    Physis,
    Praxis,
    Theoria,
    VideoDetectionResult,
)
from horasis.foundation.sphalma import AdapterError, BackendUnavailableError
from horasis.viz import _open_video_writer


class UltralyticsAdapter(Mechane):
    """`Mechane` backend powered by Ultralytics YOLO models.

    Args:
        detect_model: Path or model identifier for the detection model
            (e.g. `"yolo11n.pt"`).
        classify_model: Path or model identifier for the classification
            model (e.g. `"yolo11n-cls.pt"`).
        name: Registry name; defaults to `"ultralytics"`.
    """

    supported_praxeis = frozenset({Praxis.DETECT, Praxis.CLASSIFY})
    supported_video_praxeis = frozenset({Praxis.DETECT})
    supported_video_annotate_praxeis = frozenset({Praxis.DETECT})

    def __init__(
        self,
        *,
        detect_model: str = "yolo11n.pt",
        classify_model: str = "yolo11n-cls.pt",
        name: str = "ultralytics",
    ) -> None:
        self.name = name
        self._detect_model_path = detect_model
        self._classify_model_path = classify_model
        self._detect_model: Any = None
        self._classify_model: Any = None

    def _load_ultralytics(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BackendUnavailableError(
                "The 'ultralytics' package is required for UltralyticsAdapter. "
                "Install it with: pip install 'horasis[ultralytics]'"
            ) from exc
        return YOLO

    def warm_up(self) -> None:
        YOLO = self._load_ultralytics()
        if self._detect_model is None:
            self._detect_model = YOLO(self._detect_model_path)
        if self._classify_model is None:
            self._classify_model = YOLO(self._classify_model_path)

    def infer(self, praxis: Praxis, image: Physis, **options: object) -> Theoria:
        if not self.supports(praxis):
            raise BackendUnavailableError(
                f"UltralyticsAdapter does not support praxis {praxis.value!r}"
            )
        start = time.perf_counter()
        try:
            if praxis is Praxis.DETECT:
                result = self._infer_detect(image, **options)
            else:
                result = self._infer_classify(image, **options)
        except BackendUnavailableError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"Ultralytics inference failed for praxis {praxis.value!r}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return result.model_copy(update={"latency_ms": latency_ms})

    def infer_video(self, praxis: Praxis, video: Kinesis, **options: object) -> Theoria:
        if not self.supports_video(praxis):
            raise BackendUnavailableError(
                f"UltralyticsAdapter does not support video for praxis {praxis.value!r}"
            )
        start = time.perf_counter()
        try:
            result = self._infer_detect_video(video, **options)
        except BackendUnavailableError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"Ultralytics video inference failed for praxis {praxis.value!r}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return result.model_copy(update={"latency_ms": latency_ms})

    def infer_video_and_annotate(
        self, praxis: Praxis, video: Kinesis, *, output_path: object, **options: object
    ) -> tuple[Theoria, object]:
        if not self.supports_video_annotate(praxis):
            raise BackendUnavailableError(
                f"UltralyticsAdapter does not support combined video "
                f"detect-and-annotate for praxis {praxis.value!r}"
            )
        start = time.perf_counter()
        try:
            result, out_path = self._infer_detect_video_and_annotate(
                video, output_path=output_path, **options
            )
        except BackendUnavailableError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"Ultralytics video detect-and-annotate failed for praxis {praxis.value!r}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return result.model_copy(update={"latency_ms": latency_ms}), out_path

    def _infer_detect_video_and_annotate(
        self,
        video: Kinesis,
        *,
        output_path: object,
        show_labels: bool = True,
        show_confidence: bool = True,
        box_color: tuple[int, int, int] | None = None,
        text_color: tuple[int, int, int] = (255, 255, 255),
        line_width: int = 3,
        font_size: int = 16,
        **options: object,
    ) -> tuple[VideoDetectionResult, Any]:
        """Single decode pass: run inference and draw+write, reusing the
        same frame Ultralytics already decoded internally (`raw.orig_img`)
        instead of re-opening and re-reading the video a second time.
        """
        import os
        from pathlib import Path as _Path

        from horasis.viz import (  # noqa: PLC0415 -- lazy, optional `viz` extra
            _draw_boxes,
            _load_cv2,
        )

        cv2 = _load_cv2()
        from PIL import Image as PILImage  # noqa: PLC0415 -- lazy, optional `viz` extra

        YOLO = self._load_ultralytics()
        if self._detect_model is None:
            self._detect_model = YOLO(self._detect_model_path)

        sample_every_n_frames = int(options.pop("sample_every_n_frames", 1))  # type: ignore[arg-type]
        source, temp_file = self._video_source_for(video)

        probe = cv2.VideoCapture(source)
        fps = video.fps or (probe.get(cv2.CAP_PROP_FPS) or 30.0)
        probe.release()

        out_path = _Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        frames: list[FrameDetections] = []

        try:
            stream = self._detect_model(source, stream=True, verbose=False, **options)
            for idx, raw in enumerate(stream):
                orig_bgr = raw.orig_img
                if writer is None:
                    height, width = orig_bgr.shape[:2]
                    writer = _open_video_writer(cv2, str(out_path), fps, width, height)
                    
                detections: list[Detection] = []
                if idx % sample_every_n_frames == 0:
                    names = raw.names
                    for box in raw.boxes:
                        xyxy = box.xyxy[0].tolist()
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        detections.append(
                            Detection(
                                label=names.get(cls_id, str(cls_id))
                                if isinstance(names, dict)
                                else str(cls_id),
                                confidence=Doxa(value=conf),
                                box=BoundingBox(
                                    x_min=xyxy[0], y_min=xyxy[1], x_max=xyxy[2], y_max=xyxy[3]
                                ),
                            )
                        )
                    frames.append(FrameDetections(frame_index=idx, detections=detections))

                frame_to_write = orig_bgr
                if detections:
                    import numpy as np

                    frame_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
                    annotated_pil = _draw_boxes(
                        PILImage.fromarray(frame_rgb),
                        detections,
                        show_labels=show_labels,
                        show_confidence=show_confidence,
                        box_color=box_color,
                        text_color=text_color,
                        line_width=line_width,
                        font_size=font_size,
                    )
                    frame_to_write = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGB2BGR)

                writer.write(frame_to_write)
        finally:
            if writer is not None:
                writer.release()
            if temp_file is not None:
                os.unlink(temp_file)

        result = VideoDetectionResult(
            backend=self.name, latency_ms=0.0, frames=frames, frame_count=len(frames), fps=fps
        )
        return result, out_path

    def _source_for(self, image: Physis) -> Any:
        if image.path is not None:
            return str(image.path)
        if image.data is not None:
            import io

            from PIL import Image as PILImage

            return PILImage.open(io.BytesIO(image.data)).convert("RGB")
        raise AdapterError("Physis has neither `path` nor `data` set")

    def _video_source_for(self, video: Kinesis) -> tuple[Any, str | None]:
        """Return (source_for_ultralytics, temp_file_path_to_clean_up_or_None).

        Unlike images, videos are decoded frame-by-frame by Ultralytics
        itself, which needs an actual file path or URL -- not an in-memory
        buffer. A local path or URL is passed straight through (Ultralytics
        streams URLs natively); only explicit raw bytes need writing out to
        a temporary file first, which the caller is responsible for deleting.
        """
        if video.url is not None:
            return video.url, None
        if video.path is not None:
            return str(video.path), None
        if video.data is not None:
            import tempfile

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp:
                temp.write(video.data)
                return temp.name, temp.name
        raise AdapterError("Kinesis has neither `path`, `url`, nor `data` set")

    def _infer_detect(self, image: Physis, **options: object) -> DetectionResult:
        YOLO = self._load_ultralytics()
        if self._detect_model is None:
            self._detect_model = YOLO(self._detect_model_path)
        raw = self._detect_model(self._source_for(image), verbose=False, **options)[0]

        detections: list[Detection] = []
        names = raw.names
        for box in raw.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detections.append(
                Detection(
                    label=names.get(cls_id, str(cls_id))
                    if isinstance(names, dict)
                    else str(cls_id),
                    confidence=Doxa(value=conf),
                    box=BoundingBox(x_min=xyxy[0], y_min=xyxy[1], x_max=xyxy[2], y_max=xyxy[3]),
                )
            )
        return DetectionResult(backend=self.name, latency_ms=0.0, detections=detections)

    def _infer_classify(self, image: Physis, **options: object) -> ClassificationResult:
        YOLO = self._load_ultralytics()
        if self._classify_model is None:
            self._classify_model = YOLO(self._classify_model_path)
        raw = self._classify_model(self._source_for(image), verbose=False, **options)[0]

        entries: list[ClassificationEntry] = []
        names = raw.names
        probs = raw.probs
        if probs is not None:
            for idx, score in zip(probs.top5, probs.top5conf.tolist(), strict=False):
                label = names.get(idx, str(idx)) if isinstance(names, dict) else str(idx)
                entries.append(
                    ClassificationEntry(label=label, confidence=Doxa(value=float(score)))
                )
        return ClassificationResult(backend=self.name, latency_ms=0.0, entries=entries)

    def _infer_detect_video(self, video: Kinesis, **options: object) -> VideoDetectionResult:
        import os

        YOLO = self._load_ultralytics()
        if self._detect_model is None:
            self._detect_model = YOLO(self._detect_model_path)

        sample_every_n_frames = int(options.pop("sample_every_n_frames", 1))  # type: ignore[arg-type]
        source, temp_file = self._video_source_for(video)

        frames: list[FrameDetections] = []
        try:
            stream = self._detect_model(source, stream=True, verbose=False, **options)
            for idx, raw in enumerate(stream):
                if idx % sample_every_n_frames != 0:
                    continue
                names = raw.names
                detections: list[Detection] = []
                for box in raw.boxes:
                    xyxy = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    detections.append(
                        Detection(
                            label=names.get(cls_id, str(cls_id))
                            if isinstance(names, dict)
                            else str(cls_id),
                            confidence=Doxa(value=conf),
                            box=BoundingBox(
                                x_min=xyxy[0], y_min=xyxy[1], x_max=xyxy[2], y_max=xyxy[3]
                            ),
                        )
                    )
                frames.append(FrameDetections(frame_index=idx, detections=detections))
        finally:
            if temp_file is not None:
                os.unlink(temp_file)

        return VideoDetectionResult(
            backend=self.name,
            latency_ms=0.0,
            frames=frames,
            frame_count=len(frames),
            fps=video.fps,
        )
    
    def _load_deepface(self):
        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise BackendUnavailableError(
                "The 'deepface' package is required for demographic estimation. "
                "Install it with: pip install 'horasis[demographics]'"
            ) from exc
        return DeepFace

    def _estimate_face(self, crop_bgr) -> FaceEstimate | None:
        DeepFace = self._load_deepface()
        try:
            analysis = DeepFace.analyze(
                crop_bgr, actions=["age", "gender"], enforce_detection=False, silent=True
            )[0]
        except Exception:
            return None
        gender_scores = analysis.get("gender", {})
        top_gender = max(gender_scores, key=gender_scores.get) if gender_scores else None
        top_conf = gender_scores.get(top_gender, 0.0) / 100.0 if top_gender else None
        return FaceEstimate(
            estimated_age=analysis.get("age"),
            estimated_gender=top_gender,
            gender_confidence=Doxa(value=top_conf) if top_conf is not None else None,
        )
