from __future__ import annotations

import time
from typing import Any

import numpy as np

from horasis.backend.base import Backend
from horasis.foundation.errors import AdapterError, BackendUnavailableError, HorasisError
from horasis.foundation.imaging import load_rgb
from horasis.foundation.schemas import (
    COCO_KEYPOINTS,
    BoundingBox,
    ClassificationEntry,
    ClassificationResult,
    Confidence,
    Detection,
    DetectionResult,
    FaceEstimate,
    FrameDetections,
    Image,
    Keypoint,
    Pose,
    PoseResult,
    Result,
    Segment,
    SegmentationResult,
    Task,
    Video,
    VideoDetectionResult,
)
from horasis.viz import _open_video_writer


class UltralyticsAdapter(Backend):
    """`Backend` backend powered by Ultralytics YOLO models.

    Args:
        detect_model: Path or model identifier for the detection model
            (e.g. `"yolo11n.pt"`).
        classify_model: Path or model identifier for the classification
            model (e.g. `"yolo11n-cls.pt"`).
        name: Registry name; defaults to `"ultralytics"`.
    """

    supported_tasks = frozenset(
        {Task.DETECT, Task.CLASSIFY, Task.SEGMENT, Task.POSE}
    )
    supported_video_tasks = frozenset({Task.DETECT})
    supported_video_annotate_tasks = frozenset({Task.DETECT})

    def __init__(
        self,
        *,
        detect_model: str = "yolo11n.pt",
        classify_model: str = "yolo11n-cls.pt",
        name: str = "ultralytics",
        segment_model: str = "yolo11n-seg.pt",
        pose_model: str = "yolo11n-pose.pt",
    ) -> None:
        self.name = name
        self._detect_model_path = detect_model
        self._classify_model_path = classify_model
        self._detect_model: Any = None
        self._classify_model: Any = None
        self._segment_model_path = segment_model
        self._pose_model_path = pose_model
        self._segment_model: Any = None
        self._pose_model: Any = None

    def _load_ultralytics(self) -> Any:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BackendUnavailableError(
                "The 'ultralytics' package is required for UltralyticsAdapter. "
                "Install it with: pip install 'horasis[ultralytics]'"
            ) from exc
        return YOLO

    def is_loaded(self) -> bool:
        return any(
            m is not None
            for m in (
                self._detect_model,
                self._classify_model,
                self._segment_model,
                self._pose_model,
            )
        )

    def unload(self) -> None:
        self._detect_model = None
        self._classify_model = None
        self._segment_model = None
        self._pose_model = None

    def _with_device(self, options: dict[str, Any]) -> dict[str, Any]:
        """Add Ultralytics' ``device`` option from the effective device (auto = leave to it)."""
        device = {"cpu": "cpu", "cuda": "cuda:0", "mps": "mps"}.get(self.effective_device())
        return options if device is None or "device" in options else {**options, "device": device}

    def warm_up(self) -> None:
        YOLO = self._load_ultralytics()
        if self._detect_model is None:
            self._detect_model = YOLO(self._detect_model_path)
        if self._classify_model is None:
            self._classify_model = YOLO(self._classify_model_path)

    def infer(self, task: Task, image: Image, **options: Any) -> Result:
        if not self.supports(task):
            raise BackendUnavailableError(
                f"UltralyticsAdapter does not support task {task.value!r}"
            )
        options = self._with_device(options)
        start = time.perf_counter()
        result: Result
        try:
            if task is Task.DETECT:
                result = self._infer_detect(image, **options)
            elif task is Task.CLASSIFY:
                result = self._infer_classify(image, **options)
            elif task is Task.SEGMENT:
                result = self._infer_segment(image, **options)
            else:
                result = self._infer_pose(image, **options)
        except HorasisError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"Ultralytics inference failed for task {task.value!r}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return result.model_copy(update={"latency_ms": latency_ms})

    def infer_video(self, task: Task, video: Video, **options: Any) -> Result:
        if not self.supports_video(task):
            raise BackendUnavailableError(
                f"UltralyticsAdapter does not support video for task {task.value!r}"
            )
        options = self._with_device(options)
        start = time.perf_counter()
        try:
            result = self._infer_detect_video(video, **options)
        except HorasisError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"Ultralytics video inference failed for task {task.value!r}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return result.model_copy(update={"latency_ms": latency_ms})

    def infer_video_and_annotate(
        self, task: Task, video: Video, *, output_path: Any, **options: Any
    ) -> tuple[Result, object]:
        if not self.supports_video_annotate(task):
            raise BackendUnavailableError(
                f"UltralyticsAdapter does not support combined video "
                f"detect-and-annotate for task {task.value!r}"
            )
        options = self._with_device(options)
        start = time.perf_counter()
        try:
            result, out_path = self._infer_detect_video_and_annotate(
                video, output_path=output_path, **options
            )
        except HorasisError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"Ultralytics video detect-and-annotate failed for task {task.value!r}: {exc}"
            ) from exc
        latency_ms = (time.perf_counter() - start) * 1000
        return result.model_copy(update={"latency_ms": latency_ms}), out_path

    def _infer_detect_video_and_annotate(
        self,
        video: Video,
        *,
        output_path: Any,
        show_labels: bool = True,
        show_confidence: bool = True,
        box_color: tuple[int, int, int] | None = None,
        text_color: tuple[int, int, int] = (255, 255, 255),
        line_width: int = 3,
        font_size: int = 16,
        estimate_demographics: bool = False,
        **options: Any,
    ) -> tuple[VideoDetectionResult, Any]:
        """Single decode pass: run inference and draw+write, reusing the
        same frame Ultralytics already decoded internally (`raw.orig_img`)
        instead of re-opening and re-reading the video a second time.
        """
        import os
        from pathlib import Path as _Path

        from horasis.viz import (
            _draw_boxes,
            _load_cv2,
        )

        cv2 = _load_cv2()
        from PIL import Image as PILImage

        YOLO = self._load_ultralytics()
        if self._detect_model is None:
            self._detect_model = YOLO(self._detect_model_path)

        sample_every_n_frames = int(options.pop("sample_every_n_frames", 1))
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
                        label = (
                            names.get(cls_id, str(cls_id))
                            if isinstance(names, dict)
                            else str(cls_id)
                        )
                        detection = Detection(
                            label=label,
                            confidence=Confidence(value=conf),
                            box=BoundingBox(
                                x_min=xyxy[0], y_min=xyxy[1], x_max=xyxy[2], y_max=xyxy[3]
                            ),
                        )
                        if estimate_demographics and label == "person":
                            estimate = self._estimate_face_in_box(orig_bgr, xyxy)
                            if estimate is not None:
                                detection = detection.model_copy(update={"face_estimate": estimate})
                        detections.append(detection)
                    frames.append(FrameDetections(frame_index=idx, detections=detections))

                frame_to_write = orig_bgr
                if detections:
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

    def _source_for(self, image: Image) -> Any:
        """Upright, validated RGB pixels (shared loader), as a PIL image."""
        return load_rgb(image)

    def _video_source_for(self, video: Video) -> tuple[Any, str | None]:
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
        raise AdapterError("Video has neither `path`, `url`, nor `data` set")

    def _infer_detect(
        self, image: Image, *, estimate_demographics: bool = False, **options: Any
    ) -> DetectionResult:
        YOLO = self._load_ultralytics()
        if self._detect_model is None:
            self._detect_model = YOLO(self._detect_model_path)
        raw = self._detect_model(self._source_for(image), verbose=False, **options)[0]

        detections: list[Detection] = []
        names = raw.names
        orig_bgr = raw.orig_img if estimate_demographics else None
        for box in raw.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            detection = Detection(
                label=label,
                confidence=Confidence(value=conf),
                box=BoundingBox(x_min=xyxy[0], y_min=xyxy[1], x_max=xyxy[2], y_max=xyxy[3]),
            )
            if estimate_demographics and label == "person" and orig_bgr is not None:
                estimate = self._estimate_face_in_box(orig_bgr, xyxy)
                if estimate is not None:
                    detection = detection.model_copy(update={"face_estimate": estimate})
            detections.append(detection)
        return DetectionResult(backend=self.name, latency_ms=0.0, detections=detections)

    def _load_deepface(self) -> Any:
        try:
            from deepface import DeepFace
        except ImportError as exc:
            raise BackendUnavailableError(
                "The 'deepface' package is required for demographic estimation. "
                "Install it with: pip install 'horasis[demographics]'"
            ) from exc
        return DeepFace

    def _estimate_face_in_box(self, orig_bgr: Any, xyxy: list[float]) -> FaceEstimate | None:
        """Crop the person's box out of the frame and estimate age/gender.

        Returns `None` (rather than raising) if no usable face is found in
        the crop -- a missing estimate on one detection shouldn't fail the
        whole call.
        """
        height, width = orig_bgr.shape[:2]
        x1, y1 = max(int(xyxy[0]), 0), max(int(xyxy[1]), 0)
        x2, y2 = min(int(xyxy[2]), width), min(int(xyxy[3]), height)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = orig_bgr[y1:y2, x1:x2]

        DeepFace = self._load_deepface()
        try:
            analysis = DeepFace.analyze(
                crop, actions=["age", "gender"], enforce_detection=False, silent=True
            )
        except (ValueError, RuntimeError, TypeError, KeyError, OSError):
            return None
        if not analysis:
            return None

        entry = analysis[0]
        gender_scores = entry.get("gender") or {}
        dominant_gender = entry.get("dominant_gender")
        gender_confidence = None
        if dominant_gender and dominant_gender in gender_scores:
            gender_confidence = Confidence(value=float(gender_scores[dominant_gender]) / 100.0)

        return FaceEstimate(
            estimated_age=entry.get("age"),
            estimated_gender=dominant_gender,
            gender_confidence=gender_confidence,
        )

    @staticmethod
    def _raw_boxes(raw: Any) -> list[tuple[str, float, list[float]]]:
        """``(label, confidence, xyxy)`` for each box in an Ultralytics result."""
        names = raw.names
        out: list[tuple[str, float, list[float]]] = []
        for box in raw.boxes:
            cls_id = int(box.cls[0])
            label = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
            out.append((label, float(box.conf[0]), box.xyxy[0].tolist()))
        return out

    def _infer_segment(self, image: Image, **options: Any) -> SegmentationResult:
        YOLO = self._load_ultralytics()
        if self._segment_model is None:
            self._segment_model = YOLO(self._segment_model_path)
        raw = self._segment_model(self._source_for(image), verbose=False, **options)[0]
        polygons = (
            []
            if raw.masks is None
            else [[(float(x), float(y)) for x, y in np.asarray(p)] for p in raw.masks.xy]
        )
        segments = [
            Segment(
                label=label,
                confidence=Confidence(value=conf),
                box=BoundingBox(x_min=xy[0], y_min=xy[1], x_max=xy[2], y_max=xy[3]),
                polygon=polygon,
            )
            for (label, conf, xy), polygon in zip(self._raw_boxes(raw), polygons, strict=False)
        ]
        return SegmentationResult(backend=self.name, latency_ms=0.0, segments=segments)

    def _infer_pose(self, image: Image, **options: Any) -> PoseResult:
        YOLO = self._load_ultralytics()
        if self._pose_model is None:
            self._pose_model = YOLO(self._pose_model_path)
        raw = self._pose_model(self._source_for(image), verbose=False, **options)[0]
        kp = raw.keypoints
        points = np.asarray(kp.xy) if kp is not None else np.zeros((0, 0, 2))
        scores = np.asarray(kp.conf) if kp is not None and kp.conf is not None else None
        poses: list[Pose] = []
        for i, (label, conf, xy) in enumerate(self._raw_boxes(raw)):
            person = points[i] if i < len(points) else []
            names = COCO_KEYPOINTS if len(person) == len(COCO_KEYPOINTS) else None
            keypoints = [
                Keypoint(
                    name=names[k] if names else f"kp{k}",
                    x=float(x),
                    y=float(y),
                    confidence=Confidence(value=float(scores[i][k]) if scores is not None else 1.0),
                )
                for k, (x, y) in enumerate(person)
            ]
            poses.append(
                Pose(
                    label=label,
                    confidence=Confidence(value=conf),
                    box=BoundingBox(x_min=xy[0], y_min=xy[1], x_max=xy[2], y_max=xy[3]),
                    keypoints=keypoints,
                )
            )
        return PoseResult(backend=self.name, latency_ms=0.0, poses=poses)

    def _infer_classify(self, image: Image, **options: Any) -> ClassificationResult:
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
                    ClassificationEntry(label=label, confidence=Confidence(value=float(score)))
                )
        return ClassificationResult(backend=self.name, latency_ms=0.0, entries=entries)

    def _infer_detect_video(
        self, video: Video, *, estimate_demographics: bool = False, **options: Any
    ) -> VideoDetectionResult:
        import os

        YOLO = self._load_ultralytics()
        if self._detect_model is None:
            self._detect_model = YOLO(self._detect_model_path)

        sample_every_n_frames = int(options.pop("sample_every_n_frames", 1))
        source, temp_file = self._video_source_for(video)

        frames: list[FrameDetections] = []
        try:
            stream = self._detect_model(
                source, stream=True, verbose=False, vid_stride=sample_every_n_frames, **options
            )
            for i, raw in enumerate(stream):
                actual_idx = i * sample_every_n_frames
                names = raw.names
                orig_bgr = raw.orig_img if estimate_demographics else None
                detections: list[Detection] = []
                for box in raw.boxes:
                    xyxy = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = (
                        names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(cls_id)
                    )
                    detection = Detection(
                        label=label,
                        confidence=Confidence(value=conf),
                        box=BoundingBox(x_min=xyxy[0], y_min=xyxy[1], x_max=xyxy[2], y_max=xyxy[3]),
                    )
                    if estimate_demographics and label == "person" and orig_bgr is not None:
                        estimate = self._estimate_face_in_box(orig_bgr, xyxy)
                        if estimate is not None:
                            detection = detection.model_copy(update={"face_estimate": estimate})
                    detections.append(detection)
                frames.append(FrameDetections(frame_index=actual_idx, detections=detections))
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
