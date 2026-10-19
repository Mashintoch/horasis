"""Layer 1 -- Foundation.

The foundation layer has no dependencies on any other `horasis` layer.
It defines configuration (:class:`~horasis.foundation.config.Config`), the
exception hierarchy (:mod:`~horasis.foundation.errors`), result schemas
(:mod:`~horasis.foundation.schemas`), and telemetry hooks
(:mod:`~horasis.foundation.telemetry`) that every other layer builds on.
"""

from horasis.foundation.config import Config, InputPolicy, QuotaRule, TierRule
from horasis.foundation.errors import (
    AccessDeniedError,
    AdapterError,
    BackendUnavailableError,
    ConfigurationError,
    ExpiredApiKeyError,
    HorasisError,
    InvalidApiKeyError,
    ModelDownloadError,
    NoBackendError,
    PluginError,
    QuotaExceededError,
    RemoteError,
    RevokedApiKeyError,
    ValidationError,
    VisualizationUnavailableError,
)
from horasis.foundation.schemas import (
    BatchItem,
    BoundingBox,
    ClassificationEntry,
    ClassificationResult,
    Confidence,
    Detection,
    DetectionResult,
    Face,
    FaceResult,
    FrameDetections,
    Image,
    ModerationLabel,
    ModerationResult,
    OcrResult,
    Result,
    Task,
    TextRegion,
    Video,
    VideoDetectionResult,
)
from horasis.foundation.telemetry import (
    LoggingTelemetry,
    NullTelemetry,
    OTelTelemetry,
    Telemetry,
)

__all__ = [
    "AccessDeniedError",
    "AdapterError",
    "BackendUnavailableError",
    "BatchItem",
    "BoundingBox",
    "ClassificationEntry",
    "ClassificationResult",
    "Confidence",
    "Config",
    "ConfigurationError",
    "Detection",
    "DetectionResult",
    "ExpiredApiKeyError",
    "Face",
    "FaceResult",
    "FrameDetections",
    "HorasisError",
    "Image",
    "InputPolicy",
    "InvalidApiKeyError",
    "LoggingTelemetry",
    "ModelDownloadError",
    "ModerationLabel",
    "ModerationResult",
    "NoBackendError",
    "NullTelemetry",
    "OTelTelemetry",
    "OcrResult",
    "PluginError",
    "QuotaExceededError",
    "QuotaRule",
    "RemoteError",
    "Result",
    "RevokedApiKeyError",
    "Task",
    "Telemetry",
    "TextRegion",
    "TierRule",
    "ValidationError",
    "Video",
    "VideoDetectionResult",
    "VisualizationUnavailableError",
]
