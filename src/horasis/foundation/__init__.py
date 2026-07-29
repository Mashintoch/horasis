"""Layer 1 -- Foundation.

The foundation layer has no dependencies on any other `horasis` layer.
It defines configuration (:class:`~horasis.foundation.kanon.Kanon`), the
exception hierarchy (:mod:`~horasis.foundation.sphalma`), result schemas
(:mod:`~horasis.foundation.schemas`), and telemetry hooks
(:mod:`~horasis.foundation.aisthesis`) that every other layer builds on.
"""

from horasis.foundation.aisthesis import Aisthesis, LoggingAisthesis, NullAisthesis
from horasis.foundation.kanon import Kanon, QuotaRule, TierRule
from horasis.foundation.schemas import (
    BoundingBox,
    ClassificationEntry,
    ClassificationResult,
    Detection,
    DetectionResult,
    Doxa,
    FaceResult,
    FrameDetections,
    Glyph,
    Kinesis,
    ModerationLabel,
    ModerationResult,
    OcrResult,
    Physis,
    Praxis,
    Theoria,
    VideoDetectionResult,
)
from horasis.foundation.sphalma import (
    AccessDeniedError,
    AdapterError,
    BackendUnavailableError,
    ConfigurationError,
    ExpiredApiKeyError,
    HorasisError,
    InvalidApiKeyError,
    PluginError,
    QuotaExceededError,
    RevokedApiKeyError,
    Sphalma,
    ValidationError,
    VisualizationUnavailableError,
)

__all__ = [
    "AccessDeniedError",
    "AdapterError",
    "Aisthesis",
    "BackendUnavailableError",
    "BoundingBox",
    "ClassificationEntry",
    "ClassificationResult",
    "ConfigurationError",
    "Detection",
    "DetectionResult",
    "Doxa",
    "ExpiredApiKeyError",
    "FaceResult",
    "FrameDetections",
    "Glyph",
    "HorasisError",
    "InvalidApiKeyError",
    "Kanon",
    "Kinesis",
    "LoggingAisthesis",
    "ModerationLabel",
    "ModerationResult",
    "NullAisthesis",
    "OcrResult",
    "Physis",
    "PluginError",
    "Praxis",
    "QuotaExceededError",
    "QuotaRule",
    "RevokedApiKeyError",
    "Sphalma",
    "Theoria",
    "TierRule",
    "ValidationError",
    "VideoDetectionResult",
    "VisualizationUnavailableError",
]
