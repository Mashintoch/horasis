"""Layer 1 -- Foundation.

The foundation layer has no dependencies on any other `horasis` layer.
It defines configuration (:class:`~horasis.foundation.kanon.Kanon`), the
exception hierarchy (:mod:`~horasis.foundation.sphalma`), result schemas
(:mod:`~horasis.foundation.schemas`), and telemetry hooks
(:mod:`~horasis.foundation.aisthesis`) that every other layer builds on.
"""

from horasis.foundation.aisthesis import Aisthesis, LoggingAisthesis, NullAisthesis
from horasis.foundation.kanon import InputPolicy, Kanon, QuotaRule, TierRule
from horasis.foundation.schemas import (
    BoundingBox,
    ClassificationEntry,
    ClassificationResult,
    Detection,
    DetectionResult,
    Doxa,
    Face,
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
    NoBackendError,
    PluginError,
    QuotaExceededError,
    RemoteError,
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
    "Face",
    "FaceResult",
    "FrameDetections",
    "Glyph",
    "HorasisError",
    "InputPolicy",
    "InvalidApiKeyError",
    "Kanon",
    "Kinesis",
    "LoggingAisthesis",
    "ModerationLabel",
    "ModerationResult",
    "NoBackendError",
    "NullAisthesis",
    "OcrResult",
    "Physis",
    "PluginError",
    "Praxis",
    "QuotaExceededError",
    "QuotaRule",
    "RemoteError",
    "RevokedApiKeyError",
    "Sphalma",
    "Theoria",
    "TierRule",
    "ValidationError",
    "VideoDetectionResult",
    "VisualizationUnavailableError",
]
