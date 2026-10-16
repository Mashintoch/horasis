"""horasis -- a layered, plugin-friendly computer vision SDK.

The only supported import surface for external callers is this top-level
module and :mod:`horasis.api`. Everything under `foundation`, `access`,
`backend`, `orchestration`, and `plugins` is public for extension authors
(writing a new `Mechane` backend, a custom `Aisthesis` sink, etc.) but is
not part of the stability guarantee of the top-level API.

Quick start:
    >>> from horasis import Skopos, InMemoryApiKeyValidator, Klimax
    >>> validator = InMemoryApiKeyValidator()
    >>> validator.register_simple("demo-key", principal_id="alice", tier=Klimax.FREE)
    >>> client = Skopos.build(validator=validator)
    >>> result = client.detection.detect("photo.jpg", api_key="demo-key")
"""

from horasis.access.keys import ApiKeyValidator, InMemoryApiKeyValidator
from horasis.access.klimax import Klimax
from horasis.api.skopos import (
    ContentModerator,
    FaceDetector,
    ImageClassifier,
    ObjectDetector,
    Skopos,
    TextExtractor,
)
from horasis.foundation.kanon import InputPolicy, Kanon
from horasis.foundation.schemas import (
    ClassificationResult,
    DetectionResult,
    Face,
    FaceResult,
    Glyph,
    Kinesis,
    ModerationResult,
    OcrResult,
    Physis,
    Praxis,
    VideoDetectionResult,
)
from horasis.foundation.sphalma import (
    AccessDeniedError,
    AdapterError,
    BackendUnavailableError,
    ConfigurationError,
    ExpiredApiKeyError,
    InvalidApiKeyError,
    NoBackendError,
    QuotaExceededError,
    RemoteError,
    RevokedApiKeyError,
    Sphalma,
    ValidationError,
    VisualizationUnavailableError,
)

__version__ = "1.0.0"

__all__ = [
    "AccessDeniedError",
    "AdapterError",
    "ApiKeyValidator",
    "BackendUnavailableError",
    "ClassificationResult",
    "ConfigurationError",
    "ContentModerator",
    "DetectionResult",
    "ExpiredApiKeyError",
    "Face",
    "FaceDetector",
    "FaceResult",
    "Glyph",
    "ImageClassifier",
    "InMemoryApiKeyValidator",
    "InputPolicy",
    "InvalidApiKeyError",
    "Kanon",
    "Kinesis",
    "Klimax",
    "ModerationResult",
    "NoBackendError",
    "ObjectDetector",
    "OcrResult",
    "Physis",
    "Praxis",
    "QuotaExceededError",
    "RemoteError",
    "RevokedApiKeyError",
    "Skopos",
    "Sphalma",
    "TextExtractor",
    "ValidationError",
    "VideoDetectionResult",
    "VisualizationUnavailableError",
    "__version__",
]
