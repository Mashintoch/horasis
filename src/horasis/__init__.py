"""Computer vision SDK with API-key access control.

Applications import from this module and :mod:`horasis.api`. The
`foundation`, `access`, `backend`, `orchestration` and `plugins` packages
are for extension authors (a new `Backend`, a custom `Telemetry` sink) and
are not covered by the stability guarantee of the top-level API.

Quick start:
    >>> from horasis import Client, InMemoryApiKeyValidator, Tier
    >>> validator = InMemoryApiKeyValidator()
    >>> validator.register_simple("demo-key", principal_id="alice", tier=Tier.FREE)
    >>> client = Client.build(validator=validator)
    >>> result = client.detection.detect("photo.jpg", api_key="demo-key")
"""

from horasis.access.keys import ApiKeyValidator, HashedApiKeyValidator, InMemoryApiKeyValidator
from horasis.access.quota import UsageReport
from horasis.access.tiers import Tier
from horasis.api.async_client import AsyncClient
from horasis.api.client import (
    Client,
    ContentModerator,
    FaceDetector,
    ImageClassifier,
    ObjectDetector,
    TextExtractor,
)
from horasis.foundation.config import Config, InputPolicy
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
    QuotaExceededError,
    RateLimitedError,
    RemoteError,
    RevokedApiKeyError,
    ValidationError,
    VisualizationUnavailableError,
)
from horasis.foundation.schemas import (
    BatchItem,
    ClassificationResult,
    DetectionResult,
    Face,
    FaceResult,
    Image,
    ModerationResult,
    OcrResult,
    Task,
    TextRegion,
    Video,
    VideoDetectionResult,
)

__version__ = "1.0.0"

__all__ = [
    "AccessDeniedError",
    "AdapterError",
    "ApiKeyValidator",
    "AsyncClient",
    "BackendUnavailableError",
    "BatchItem",
    "ClassificationResult",
    "Client",
    "Config",
    "ConfigurationError",
    "ContentModerator",
    "DetectionResult",
    "ExpiredApiKeyError",
    "Face",
    "FaceDetector",
    "FaceResult",
    "HashedApiKeyValidator",
    "HorasisError",
    "Image",
    "ImageClassifier",
    "InMemoryApiKeyValidator",
    "InputPolicy",
    "InvalidApiKeyError",
    "ModelDownloadError",
    "ModerationResult",
    "NoBackendError",
    "ObjectDetector",
    "OcrResult",
    "QuotaExceededError",
    "RateLimitedError",
    "RemoteError",
    "RevokedApiKeyError",
    "Task",
    "TextExtractor",
    "TextRegion",
    "Tier",
    "UsageReport",
    "ValidationError",
    "Video",
    "VideoDetectionResult",
    "VisualizationUnavailableError",
    "__version__",
]
