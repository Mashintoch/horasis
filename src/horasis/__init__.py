"""horasis -- a layered, plugin-friendly computer vision SDK.

The only supported import surface for external callers is this top-level
module and :mod:`horasis.api`. Everything under `foundation`, `access`,
`backend`, `orchestration`, and `plugins` is public for extension authors
(writing a new `Backend` backend, a custom `Telemetry` sink, etc.) but is
not part of the stability guarantee of the top-level API.

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

__version__ = "2.0.0"

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
