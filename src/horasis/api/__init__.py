"""Layer 5 -- Public API.

The only layer external callers should import from directly. Small,
explicit, and stable: it exposes `Client` and the five capability classes,
and hides every backend, orchestration, and access-control detail behind
them.
"""

from horasis.api.async_client import AsyncClient
from horasis.api.client import (
    Client,
    ContentModerator,
    FaceDetector,
    ImageClassifier,
    ObjectDetector,
    TextExtractor,
)

__all__ = [
    "AsyncClient",
    "Client",
    "ContentModerator",
    "FaceDetector",
    "ImageClassifier",
    "ObjectDetector",
    "TextExtractor",
]
