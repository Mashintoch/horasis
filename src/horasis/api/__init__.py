"""Layer 5 -- Public API.

The only layer external callers should import from directly. Small,
explicit, and stable: it exposes `Skopos` and the five capability classes,
and hides every backend, orchestration, and access-control detail behind
them.
"""

from horasis.api.skopos import (
    ContentModerator,
    FaceDetector,
    ImageClassifier,
    ObjectDetector,
    Skopos,
    TextExtractor,
)

__all__ = [
    "ContentModerator",
    "FaceDetector",
    "ImageClassifier",
    "ObjectDetector",
    "Skopos",
    "TextExtractor",
]
