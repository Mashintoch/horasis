"""Layer 2: the `Backend` interface, its adapters and `BackendRegistry`.

Imports only from the foundation layer.
"""

from horasis.backend.base import Backend
from horasis.backend.easyocr_adapter import EasyOCRAdapter
from horasis.backend.face_adapter import YuNetFaceAdapter
from horasis.backend.moderation_adapter import TransformersModerationAdapter
from horasis.backend.onnx_adapter import OnnxAdapter
from horasis.backend.registry import BackendRegistry
from horasis.backend.remote_adapter import RemoteAdapter
from horasis.backend.tesseract_adapter import TesseractAdapter
from horasis.backend.ultralytics_adapter import UltralyticsAdapter

__all__ = [
    "Backend",
    "BackendRegistry",
    "EasyOCRAdapter",
    "OnnxAdapter",
    "RemoteAdapter",
    "TesseractAdapter",
    "TransformersModerationAdapter",
    "UltralyticsAdapter",
    "YuNetFaceAdapter",
]
