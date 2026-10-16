"""Backend abstraction.

Defines the `Mechane` interface, concrete adapters, and the `Taxis`
registry used to resolve which adapter handles a given task. Depends only
on the foundation layer.
"""

from horasis.backend.easyocr_adapter import EasyOCRAdapter
from horasis.backend.face_adapter import YuNetFaceAdapter
from horasis.backend.mechane import Mechane
from horasis.backend.moderation_adapter import TransformersModerationAdapter
from horasis.backend.onnx_adapter import OnnxAdapter
from horasis.backend.remote_adapter import RemoteAdapter
from horasis.backend.taxis import Taxis
from horasis.backend.tesseract_adapter import TesseractAdapter
from horasis.backend.ultralytics_adapter import UltralyticsAdapter

__all__ = [
    "EasyOCRAdapter",
    "Mechane",
    "OnnxAdapter",
    "RemoteAdapter",
    "Taxis",
    "TesseractAdapter",
    "TransformersModerationAdapter",
    "UltralyticsAdapter",
    "YuNetFaceAdapter",
]
