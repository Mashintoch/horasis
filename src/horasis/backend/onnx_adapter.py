from __future__ import annotations

from horasis.backend.mechane import Mechane
from horasis.foundation.schemas import Physis, Praxis, Theoria
from horasis.foundation.sphalma import BackendUnavailableError


class OnnxAdapter(Mechane):
    """Stub backend for ONNX Runtime models.

    Args:
        model_path: Path to an `.onnx` model file, loaded lazily on first use.
        name: Registry name; defaults to `"onnx"`.
    """

    supported_praxeis: frozenset[Praxis] = frozenset()

    def __init__(self, *, model_path: str, name: str = "onnx") -> None:
        self.name = name
        self._model_path = model_path

    def infer(self, praxis: Praxis, image: Physis, **options: object) -> Theoria:
        raise BackendUnavailableError(
            "OnnxAdapter is a placeholder and does not yet implement inference. "
            "Track progress in the horasis backend roadmap."
        )
