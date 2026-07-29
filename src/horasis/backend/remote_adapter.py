from __future__ import annotations

from horasis.backend.mechane import Mechane
from horasis.foundation.schemas import Physis, Praxis, Theoria
from horasis.foundation.sphalma import BackendUnavailableError


class RemoteAdapter(Mechane):
    """Stub backend for remote HTTP inference services.

    Args:
        endpoint_url: Base URL of the remote inference service.
        api_key: Credential for the remote service (not a `horasis` API key).
        name: Registry name; defaults to `"remote"`.
    """

    supported_praxeis: frozenset[Praxis] = frozenset()

    def __init__(
        self, *, endpoint_url: str, api_key: str | None = None, name: str = "remote"
    ) -> None:
        self.name = name
        self._endpoint_url = endpoint_url
        self._api_key = api_key

    def infer(self, praxis: Praxis, image: Physis, **options: object) -> Theoria:
        raise BackendUnavailableError(
            "RemoteAdapter is a placeholder and does not yet implement inference. "
            "Track progress in the horasis backend roadmap."
        )
