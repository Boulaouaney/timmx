from __future__ import annotations

from timmx.export.base import ExportBackend
from timmx.export.coreml_backend import CoreMLBackend
from timmx.export.litert_backend import LiteRTBackend
from timmx.export.onnx_backend import OnnxBackend
from timmx.export.torch_export_backend import TorchExportBackend


class BackendRegistry:
    """In-repo backend registry used by the CLI."""

    def __init__(self) -> None:
        self._backends: dict[str, ExportBackend] = {}

    def register(self, backend: ExportBackend) -> None:
        if backend.name in self._backends:
            raise ValueError(f"Backend {backend.name!r} is already registered.")
        self._backends[backend.name] = backend

    def get(self, name: str) -> ExportBackend:
        try:
            return self._backends[name]
        except KeyError as exc:
            raise KeyError(f"Unknown backend {name!r}.") from exc

    def items(self) -> list[tuple[str, ExportBackend]]:
        return sorted(self._backends.items(), key=lambda item: item[0])

    def names(self) -> list[str]:
        return [name for name, _ in self.items()]


def create_builtin_registry() -> BackendRegistry:
    registry = BackendRegistry()
    registry.register(CoreMLBackend())
    registry.register(LiteRTBackend())
    registry.register(OnnxBackend())
    registry.register(TorchExportBackend())
    return registry
