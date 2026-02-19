import pytest

from timmx.export.coreml_backend import CoreMLBackend
from timmx.export.litert_backend import LiteRTBackend
from timmx.export.onnx_backend import OnnxBackend
from timmx.export.registry import BackendRegistry, create_builtin_registry
from timmx.export.tensorrt_backend import TensorRTBackend
from timmx.export.torch_export_backend import TorchExportBackend


def test_builtin_registry_contains_all_backends() -> None:
    registry = create_builtin_registry()
    assert registry.names() == ["coreml", "litert", "onnx", "tensorrt", "torch-export"]
    assert isinstance(registry.get("coreml"), CoreMLBackend)
    assert isinstance(registry.get("litert"), LiteRTBackend)
    assert isinstance(registry.get("onnx"), OnnxBackend)
    assert isinstance(registry.get("tensorrt"), TensorRTBackend)
    assert isinstance(registry.get("torch-export"), TorchExportBackend)


def test_registry_rejects_duplicate_backend_names() -> None:
    registry = BackendRegistry()
    registry.register(OnnxBackend())
    with pytest.raises(ValueError):
        registry.register(OnnxBackend())
