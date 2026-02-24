import pytest

from timmx.export.base import DependencyStatus
from timmx.export.coreml_backend import CoreMLBackend
from timmx.export.litert_backend import LiteRTBackend
from timmx.export.ncnn_backend import NcnnBackend
from timmx.export.onnx_backend import OnnxBackend
from timmx.export.registry import BackendRegistry, create_builtin_registry
from timmx.export.tensorrt_backend import TensorRTBackend
from timmx.export.torch_export_backend import TorchExportBackend
from timmx.export.torchscript_backend import TorchScriptBackend


def test_builtin_registry_contains_all_backends() -> None:
    registry = create_builtin_registry()
    assert registry.names() == [
        "coreml",
        "litert",
        "ncnn",
        "onnx",
        "tensorrt",
        "torch-export",
        "torchscript",
    ]
    assert isinstance(registry.get("coreml"), CoreMLBackend)
    assert isinstance(registry.get("litert"), LiteRTBackend)
    assert isinstance(registry.get("ncnn"), NcnnBackend)
    assert isinstance(registry.get("onnx"), OnnxBackend)
    assert isinstance(registry.get("tensorrt"), TensorRTBackend)
    assert isinstance(registry.get("torch-export"), TorchExportBackend)
    assert isinstance(registry.get("torchscript"), TorchScriptBackend)


def test_all_backends_return_dependency_status() -> None:
    registry = create_builtin_registry()
    for name, backend in registry.items():
        status = backend.check_dependencies()
        assert isinstance(status, DependencyStatus), f"{name} returned wrong type"
        assert isinstance(status.available, bool)
        assert isinstance(status.missing_packages, list)
        assert isinstance(status.install_hint, str)


def test_registry_rejects_duplicate_backend_names() -> None:
    registry = BackendRegistry()
    registry.register(OnnxBackend())
    with pytest.raises(ValueError):
        registry.register(OnnxBackend())
