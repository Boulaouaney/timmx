import pytest

from timmx.export.onnx_backend import OnnxBackend
from timmx.export.registry import BackendRegistry, create_builtin_registry


def test_builtin_registry_contains_onnx() -> None:
    registry = create_builtin_registry()
    assert registry.names() == ["onnx"]
    assert isinstance(registry.get("onnx"), OnnxBackend)


def test_registry_rejects_duplicate_backend_names() -> None:
    registry = BackendRegistry()
    registry.register(OnnxBackend())
    with pytest.raises(ValueError):
        registry.register(OnnxBackend())
