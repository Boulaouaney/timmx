from pathlib import Path

import pytest

from timmx.export.onnx_backend import OnnxBackend

onnx = pytest.importorskip("onnx")


def _build_kwargs(output_path: Path, dynamic_batch: bool = False, slim: bool = True) -> dict:
    return {
        "model_name": "resnet18",
        "output": output_path,
        "checkpoint": None,
        "pretrained": False,
        "num_classes": None,
        "in_chans": None,
        "batch_size": 1,
        "input_size": (3, 32, 32),
        "opset": 18,
        "dynamic_batch": dynamic_batch,
        "device": "cpu",
        "external_data": False,
        "check": True,
        "slim": slim,
    }


def test_export_onnx_and_validate_with_checker(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.onnx"
    kwargs = _build_kwargs(output_path)

    backend = OnnxBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    onnx.checker.check_model(str(output_path))


def test_dynamic_batch_creates_symbolic_batch_dimension(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_dynamic.onnx"
    kwargs = _build_kwargs(output_path, dynamic_batch=True)

    backend = OnnxBackend()
    command = backend.create_command()
    command(**kwargs)

    model = onnx.load(str(output_path))
    batch_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    assert batch_dim.dim_param != ""
