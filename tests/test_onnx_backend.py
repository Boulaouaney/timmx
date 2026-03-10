import importlib
from pathlib import Path

import numpy as np
import pytest
import torch

from timmx.export.common import create_timm_model, wrap_with_preprocessing
from timmx.export.onnx_backend import OnnxBackend

onnx = pytest.importorskip("onnx")
numpy_helper = importlib.import_module("onnx.numpy_helper")
ReferenceEvaluator = importlib.import_module("onnx.reference").ReferenceEvaluator


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
        "normalize": False,
        "softmax": False,
        "mean": None,
        "std": None,
    }


def _constant_arrays(model: onnx.ModelProto) -> list[np.ndarray]:
    arrays = [numpy_helper.to_array(initializer) for initializer in model.graph.initializer]
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value":
                arrays.append(numpy_helper.to_array(attr.t))
    return arrays


def test_export_onnx_and_validate_with_checker(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.onnx"
    kwargs = _build_kwargs(output_path)

    backend = OnnxBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    onnx.checker.check_model(str(output_path))


def test_export_onnx_without_slim(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_no_slim.onnx"
    kwargs = _build_kwargs(output_path, slim=False)

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


def test_export_onnx_normalize_matches_wrapped_pytorch(tmp_path: Path) -> None:
    seed = 123
    x = torch.rand(1, 3, 32, 32)

    torch.manual_seed(seed)
    reference_model = create_timm_model(
        "resnet18",
        pretrained=False,
        checkpoint=None,
        num_classes=None,
        in_chans=None,
    ).eval()
    wrapped = wrap_with_preprocessing(reference_model).eval()

    output_path = tmp_path / "resnet18_normalize.onnx"
    kwargs = _build_kwargs(output_path, slim=False)
    kwargs["normalize"] = True

    torch.manual_seed(seed)
    OnnxBackend().create_command()(**kwargs)

    exported = onnx.load(str(output_path))
    assert "Softmax" not in {node.op_type for node in exported.graph.node}

    onnx_out = ReferenceEvaluator(exported).run(None, {"input": x.numpy()})[0]
    torch_out = wrapped(x).detach().numpy()
    np.testing.assert_allclose(onnx_out, torch_out, atol=1e-5, rtol=1e-5)


def test_export_onnx_softmax_and_custom_stats_round_trip(tmp_path: Path) -> None:
    seed = 456
    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    x = torch.rand(2, 3, 32, 32)

    torch.manual_seed(seed)
    reference_model = create_timm_model(
        "resnet18",
        pretrained=False,
        checkpoint=None,
        num_classes=None,
        in_chans=None,
    ).eval()
    wrapped = wrap_with_preprocessing(
        reference_model,
        normalize=True,
        softmax=True,
        mean=mean,
        std=std,
    ).eval()

    output_path = tmp_path / "resnet18_softmax.onnx"
    kwargs = _build_kwargs(output_path, slim=False)
    kwargs.update(
        {
            "batch_size": 2,
            "normalize": True,
            "softmax": True,
            "mean": mean,
            "std": std,
        }
    )

    torch.manual_seed(seed)
    OnnxBackend().create_command()(**kwargs)

    exported = onnx.load(str(output_path))
    assert "Softmax" in {node.op_type for node in exported.graph.node}

    constants = [
        array.flatten()
        for array in _constant_arrays(exported)
        if tuple(array.shape) == (1, 3, 1, 1)
    ]
    assert any(np.allclose(array, np.array(mean), atol=1e-6) for array in constants)
    assert any(np.allclose(array, np.array(std), atol=1e-6) for array in constants)

    onnx_out = ReferenceEvaluator(exported).run(None, {"input": x.numpy()})[0]
    torch_out = wrapped(x).detach().numpy()
    np.testing.assert_allclose(onnx_out, torch_out, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(onnx_out.sum(axis=-1), np.ones(2), atol=1e-5, rtol=0)
