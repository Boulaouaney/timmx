import argparse
from pathlib import Path

import onnx

from timmx.export.onnx_backend import OnnxBackend


def _build_args(output_path: Path, dynamic_batch: bool = False) -> argparse.Namespace:
    return argparse.Namespace(
        model_name="resnet18",
        output=output_path,
        checkpoint=None,
        pretrained=False,
        num_classes=None,
        in_chans=None,
        batch_size=1,
        input_size=[3, 32, 32],
        opset=18,
        dynamic_batch=dynamic_batch,
        device="cpu",
        external_data=False,
        check=True,
        exportable=True,
    )


def test_export_onnx_and_validate_with_checker(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.onnx"
    args = _build_args(output_path)

    backend = OnnxBackend()
    exit_code = backend.run(args)

    assert exit_code == 0
    assert output_path.exists()
    onnx.checker.check_model(str(output_path))


def test_dynamic_batch_creates_symbolic_batch_dimension(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_dynamic.onnx"
    args = _build_args(output_path, dynamic_batch=True)

    backend = OnnxBackend()
    exit_code = backend.run(args)

    assert exit_code == 0
    model = onnx.load(str(output_path))
    batch_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
    assert batch_dim.dim_param != ""
