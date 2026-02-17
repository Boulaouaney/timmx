import argparse
from pathlib import Path

import coremltools as ct
import pytest

from timmx.errors import ConfigurationError
from timmx.export.coreml_backend import CoreMLBackend


def _build_args(
    output_path: Path,
    *,
    convert_to: str = "mlprogram",
    dynamic_batch: bool = False,
    batch_size: int = 1,
    batch_upper_bound: int = 8,
    compute_precision: str | None = None,
    verify: bool = True,
) -> argparse.Namespace:
    return argparse.Namespace(
        model_name="resnet18",
        output=output_path,
        checkpoint=None,
        pretrained=False,
        num_classes=None,
        in_chans=None,
        batch_size=batch_size,
        input_size=[3, 32, 32],
        dynamic_batch=dynamic_batch,
        batch_upper_bound=batch_upper_bound,
        device="cpu",
        convert_to=convert_to,
        compute_precision=compute_precision,
        verify=verify,
        exportable=True,
    )


def test_export_coreml_mlprogram_and_verify(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.mlpackage"
    args = _build_args(output_path, compute_precision="float16")

    backend = CoreMLBackend()
    exit_code = backend.run(args)

    assert exit_code == 0
    assert output_path.exists()
    loaded_model = ct.models.MLModel(str(output_path), skip_model_load=True)
    assert type(loaded_model).__name__ == "MLModel"


def test_dynamic_batch_sets_shape_range(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_dynamic.mlpackage"
    args = _build_args(
        output_path,
        dynamic_batch=True,
        batch_size=2,
        batch_upper_bound=8,
        verify=False,
    )

    backend = CoreMLBackend()
    exit_code = backend.run(args)

    assert exit_code == 0
    model = ct.models.MLModel(str(output_path), skip_model_load=True)
    spec = model.get_spec()
    batch_range = spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[0]
    assert batch_range.lowerBound == 1
    assert batch_range.upperBound == 8


def test_neuralnetwork_rejects_compute_precision(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.mlmodel"
    args = _build_args(
        output_path,
        convert_to="neuralnetwork",
        compute_precision="float16",
    )

    backend = CoreMLBackend()
    with pytest.raises(ConfigurationError):
        backend.run(args)
