from pathlib import Path

import pytest

from timmx.errors import ConfigurationError
from timmx.export.coreml_backend import CoreMLBackend

ct = pytest.importorskip("coremltools")


def _build_kwargs(
    output_path: Path,
    *,
    convert_to: str = "mlprogram",
    dynamic_batch: bool = False,
    batch_size: int = 1,
    batch_upper_bound: int = 8,
    compute_precision: str | None = None,
    verify: bool = True,
    source: str = "trace",
) -> dict:
    return {
        "model_name": "resnet18",
        "output": output_path,
        "checkpoint": None,
        "pretrained": False,
        "num_classes": None,
        "in_chans": None,
        "batch_size": batch_size,
        "input_size": (3, 32, 32),
        "dynamic_batch": dynamic_batch,
        "batch_upper_bound": batch_upper_bound,
        "device": "cpu",
        "source": source,
        "convert_to": convert_to,
        "compute_precision": compute_precision,
        "verify": verify,
    }


def test_export_coreml_mlprogram_and_verify(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.mlpackage"
    kwargs = _build_kwargs(output_path, compute_precision="float16")

    backend = CoreMLBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    loaded_model = ct.models.MLModel(str(output_path), skip_model_load=True)
    assert type(loaded_model).__name__ == "MLModel"


def test_dynamic_batch_sets_shape_range(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_dynamic.mlpackage"
    kwargs = _build_kwargs(
        output_path,
        dynamic_batch=True,
        batch_size=2,
        batch_upper_bound=8,
        verify=False,
    )

    backend = CoreMLBackend()
    command = backend.create_command()
    command(**kwargs)

    model = ct.models.MLModel(str(output_path), skip_model_load=True)
    spec = model.get_spec()
    batch_range = spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[0]
    assert batch_range.lowerBound == 1
    assert batch_range.upperBound == 8


def test_neuralnetwork_rejects_compute_precision(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.mlmodel"
    kwargs = _build_kwargs(
        output_path,
        convert_to="neuralnetwork",
        compute_precision="float16",
    )

    backend = CoreMLBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError):
        command(**kwargs)


# --- torch-export source tests ---


def test_export_coreml_torch_export_source(tmp_path: Path) -> None:
    """torch-export source produces a valid mlpackage."""
    output_path = tmp_path / "resnet18_te.mlpackage"
    kwargs = _build_kwargs(output_path, source="torch-export", compute_precision="float16")

    backend = CoreMLBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    loaded_model = ct.models.MLModel(str(output_path), skip_model_load=True)
    assert type(loaded_model).__name__ == "MLModel"


def test_torch_export_dynamic_batch(tmp_path: Path) -> None:
    """torch-export source with dynamic batch produces a valid model."""
    output_path = tmp_path / "resnet18_te_dynamic.mlpackage"
    kwargs = _build_kwargs(
        output_path,
        source="torch-export",
        dynamic_batch=True,
        batch_size=2,
    )

    backend = CoreMLBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()


def test_torch_export_dynamic_batch_requires_batch_ge_2(tmp_path: Path) -> None:
    """torch-export dynamic batch rejects batch_size=1."""
    output_path = tmp_path / "resnet18_te_invalid.mlpackage"
    kwargs = _build_kwargs(
        output_path,
        source="torch-export",
        dynamic_batch=True,
        batch_size=1,
    )

    backend = CoreMLBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError):
        command(**kwargs)
