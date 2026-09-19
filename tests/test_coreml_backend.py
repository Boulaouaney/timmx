import platform
from pathlib import Path

import pytest

from timmx.errors import ConfigurationError
from timmx.export.coreml_backend import CoreMLBackend, ExportSource

ct = pytest.importorskip("coremltools")


def _build_kwargs(
    output_path: Path,
    *,
    convert_to: str = "mlprogram",
    dynamic_batch: bool = False,
    batch_size: int = 1,
    batch_upper_bound: int = 8,
    compute_precision: str | None = None,
    half: bool = False,
    int8: bool = False,
    int4: bool = False,
    normalize: bool = False,
    softmax: bool = False,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
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
        "half": half,
        "int8": int8,
        "int4": int4,
        "normalize": normalize,
        "softmax": softmax,
        "mean": mean,
        "std": std,
        "verify": verify,
    }


def _get_mlprogram_operations(output_path: Path) -> list:
    model = ct.models.MLModel(str(output_path), skip_model_load=True)
    spec = model.get_spec()
    return spec.mlProgram.functions["main"].block_specializations["CoreML5"].operations


def _has_const_float_values(operations: list, expected: tuple[float, ...] | list[float]) -> bool:
    expected_values = list(expected)
    for op in operations:
        if op.type != "const":
            continue
        values = list(op.attributes["val"].immediateValue.tensor.floats.values)
        if values == expected_values:
            return True
    return False


def test_export_coreml_mlprogram_and_verify(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.mlpackage"
    kwargs = _build_kwargs(output_path, compute_precision="float16")

    backend = CoreMLBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    loaded_model = ct.models.MLModel(str(output_path), skip_model_load=True)
    assert type(loaded_model).__name__ == "MLModel"


def test_export_coreml_trace_wraps_preprocessing_and_softmax(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_wrapped.mlpackage"
    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    kwargs = _build_kwargs(
        output_path,
        compute_precision="float32",
        normalize=True,
        softmax=True,
        mean=mean,
        std=std,
        verify=False,
    )

    CoreMLBackend().create_command()(**kwargs)

    operations = _get_mlprogram_operations(output_path)
    op_types = [op.type for op in operations]
    assert "sub" in op_types
    assert "mul" in op_types
    assert "softmax" in op_types
    assert _has_const_float_values(operations, mean)
    assert _has_const_float_values(operations, [8.0, 2.0, 4.0])


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


def test_rejects_output_suffix_that_does_not_match_convert_to(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=".mlpackage path for --convert-to mlprogram"):
        CoreMLBackend().create_command()(**_build_kwargs(tmp_path / "resnet18.mlmodel"))


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


def test_export_coreml_torch_export_source_wraps_preprocessing_and_softmax(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "resnet18_te_wrapped.mlpackage"
    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    kwargs = _build_kwargs(
        output_path,
        source="torch-export",
        compute_precision="float32",
        normalize=True,
        softmax=True,
        mean=mean,
        std=std,
        verify=False,
    )

    CoreMLBackend().create_command()(**kwargs)

    operations = _get_mlprogram_operations(output_path)
    op_types = [op.type for op in operations]
    assert "sub" in op_types
    assert "mul" in op_types
    assert "softmax" in op_types
    assert _has_const_float_values(operations, mean)
    assert _has_const_float_values(operations, [8.0, 2.0, 4.0])


def test_torch_export_dynamic_batch(tmp_path: Path) -> None:
    """torch-export dynamic batch preserves a runnable batch range starting at 1."""
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
    model = ct.models.MLModel(str(output_path), skip_model_load=True)
    spec = model.get_spec()
    batch_range = spec.description.input[0].type.multiArrayType.shapeRange.sizeRanges[0]
    assert batch_range.lowerBound == 1
    assert batch_range.upperBound == 8


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


def test_torch_export_dynamic_batch_rejects_upper_bound_below_batch_size(
    tmp_path: Path,
) -> None:
    """torch-export dynamic batch rejects upper bounds below the sample batch size."""
    output_path = tmp_path / "resnet18_te_invalid_upper.mlpackage"
    kwargs = _build_kwargs(
        output_path,
        source="torch-export",
        dynamic_batch=True,
        batch_size=2,
        batch_upper_bound=1,
    )

    backend = CoreMLBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--batch-upper-bound must be >= --batch-size."):
        command(**kwargs)


# --- quantization validation tests ---


def test_quantization_flags_are_mutually_exclusive(tmp_path: Path) -> None:
    """Only one of --half, --int8, --int4 can be specified."""
    output_path = tmp_path / "out.mlpackage"
    kwargs = _build_kwargs(output_path, half=True, int8=True)

    backend = CoreMLBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="Only one of"):
        command(**kwargs)


def test_int4_rejects_neuralnetwork(tmp_path: Path) -> None:
    """--int4 is only supported with --convert-to mlprogram."""
    output_path = tmp_path / "out.mlmodel"
    kwargs = _build_kwargs(output_path, convert_to="neuralnetwork", int4=True)

    backend = CoreMLBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--int4"):
        command(**kwargs)


# --- quantization export tests ---


def test_export_coreml_half_neuralnetwork(tmp_path: Path) -> None:
    """--half with neuralnetwork quantizes weights to float16."""
    output_path = tmp_path / "resnet18_half.mlmodel"
    kwargs = _build_kwargs(output_path, convert_to="neuralnetwork", half=True)

    CoreMLBackend().create_command()(**kwargs)
    assert output_path.exists()


def test_export_coreml_int8_neuralnetwork(tmp_path: Path) -> None:
    """--int8 with neuralnetwork quantizes weights to 8-bit linear_symmetric."""
    output_path = tmp_path / "resnet18_int8.mlmodel"
    kwargs = _build_kwargs(output_path, convert_to="neuralnetwork", int8=True)

    CoreMLBackend().create_command()(**kwargs)
    assert output_path.exists()


def test_export_coreml_half_mlprogram_is_noop(tmp_path: Path) -> None:
    """--half with mlprogram is a no-op (weights are already fp16)."""
    output_path = tmp_path / "resnet18_half.mlpackage"
    kwargs = _build_kwargs(output_path, half=True)

    CoreMLBackend().create_command()(**kwargs)
    assert output_path.exists()
    ct.models.MLModel(str(output_path), skip_model_load=True)


def _mil_op_types(model_path: Path) -> set[str]:
    spec = ct.models.MLModel(str(model_path), skip_model_load=True).get_spec()
    main = spec.mlProgram.functions["main"]
    return {op.type for block in main.block_specializations.values() for op in block.operations}


def test_export_coreml_int8_mlprogram(tmp_path: Path) -> None:
    """--int8 with mlprogram applies linear int8 weight quantization."""
    output_path = tmp_path / "resnet18_int8.mlpackage"
    kwargs = _build_kwargs(output_path, compute_precision="float16", int8=True)

    CoreMLBackend().create_command()(**kwargs)
    assert output_path.exists()
    assert "constexpr_affine_dequantize" in _mil_op_types(output_path)


def test_export_coreml_int4_mlprogram(tmp_path: Path) -> None:
    """--int4 with mlprogram palettizes weights to 4-bit."""
    output_path = tmp_path / "resnet18_int4.mlpackage"
    kwargs = _build_kwargs(output_path, compute_precision="float16", int4=True)

    CoreMLBackend().create_command()(**kwargs)
    assert output_path.exists()
    assert "constexpr_lut_to_dense" in _mil_op_types(output_path)


@pytest.mark.skipif(platform.system() != "Darwin", reason="Core ML inference needs macOS")
def test_verify_compute_units_reach_the_loaded_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[object] = []
    original = ct.models.MLModel.__init__

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("compute_units"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(ct.models.MLModel, "__init__", spy)
    kwargs = _build_kwargs(tmp_path / "resnet18.mlpackage") | {"verify_compute_units": "cpu"}
    CoreMLBackend().create_command()(**kwargs)

    assert ct.ComputeUnit.CPU_ONLY in seen


def test_default_source_is_torch_export(tmp_path: Path) -> None:
    import inspect

    command = CoreMLBackend().create_command()
    assert inspect.signature(command).parameters["source"].default == ExportSource.torch_export

    output = tmp_path / "default_source.mlpackage"
    kwargs = _build_kwargs(output)
    del kwargs["source"]
    command(**kwargs)
    spec = ct.models.MLModel(str(output), skip_model_load=True).get_spec()
    assert [feature.name for feature in spec.description.input] == ["input"]
    assert [feature.name for feature in spec.description.output] == ["output"]


# --- image input / classifier tests ---


def _write_labels(tmp_path: Path, count: int = 5) -> Path:
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("\n".join(f"class{i}" for i in range(count)) + "\n")
    return labels_path


def test_image_input_requires_normalize(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "out.mlpackage") | {"image_input": True}
    with pytest.raises(ConfigurationError, match="--image-input requires --normalize"):
        CoreMLBackend().create_command()(**kwargs)


def test_image_input_requires_batch_size_one(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "out.mlpackage", normalize=True, batch_size=2)
    with pytest.raises(ConfigurationError, match="--batch-size 1"):
        CoreMLBackend().create_command()(**kwargs | {"image_input": True})


def test_class_labels_must_match_model_outputs(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "out.mlpackage") | {
        "class_labels": _write_labels(tmp_path, count=3)
    }
    with pytest.raises(ConfigurationError, match="has 3 labels but the model has 1000 outputs"):
        CoreMLBackend().create_command()(**kwargs)


def test_class_labels_require_batch_size_one(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "out.mlpackage", batch_size=2) | {
        "class_labels": _write_labels(tmp_path, count=1000)
    }
    with pytest.raises(ConfigurationError, match="--class-labels requires --batch-size 1"):
        CoreMLBackend().create_command()(**kwargs)


def test_class_labels_must_be_unique(tmp_path: Path) -> None:
    labels = tmp_path / "dup.txt"
    labels.write_text("cat\ndog\ncat\n")
    kwargs = _build_kwargs(tmp_path / "out.mlpackage") | {"class_labels": labels}
    with pytest.raises(ConfigurationError, match="duplicate labels"):
        CoreMLBackend().create_command()(**kwargs)


def test_class_labels_file_must_not_be_empty(tmp_path: Path) -> None:
    empty = tmp_path / "empty.txt"
    empty.write_text("\n\n")
    kwargs = _build_kwargs(tmp_path / "out.mlpackage") | {"class_labels": empty}
    with pytest.raises(ConfigurationError, match="has no labels"):
        CoreMLBackend().create_command()(**kwargs)


@pytest.mark.parametrize(
    ("source", "convert_to"),
    [("torch-export", "mlprogram"), ("trace", "mlprogram"), ("trace", "neuralnetwork")],
)
def test_export_image_input_classifier(tmp_path: Path, source: str, convert_to: str) -> None:
    """--image-input + --class-labels: image feature in, classLabel/classLabel_probs out."""
    suffix = ".mlpackage" if convert_to == "mlprogram" else ".mlmodel"
    output_path = tmp_path / f"resnet18_{source}_classifier{suffix}"
    kwargs = _build_kwargs(
        output_path,
        source=source,
        convert_to=convert_to,
        compute_precision="float32" if convert_to == "mlprogram" else None,
        normalize=True,
        softmax=True,
    ) | {"num_classes": 5, "image_input": True, "class_labels": _write_labels(tmp_path)}

    CoreMLBackend().create_command()(**kwargs)

    spec = ct.models.MLModel(str(output_path), skip_model_load=True).get_spec()
    (model_input,) = spec.description.input
    assert model_input.name == "input"
    assert model_input.type.WhichOneof("Type") == "imageType"
    assert model_input.type.imageType.colorSpace == ct.proto.FeatureTypes_pb2.ImageFeatureType.RGB
    # mlprogram lists classLabel first, neuralnetwork the probabilities dict; names are what matter
    assert {o.name: o.type.WhichOneof("Type") for o in spec.description.output} == {
        "classLabel": "stringType",
        "classLabel_probs": "dictionaryType",
    }
    assert spec.description.predictedFeatureName == "classLabel"
    assert spec.description.predictedProbabilitiesName == "classLabel_probs"


def test_export_image_input_grayscale_keeps_output_name(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_gray.mlpackage"
    kwargs = _build_kwargs(output_path, compute_precision="float32", normalize=True) | {
        "in_chans": 1,
        "input_size": (1, 32, 32),
        "image_input": True,
    }

    CoreMLBackend().create_command()(**kwargs)

    spec = ct.models.MLModel(str(output_path), skip_model_load=True).get_spec()
    image_type = spec.description.input[0].type.imageType
    assert image_type.colorSpace == ct.proto.FeatureTypes_pb2.ImageFeatureType.GRAYSCALE
    assert (image_type.width, image_type.height) == (32, 32)
    assert [o.name for o in spec.description.output] == ["output"]
