from pathlib import Path

import numpy as np
import pytest
import torch

from timmx.errors import ConfigurationError
from timmx.export.common import create_timm_model, wrap_with_preprocessing
from timmx.export.openvino_backend import OpenVINOBackend

ov = pytest.importorskip("openvino")


def _build_kwargs(output_path: Path, **overrides: object) -> dict:
    kwargs: dict[str, object] = {
        "model_name": "resnet18",
        "output": output_path,
        "checkpoint": None,
        "pretrained": False,
        "num_classes": None,
        "in_chans": None,
        "batch_size": 1,
        "input_size": (3, 32, 32),
        "dynamic_batch": False,
        "device": "cpu",
        "fp16": True,
        "verify": True,
        "normalize": False,
        "softmax": False,
        "mean": None,
        "std": None,
    }
    kwargs.update(overrides)
    return kwargs


def test_export_openvino_writes_ir_and_verifies(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.xml"
    OpenVINOBackend().create_command()(**_build_kwargs(output_path))

    assert output_path.exists()
    assert output_path.with_suffix(".bin").exists()
    model = ov.Core().read_model(str(output_path))
    assert model.inputs[0].any_name == "input"
    assert model.outputs[0].any_name == "output"
    assert list(model.inputs[0].shape) == [1, 3, 32, 32]


def test_rejects_non_xml_output(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=".xml"):
        OpenVINOBackend().create_command()(**_build_kwargs(tmp_path / "resnet18.bin"))


@pytest.mark.parametrize("batch_size", [1, 2])
def test_dynamic_batch_marks_batch_dimension_dynamic(tmp_path: Path, batch_size: int) -> None:
    output_path = tmp_path / "resnet18_dynamic.xml"
    OpenVINOBackend().create_command()(
        **_build_kwargs(output_path, batch_size=batch_size, dynamic_batch=True)
    )

    model = ov.Core().read_model(str(output_path))
    shape = model.inputs[0].get_partial_shape()
    assert shape[0].is_dynamic
    assert [int(shape[i].get_length()) for i in range(1, 4)] == [3, 32, 32]

    compiled = ov.Core().compile_model(model, "CPU")
    assert compiled(np.random.rand(3, 3, 32, 32).astype(np.float32))[0].shape[0] == 3


def test_export_openvino_normalize_softmax_matches_wrapped_pytorch(tmp_path: Path) -> None:
    seed = 456
    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    x = torch.rand(2, 3, 32, 32)

    torch.manual_seed(seed)
    reference_model = create_timm_model(
        "resnet18", pretrained=False, checkpoint=None, num_classes=None, in_chans=None
    ).eval()
    wrapped = wrap_with_preprocessing(
        reference_model, normalize=True, softmax=True, mean=mean, std=std
    ).eval()

    output_path = tmp_path / "resnet18_wrapped.xml"
    torch.manual_seed(seed)
    OpenVINOBackend().create_command()(
        **_build_kwargs(
            output_path,
            batch_size=2,
            fp16=False,
            normalize=True,
            softmax=True,
            mean=mean,
            std=std,
        )
    )

    compiled = ov.Core().compile_model(str(output_path), "CPU")
    ov_out = compiled(x.numpy())[0]
    torch_out = wrapped(x).detach().numpy()
    np.testing.assert_allclose(ov_out, torch_out, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(ov_out.sum(axis=-1), np.ones(2), atol=1e-4, rtol=0)
