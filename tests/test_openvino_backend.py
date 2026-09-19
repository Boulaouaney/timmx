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

    compiled = ov.Core().compile_model(str(output_path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    ov_out = compiled(x.numpy())[0]
    torch_out = wrapped(x).detach().numpy()
    np.testing.assert_allclose(ov_out, torch_out, atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(ov_out.sum(axis=-1), np.ones(2), atol=1e-4, rtol=0)


def _spy_compile(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    configs: list[object] = []
    original = ov.Core.compile_model

    def spy(self, model, device_name=None, config=None, *args, **kwargs):
        configs.append(config)
        return original(self, model, device_name, config, *args, **kwargs)

    monkeypatch.setattr(ov.Core, "compile_model", spy)
    return configs


def test_verify_uses_platform_default_when_it_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    configs = _spy_compile(monkeypatch)
    OpenVINOBackend().create_command()(**_build_kwargs(tmp_path / "resnet18.xml"))

    assert configs == [None]
    assert "INFERENCE_PRECISION_HINT" not in capsys.readouterr().out


def test_verify_retries_in_f32_and_notes_when_default_precision_diverges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from timmx.errors import ExportError
    from timmx.export import openvino_backend

    configs = _spy_compile(monkeypatch)
    monkeypatch.setattr(ov.Core, "get_property", lambda self, device, name: ov.Type.f16)
    calls: list[object] = []

    def fake_verify(expected, actual, *, backend):
        calls.append(actual)
        if len(calls) == 1:  # the platform-default pass "diverges"
            raise ExportError("diverges")

    monkeypatch.setattr(openvino_backend, "verify_outputs", fake_verify)
    OpenVINOBackend().create_command()(**_build_kwargs(tmp_path / "resnet18.xml"))

    assert configs == [None, {"INFERENCE_PRECISION_HINT": "f32"}]
    out = capsys.readouterr().out
    assert "only with INFERENCE_PRECISION_HINT=f32" in out
    assert "default is f16" in out


def test_verify_raises_when_f32_also_diverges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from timmx.errors import ExportError
    from timmx.export import openvino_backend

    monkeypatch.setattr(ov.Core, "get_property", lambda self, device, name: ov.Type.f16)

    def fake_verify(expected, actual, *, backend):
        raise ExportError("diverges")

    monkeypatch.setattr(openvino_backend, "verify_outputs", fake_verify)
    with pytest.raises(ExportError, match="diverges"):
        OpenVINOBackend().create_command()(**_build_kwargs(tmp_path / "resnet18.xml"))
