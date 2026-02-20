from pathlib import Path

import numpy as np
import pytest
import torch
from ai_edge_litert import interpreter as tfl_interpreter

from timmx.errors import ConfigurationError
from timmx.export.litert_backend import LiteRTBackend


class _ConvModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _build_kwargs(
    output_path: Path,
    *,
    mode: str = "fp32",
    nhwc_input: bool = False,
    calibration_data: Path | None = None,
    calibration_steps: int | None = None,
    verify: bool = True,
) -> dict:
    return {
        "model_name": "dummy",
        "output": output_path,
        "checkpoint": None,
        "pretrained": False,
        "num_classes": None,
        "in_chans": None,
        "batch_size": 2,
        "input_size": (3, 16, 16),
        "device": "cpu",
        "mode": mode,
        "calibration_data": calibration_data,
        "calibration_steps": calibration_steps,
        "nhwc_input": nhwc_input,
        "verify": verify,
    }


def _patch_model_helpers(monkeypatch: pytest.MonkeyPatch, model: torch.nn.Module) -> None:
    monkeypatch.setattr(
        "timmx.export.litert_backend.create_timm_model",
        lambda *_args, **_kwargs: model,
    )
    monkeypatch.setattr(
        "timmx.export.litert_backend.resolve_input_size",
        lambda _model, _requested: (3, 16, 16),
    )


@pytest.mark.parametrize(
    ("mode", "expected_dtype"),
    [
        ("fp32", np.float32),
        ("fp16", np.float16),
        ("dynamic-int8", np.int8),
        ("int8", np.int8),
    ],
)
def test_export_litert_modes_include_expected_tensor_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_dtype: type[np.generic],
) -> None:
    output_path = tmp_path / f"model_{mode}.tflite"
    kwargs = _build_kwargs(output_path, mode=mode)
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = LiteRTBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    interpreter = tfl_interpreter.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()
    dtypes = {detail["dtype"] for detail in interpreter.get_tensor_details()}
    assert expected_dtype in dtypes


def test_export_litert_nhwc_input_changes_input_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "model_nhwc.tflite"
    kwargs = _build_kwargs(output_path, mode="fp32", nhwc_input=True)
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = LiteRTBackend()
    command = backend.create_command()
    command(**kwargs)

    interpreter = tfl_interpreter.Interpreter(model_path=str(output_path))
    runner = interpreter.get_signature_runner("serving_default")
    input_name, input_details = next(iter(runner.get_input_details().items()))
    assert tuple(input_details["shape"]) == (2, 16, 16, 3)
    outputs = runner(**{input_name: np.random.randn(2, 16, 16, 3).astype(np.float32)})
    output_tensor = next(iter(outputs.values()))
    assert output_tensor.shape == (2, 4, 16, 16)


def test_export_litert_rejects_calibration_args_for_fp32(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "invalid_calibration_mode.tflite"
    calibration_path = tmp_path / "calibration.pt"
    torch.save(torch.randn(4, 3, 16, 16), calibration_path)

    kwargs = _build_kwargs(output_path, mode="fp32", calibration_data=calibration_path)

    backend = LiteRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError):
        command(**kwargs)


def test_export_litert_int8_with_calibration_data_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "model_int8_calibrated.tflite"
    calibration_path = tmp_path / "calibration.pt"
    torch.save(torch.randn(6, 3, 16, 16), calibration_path)

    kwargs = _build_kwargs(
        output_path,
        mode="int8",
        calibration_data=calibration_path,
        calibration_steps=2,
    )
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = LiteRTBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
