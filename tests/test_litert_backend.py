import argparse
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


class _VectorModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        return torch.flatten(x, 1)


def _build_args(
    output_path: Path,
    *,
    mode: str = "fp32",
    nhwc_output: bool = False,
    verify: bool = True,
) -> argparse.Namespace:
    return argparse.Namespace(
        model_name="dummy",
        output=output_path,
        checkpoint=None,
        pretrained=False,
        num_classes=None,
        in_chans=None,
        batch_size=2,
        input_size=[3, 16, 16],
        device="cpu",
        mode=mode,
        nhwc_output=nhwc_output,
        verify=verify,
        exportable=True,
    )


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
    args = _build_args(output_path, mode=mode)
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = LiteRTBackend()
    exit_code = backend.run(args)

    assert exit_code == 0
    assert output_path.exists()
    interpreter = tfl_interpreter.Interpreter(model_path=str(output_path))
    interpreter.allocate_tensors()
    dtypes = {detail["dtype"] for detail in interpreter.get_tensor_details()}
    assert expected_dtype in dtypes


def test_export_litert_nhwc_output_changes_output_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "model_nhwc.tflite"
    args = _build_args(output_path, mode="fp32", nhwc_output=True)
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = LiteRTBackend()
    exit_code = backend.run(args)

    assert exit_code == 0
    interpreter = tfl_interpreter.Interpreter(model_path=str(output_path))
    runner = interpreter.get_signature_runner("serving_default")
    input_key = next(iter(runner.get_input_details().keys()))
    outputs = runner(**{input_key: np.random.randn(1, 3, 16, 16).astype(np.float32)})
    output_tensor = next(iter(outputs.values()))
    assert output_tensor.shape == (1, 16, 16, 4)


def test_export_litert_nhwc_output_rejects_rank_2_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "invalid_nhwc.tflite"
    args = _build_args(output_path, mode="fp32", nhwc_output=True)
    _patch_model_helpers(monkeypatch, _VectorModel().eval())

    backend = LiteRTBackend()
    with pytest.raises(ConfigurationError):
        backend.run(args)
