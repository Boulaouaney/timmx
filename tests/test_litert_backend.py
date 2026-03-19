import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from timmx.errors import ConfigurationError
from timmx.export.common import wrap_with_preprocessing
from timmx.export.litert_backend import LiteRTBackend

pytest.importorskip("ai_edge_litert")
from ai_edge_litert import interpreter as tfl_interpreter  # noqa: E402


class _ConvModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ClassifierModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _build_kwargs(
    output_path: Path,
    *,
    batch_size: int = 2,
    mode: str = "fp32",
    nhwc_input: bool = False,
    calibration_data: Path | None = None,
    calibration_steps: int | None = None,
    calibration_samples: int | None = None,
    random_calibration: bool = False,
    verify: bool = True,
    normalize: bool = False,
    softmax: bool = False,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> dict:
    return {
        "model_name": "dummy",
        "output": output_path,
        "checkpoint": None,
        "pretrained": False,
        "num_classes": None,
        "in_chans": None,
        "batch_size": batch_size,
        "input_size": (3, 16, 16),
        "device": "cpu",
        "mode": mode,
        "calibration_data": calibration_data,
        "calibration_steps": calibration_steps,
        "calibration_samples": calibration_samples,
        "random_calibration": random_calibration,
        "nhwc_input": nhwc_input,
        "verify": verify,
        "normalize": normalize,
        "softmax": softmax,
        "mean": mean,
        "std": std,
    }


def _patch_model_helpers(monkeypatch: pytest.MonkeyPatch, model: torch.nn.Module) -> None:
    monkeypatch.setattr(
        "timmx.export.common.create_timm_model",
        lambda *_args, **_kwargs: model,
    )
    monkeypatch.setattr(
        "timmx.export.common.resolve_input_size",
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
    needs_random_cal = mode in {"dynamic-int8", "int8"}
    kwargs = _build_kwargs(output_path, mode=mode, random_calibration=needs_random_cal)
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


def test_rejects_mean_std_without_wrapper_flags_outside_int8(tmp_path: Path) -> None:
    backend = LiteRTBackend()
    command = backend.create_command()
    with pytest.raises(
        ConfigurationError,
        match="--mean/--std require --normalize unless used for --mode dynamic-int8 or --mode int8 calibration",
    ):
        command(
            **_build_kwargs(
                tmp_path / "invalid_mean_std.tflite",
                mode="fp32",
                mean=(0.5, 0.25, 0.75),
                std=(0.125, 0.5, 0.25),
            )
        )


def test_allows_mean_std_for_int8_calibration_without_wrapper_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _FakeEdgeModel:
        def export(self, path: str) -> None:
            Path(path).write_bytes(b"tfl3")

    class _FakeLiteRTTorch:
        @staticmethod
        def convert(*_args, **_kwargs) -> _FakeEdgeModel:
            return _FakeEdgeModel()

    def fake_resolve_calibration_batches(**kwargs):
        captured.update(kwargs)
        return [torch.randn(2, 3, 16, 16)]

    _patch_model_helpers(monkeypatch, _ConvModel().eval())
    monkeypatch.setattr(
        "timmx.export.litert_backend.resolve_calibration_batches",
        fake_resolve_calibration_batches,
    )
    monkeypatch.setattr(
        "timmx.export.litert_backend._prepare_pt2e_quantized_module",
        lambda model, example_input, *, calibration_batches, is_dynamic: (model, object()),
    )
    monkeypatch.setattr(
        "timmx.export.litert_backend._import_litert_torch",
        lambda: _FakeLiteRTTorch(),
    )

    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    output = tmp_path / "model_int8_mean_std.tflite"
    LiteRTBackend().create_command()(
        **_build_kwargs(
            output,
            mode="int8",
            random_calibration=True,
            verify=False,
            mean=mean,
            std=std,
        )
    )

    assert output.exists()
    assert captured["mean"] == mean
    assert captured["std"] == std
    assert captured["normalize_images"] is True


def test_int8_wrapper_disables_image_normalization_for_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _FakeEdgeModel:
        def export(self, path: str) -> None:
            Path(path).write_bytes(b"tfl3")

    class _FakeLiteRTTorch:
        @staticmethod
        def convert(*_args, **_kwargs) -> _FakeEdgeModel:
            return _FakeEdgeModel()

    def fake_resolve_calibration_batches(**kwargs):
        captured.update(kwargs)
        return [torch.randn(2, 3, 16, 16)]

    _patch_model_helpers(monkeypatch, _ConvModel().eval())
    monkeypatch.setattr(
        "timmx.export.litert_backend.resolve_calibration_batches",
        fake_resolve_calibration_batches,
    )
    monkeypatch.setattr(
        "timmx.export.litert_backend._prepare_pt2e_quantized_module",
        lambda model, example_input, *, calibration_batches, is_dynamic: (model, object()),
    )
    monkeypatch.setattr(
        "timmx.export.litert_backend._import_litert_torch",
        lambda: _FakeLiteRTTorch(),
    )

    output = tmp_path / "model_int8_wrapped.tflite"
    LiteRTBackend().create_command()(
        **_build_kwargs(
            output,
            mode="int8",
            random_calibration=True,
            verify=False,
            normalize=True,
            softmax=True,
            mean=(0.5, 0.25, 0.75),
            std=(0.125, 0.5, 0.25),
        )
    )

    assert output.exists()
    assert captured["normalize_images"] is False


def test_int8_softmax_only_keeps_image_normalization_for_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _FakeEdgeModel:
        def export(self, path: str) -> None:
            Path(path).write_bytes(b"tfl3")

    class _FakeLiteRTTorch:
        @staticmethod
        def convert(*_args, **_kwargs) -> _FakeEdgeModel:
            return _FakeEdgeModel()

    def fake_resolve_calibration_batches(**kwargs):
        captured.update(kwargs)
        return [torch.randn(2, 3, 16, 16)]

    _patch_model_helpers(monkeypatch, _ConvModel().eval())
    monkeypatch.setattr(
        "timmx.export.litert_backend.resolve_calibration_batches",
        fake_resolve_calibration_batches,
    )
    monkeypatch.setattr(
        "timmx.export.litert_backend._prepare_pt2e_quantized_module",
        lambda model, example_input, *, calibration_batches, is_dynamic: (model, object()),
    )
    monkeypatch.setattr(
        "timmx.export.litert_backend._import_litert_torch",
        lambda: _FakeLiteRTTorch(),
    )

    output = tmp_path / "model_int8_softmax_only.tflite"
    LiteRTBackend().create_command()(
        **_build_kwargs(
            output,
            mode="int8",
            random_calibration=True,
            verify=False,
            softmax=True,
        )
    )

    assert output.exists()
    assert captured["normalize_images"] is True


def test_export_litert_fp32_wraps_preprocessing_and_softmax(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "model_wrapped.tflite"
    export_model = _ClassifierModel().eval()
    reference_model = copy.deepcopy(export_model).eval()
    _patch_model_helpers(monkeypatch, export_model)

    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    backend = LiteRTBackend()
    command = backend.create_command()
    command(
        **_build_kwargs(
            output_path,
            mode="fp32",
            batch_size=1,
            normalize=True,
            softmax=True,
            mean=mean,
            std=std,
        )
    )

    interpreter = tfl_interpreter.Interpreter(model_path=str(output_path))
    runner = interpreter.get_signature_runner("serving_default")
    input_name = next(iter(runner.get_input_details()))
    example_input = torch.rand(1, 3, 16, 16)
    outputs = runner(**{input_name: example_input.numpy().astype(np.float32)})
    actual = torch.from_numpy(next(iter(outputs.values())))
    expected = wrap_with_preprocessing(
        reference_model,
        normalize=True,
        softmax=True,
        mean=mean,
        std=std,
    ).eval()(example_input)

    assert output_path.exists()
    assert torch.allclose(actual, expected, atol=5e-4, rtol=1e-4)


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


def test_pt2e_quantized_module_is_in_eval_mode() -> None:
    from timmx.export.litert_backend import _prepare_pt2e_quantized_module

    model = _ConvModel().eval()
    example_input = torch.randn(2, 3, 16, 16)
    calibration_batches = [torch.randn(2, 3, 16, 16)]

    quantized, _ = _prepare_pt2e_quantized_module(
        model,
        example_input,
        calibration_batches=calibration_batches,
        is_dynamic=False,
    )
    assert quantized.training is False
