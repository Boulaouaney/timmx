from pathlib import Path

import pytest
import torch

from timmx.errors import ConfigurationError
from timmx.export.tensorrt_backend import TensorRTBackend


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
    device: str = "cuda",
    workspace: int = 4,
    dynamic_batch: bool = False,
    batch_min: int = 1,
    batch_max: int = 16,
    batch_size: int = 1,
    opset: int = 18,
    calibration_data: Path | None = None,
    calibration_steps: int | None = None,
    calibration_cache: Path | None = None,
    keep_onnx: bool = False,
    verbose: bool = False,
) -> dict:
    return {
        "model_name": "resnet18",
        "output": output_path,
        "checkpoint": None,
        "pretrained": False,
        "num_classes": None,
        "in_chans": None,
        "batch_size": batch_size,
        "input_size": (3, 16, 16),
        "device": device,
        "mode": mode,
        "workspace": workspace,
        "opset": opset,
        "dynamic_batch": dynamic_batch,
        "batch_min": batch_min,
        "batch_max": batch_max,
        "calibration_data": calibration_data,
        "calibration_steps": calibration_steps,
        "calibration_cache": calibration_cache,
        "keep_onnx": keep_onnx,
        "verbose": verbose,
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


@pytest.fixture()
def _fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend CUDA is available so validation tests can reach TRT-specific checks."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)


# ---------------------------------------------------------------------------
# Validation tests (run on any platform, no GPU needed)
# ---------------------------------------------------------------------------


def test_tensorrt_rejects_cpu_device(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.engine", device="cpu")
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--device cuda"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_calibration_args_for_fp32(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.pt"
    torch.save(torch.randn(4, 3, 16, 16), calibration_path)
    kwargs = _build_kwargs(
        tmp_path / "model.engine",
        mode="fp32",
        calibration_data=calibration_path,
    )
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="only valid with --mode int8"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_calibration_args_for_fp16(tmp_path: Path) -> None:
    kwargs = _build_kwargs(
        tmp_path / "model.engine",
        mode="fp16",
        calibration_steps=5,
    )
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="only valid with --mode int8"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_invalid_workspace(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.engine", workspace=0)
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--workspace"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_invalid_opset(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.engine", opset=5)
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--opset"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_dynamic_batch_with_batch_size_1(tmp_path: Path) -> None:
    kwargs = _build_kwargs(
        tmp_path / "model.engine",
        dynamic_batch=True,
        batch_size=1,
    )
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--batch-size must be >= 2"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_batch_max_lt_batch_size(tmp_path: Path) -> None:
    kwargs = _build_kwargs(
        tmp_path / "model.engine",
        dynamic_batch=True,
        batch_size=8,
        batch_max=4,
    )
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--batch-max"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_batch_min_gt_batch_size(tmp_path: Path) -> None:
    kwargs = _build_kwargs(
        tmp_path / "model.engine",
        dynamic_batch=True,
        batch_size=2,
        batch_min=4,
    )
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--batch-min"):
        command(**kwargs)


# ---------------------------------------------------------------------------
# GPU-dependent tests (skipped when CUDA or tensorrt is not available)
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TensorRT export requires NVIDIA GPU with CUDA",
)

try:
    import tensorrt as _trt  # noqa: F401

    _has_tensorrt = True
except ImportError:
    _has_tensorrt = False

requires_tensorrt = pytest.mark.skipif(
    not _has_tensorrt,
    reason="tensorrt package not installed",
)

try:
    import onnxscript as _onnxscript  # noqa: F401

    _has_onnxscript = True
except ImportError:
    _has_onnxscript = False

requires_onnxscript = pytest.mark.skipif(
    not _has_onnxscript,
    reason="onnxscript package not installed",
)


@requires_cuda
@requires_tensorrt
@requires_onnxscript
def test_export_tensorrt_fp32(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.engine"
    kwargs = _build_kwargs(output_path, mode="fp32")
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = TensorRTBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@requires_cuda
@requires_tensorrt
@requires_onnxscript
def test_export_tensorrt_fp16(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.engine"
    kwargs = _build_kwargs(output_path, mode="fp16")
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = TensorRTBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@requires_cuda
@requires_tensorrt
@requires_onnxscript
def test_export_tensorrt_dynamic_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.engine"
    kwargs = _build_kwargs(
        output_path,
        mode="fp32",
        dynamic_batch=True,
        batch_size=2,
        batch_min=1,
        batch_max=8,
    )
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = TensorRTBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@requires_cuda
@requires_tensorrt
@requires_onnxscript
def test_export_tensorrt_int8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.engine"
    kwargs = _build_kwargs(output_path, mode="int8")
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = TensorRTBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@requires_cuda
@requires_tensorrt
@requires_onnxscript
def test_export_tensorrt_keep_onnx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.engine"
    kwargs = _build_kwargs(output_path, keep_onnx=True)
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = TensorRTBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    onnx_path = output_path.with_suffix(".onnx")
    assert onnx_path.exists()
