from pathlib import Path

import pytest
import torch

from timmx.errors import ConfigurationError
from timmx.export.executorch_backend import ExecuTorchBackend


def _build_kwargs(
    output_path: Path,
    *,
    delegate: str = "xnnpack",
    mode: str = "fp32",
    batch_size: int = 1,
    dynamic_batch: bool = False,
    calibration_data: Path | None = None,
    calibration_steps: int | None = None,
    per_channel: bool = True,
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
        "device": "cpu",
        "delegate": delegate,
        "mode": mode,
        "dynamic_batch": dynamic_batch,
        "calibration_data": calibration_data,
        "calibration_steps": calibration_steps,
        "per_channel": per_channel,
    }


# ---------------------------------------------------------------------------
# Validation tests (no executorch needed)
# ---------------------------------------------------------------------------


def test_rejects_fp16_with_xnnpack(tmp_path: Path) -> None:
    backend = ExecuTorchBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--mode fp16.*--delegate coreml"):
        command(**_build_kwargs(tmp_path / "m.pte", mode="fp16", delegate="xnnpack"))


def test_rejects_int8_with_coreml(tmp_path: Path) -> None:
    backend = ExecuTorchBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--mode int8.*--delegate xnnpack"):
        command(**_build_kwargs(tmp_path / "m.pte", mode="int8", delegate="coreml"))


def test_rejects_calibration_args_without_int8(tmp_path: Path) -> None:
    cal = tmp_path / "cal.pt"
    torch.save(torch.randn(4, 3, 32, 32), cal)
    backend = ExecuTorchBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="only valid with --mode int8"):
        command(**_build_kwargs(tmp_path / "m.pte", mode="fp32", calibration_data=cal))


def test_rejects_dynamic_batch_with_batch_size_1(tmp_path: Path) -> None:
    backend = ExecuTorchBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--batch-size >= 2"):
        command(**_build_kwargs(tmp_path / "m.pte", dynamic_batch=True, batch_size=1))


# ---------------------------------------------------------------------------
# Runtime tests (skipped when executorch is not installed)
# ---------------------------------------------------------------------------

try:
    import executorch  # noqa: F401

    _has_executorch = True
except ImportError:
    _has_executorch = False

requires_executorch = pytest.mark.skipif(
    not _has_executorch,
    reason="executorch not installed",
)


@requires_executorch
def test_export_xnnpack_fp32(tmp_path: Path) -> None:
    output = tmp_path / "model.pte"
    backend = ExecuTorchBackend()
    command = backend.create_command()
    command(**_build_kwargs(output))
    assert output.exists()
    assert output.stat().st_size > 0


@requires_executorch
def test_export_xnnpack_dynamic_batch(tmp_path: Path) -> None:
    output = tmp_path / "model_dynamic.pte"
    backend = ExecuTorchBackend()
    command = backend.create_command()
    command(**_build_kwargs(output, dynamic_batch=True, batch_size=2))
    assert output.exists()
    assert output.stat().st_size > 0


# CoreML delegate tests require macOS + executorch CoreML support
try:
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner  # noqa: F401

    _has_coreml_delegate = True
except (ImportError, ModuleNotFoundError):
    _has_coreml_delegate = False

requires_coreml_delegate = pytest.mark.skipif(
    not (_has_executorch and _has_coreml_delegate),
    reason="executorch CoreML delegate not available",
)


@requires_coreml_delegate
def test_export_coreml_fp32(tmp_path: Path) -> None:
    output = tmp_path / "model_coreml.pte"
    backend = ExecuTorchBackend()
    command = backend.create_command()
    command(**_build_kwargs(output, delegate="coreml"))
    assert output.exists()
    assert output.stat().st_size > 0
