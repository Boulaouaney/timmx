import struct
from pathlib import Path

import pytest

from timmx.errors import ConfigurationError
from timmx.export.ncnn_backend import NcnnBackend

pnnx = pytest.importorskip("pnnx")


def _build_kwargs(output_dir: Path, **overrides: object) -> dict:
    defaults: dict[str, object] = {
        "model_name": "resnet18",
        "output": output_dir,
        "checkpoint": None,
        "pretrained": False,
        "num_classes": None,
        "in_chans": None,
        "batch_size": 1,
        "input_size": (3, 32, 32),
        "device": "cpu",
        "fp16": True,
        "normalize": False,
        "softmax": False,
        "mean": None,
        "std": None,
    }
    defaults.update(overrides)
    return defaults


def _read_param(output_dir: Path) -> str:
    return (output_dir / "model.ncnn.param").read_text()


def _has_float_sequence(output_dir: Path, expected: tuple[float, ...]) -> bool:
    data = (output_dir / "model.ncnn.bin").read_bytes()
    values = struct.unpack("<" + "f" * (len(data) // 4), data[: 4 * (len(data) // 4)])
    width = len(expected)
    for index in range(len(values) - width + 1):
        window = values[index : index + width]
        if all(abs(actual - target) < 1e-6 for actual, target in zip(window, expected)):
            return True
    return False


def test_export_ncnn_creates_param_and_bin(tmp_path: Path) -> None:
    output_dir = tmp_path / "ncnn_out"
    kwargs = _build_kwargs(output_dir)

    backend = NcnnBackend()
    command = backend.create_command()
    command(**kwargs)

    assert (output_dir / "model.ncnn.param").exists()
    assert (output_dir / "model.ncnn.bin").exists()
    assert (output_dir / "model_ncnn.py").exists()


def test_export_ncnn_cleans_up_pnnx_intermediates(tmp_path: Path) -> None:
    output_dir = tmp_path / "ncnn_out"
    kwargs = _build_kwargs(output_dir)

    backend = NcnnBackend()
    command = backend.create_command()
    command(**kwargs)

    assert not (output_dir / "model.pt").exists()
    assert not (output_dir / "model.pnnx.param").exists()
    assert not (output_dir / "model.pnnx.bin").exists()
    assert not (output_dir / "model_pnnx.py").exists()
    assert not (output_dir / "model.pnnx.onnx").exists()
    assert not (output_dir / "__pycache__").exists()


def test_export_ncnn_fp32(tmp_path: Path) -> None:
    output_dir = tmp_path / "ncnn_fp32"
    kwargs = _build_kwargs(output_dir, fp16=False)

    backend = NcnnBackend()
    command = backend.create_command()
    command(**kwargs)

    assert (output_dir / "model.ncnn.param").exists()
    assert (output_dir / "model.ncnn.bin").exists()
    assert (output_dir / "model_ncnn.py").exists()


def test_export_ncnn_wraps_preprocessing_and_softmax(tmp_path: Path) -> None:
    output_dir = tmp_path / "ncnn_wrapped"
    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    kwargs = _build_kwargs(
        output_dir,
        fp16=False,
        normalize=True,
        softmax=True,
        mean=mean,
        std=std,
    )

    NcnnBackend().create_command()(**kwargs)

    param_text = _read_param(output_dir)
    assert "MemoryData               std" in param_text
    assert "MemoryData               mean" in param_text
    assert "BinaryOp                 sub_" in param_text
    assert "BinaryOp                 div_" in param_text
    assert "Softmax" in param_text
    assert _has_float_sequence(output_dir, std)
    assert _has_float_sequence(output_dir, mean)


def test_export_ncnn_rejects_mean_std_without_wrapper_flags(tmp_path: Path) -> None:
    output_dir = tmp_path / "ncnn_invalid"
    kwargs = _build_kwargs(
        output_dir,
        mean=(0.5, 0.25, 0.75),
        std=(0.125, 0.5, 0.25),
    )

    with pytest.raises(ConfigurationError, match="--mean/--std require --normalize"):
        NcnnBackend().create_command()(**kwargs)


def test_ncnn_backend_check_dependencies() -> None:
    backend = NcnnBackend()
    status = backend.check_dependencies()
    assert status.available
    assert status.missing_packages == []
