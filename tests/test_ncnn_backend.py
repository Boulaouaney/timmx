from pathlib import Path

import pytest

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
    }
    defaults.update(overrides)
    return defaults


def test_export_ncnn_creates_param_and_bin(tmp_path: Path) -> None:
    output_dir = tmp_path / "ncnn_out"
    kwargs = _build_kwargs(output_dir)

    backend = NcnnBackend()
    command = backend.create_command()
    command(**kwargs)

    assert (output_dir / "model.ncnn.param").exists()
    assert (output_dir / "model.ncnn.bin").exists()


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


def test_ncnn_backend_check_dependencies() -> None:
    backend = NcnnBackend()
    status = backend.check_dependencies()
    assert status.available
    assert status.missing_packages == []
