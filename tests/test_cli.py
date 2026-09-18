import re

from typer.testing import CliRunner

from timmx.cli import app
from timmx.export import create_builtin_registry

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI escape codes so assertions work in CI (Rich emits them on GitHub Actions)."""
    return _ANSI_RE.sub("", text)


def test_root_help_lists_commands() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    output = _plain(result.output)
    for command in ("export", "info", "doctor", "list"):
        assert command in output


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "timmx" in _plain(result.output)


def test_export_help_lists_backends() -> None:
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    output = _plain(result.output)
    for name in create_builtin_registry().names():
        assert name in output


def test_export_coreai_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "coreai", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--dynamic-batch" in _plain(result.output)
    assert "--verify" in _plain(result.output)
    assert "--normalize" in _plain(result.output)
    assert "--softmax" in _plain(result.output)


def test_export_executorch_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "executorch", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--delegate" in _plain(result.output)
    assert "--mode" in _plain(result.output)
    assert "compute-preci" in _plain(result.output)
    assert "--dynamic-batch" in _plain(result.output)
    assert "--batch-upper-b" in _plain(result.output)
    assert "--calibration-d" in _plain(result.output)
    assert "--per-channel" in _plain(result.output)
    assert "--verify" in _plain(result.output)
    assert "--normalize" in _plain(result.output)
    assert "--softmax" in _plain(result.output)
    assert "--mean" in _plain(result.output)
    assert "--std" in _plain(result.output)


def test_export_onnx_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "onnx", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--opset" in _plain(result.output)
    assert "--dynamic-batch" in _plain(result.output)
    assert "--check" in _plain(result.output)
    assert "--verify" in _plain(result.output)


def test_export_openvino_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "openvino", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--dynamic-batch" in _plain(result.output)
    assert "--fp16" in _plain(result.output)
    assert "--verify" in _plain(result.output)
    assert "--normalize" in _plain(result.output)
    assert "--softmax" in _plain(result.output)


def test_export_coreml_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "coreml", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--convert-to" in _plain(result.output)
    assert "compute-preci" in _plain(result.output)
    assert "--dynamic-batch" in _plain(result.output)
    assert "--source" in _plain(result.output)
    assert "--half" in _plain(result.output)
    assert "--int8" in _plain(result.output)
    assert "--int4" in _plain(result.output)
    assert "--normalize" in _plain(result.output)
    assert "--softmax" in _plain(result.output)
    assert "--mean" in _plain(result.output)
    assert "--std" in _plain(result.output)


def test_export_litert_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "litert", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--mode" in _plain(result.output)
    assert "--calibration-d" in _plain(result.output)
    assert "--nhwc-input" in _plain(result.output)
    assert "--per-channel" in _plain(result.output)
    assert "--normalize" in _plain(result.output)
    assert "--softmax" in _plain(result.output)
    assert "--mean" in _plain(result.output)
    assert "--std" in _plain(result.output)


def test_export_tensorrt_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "tensorrt", "--help"])
    assert result.exit_code == 0
    assert "--verify" in _plain(result.output)
    assert "--output" in _plain(result.output)
    assert "--mode" in _plain(result.output)
    assert "--workspace" in _plain(result.output)
    assert "--dynamic-batch" in _plain(result.output)
    assert "calibration-da" in _plain(result.output)
    assert "--keep-onnx" in _plain(result.output)
    assert "--normalize" in _plain(result.output)
    assert "--softmax" in _plain(result.output)
    assert "--mean" in _plain(result.output)
    assert "--std" in _plain(result.output)


def test_export_torch_export_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "torch-export", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--dynamic-batch" in _plain(result.output)
    assert "--strict" in _plain(result.output)
    assert "--normalize" in _plain(result.output)
    assert "--softmax" in _plain(result.output)
    assert "--mean" in _plain(result.output)
    assert "--std" in _plain(result.output)


def test_export_torchscript_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "torchscript", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--method" in _plain(result.output)
    assert "--verify" in _plain(result.output)


def test_export_ncnn_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "ncnn", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--fp16" in _plain(result.output)
    assert "--verify" in _plain(result.output)
    assert "--device" in _plain(result.output)
    assert "--normalize" in _plain(result.output)
    assert "--softmax" in _plain(result.output)
    assert "--mean" in _plain(result.output)
    assert "--std" in _plain(result.output)


def test_export_defaults_output_to_model_name(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(
        app, ["export", "torchscript", "resnet18", "--input-size", "3", "32", "32"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "resnet18.pt").exists()
    assert f"saved: {tmp_path / 'resnet18.pt'}" in _plain(result.output)
