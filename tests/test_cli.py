import re

from typer.testing import CliRunner

from timmx.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Strip ANSI escape codes so assertions work in CI (Rich emits them on GitHub Actions)."""
    return _ANSI_RE.sub("", text)


def test_root_help_mentions_export() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "export" in _plain(result.output)


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "timmx" in _plain(result.output)


def test_root_help_mentions_doctor() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in _plain(result.output)


def test_export_help_lists_backends() -> None:
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "coreml" in _plain(result.output)
    assert "executorch" in _plain(result.output)
    assert "litert" in _plain(result.output)
    assert "ncnn" in _plain(result.output)
    assert "onnx" in _plain(result.output)
    assert "tensorrt" in _plain(result.output)
    assert "torch-export" in _plain(result.output)
    assert "torchscript" in _plain(result.output)


def test_export_executorch_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "executorch", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--delegate" in _plain(result.output)
    assert "--mode" in _plain(result.output)
    assert "compute-preci" in _plain(result.output)
    assert "--dynamic-batch" in _plain(result.output)
    assert "--calibration-d" in _plain(result.output)
    assert "--per-channel" in _plain(result.output)


def test_export_onnx_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "onnx", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--opset" in _plain(result.output)
    assert "--dynamic-batch" in _plain(result.output)
    assert "--check" in _plain(result.output)


def test_export_coreml_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "coreml", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--convert-to" in _plain(result.output)
    assert "compute-preci" in _plain(result.output)
    assert "--dynamic-batch" in _plain(result.output)
    assert "--source" in _plain(result.output)


def test_export_litert_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "litert", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--mode" in _plain(result.output)
    assert "calibration-da" in _plain(result.output)
    assert "--nhwc-input" in _plain(result.output)


def test_export_tensorrt_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "tensorrt", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--mode" in _plain(result.output)
    assert "--workspace" in _plain(result.output)
    assert "--dynamic-batch" in _plain(result.output)
    assert "calibration-da" in _plain(result.output)
    assert "--keep-onnx" in _plain(result.output)


def test_export_torch_export_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "torch-export", "--help"])
    assert result.exit_code == 0
    assert "--output" in _plain(result.output)
    assert "--dynamic-batch" in _plain(result.output)
    assert "--strict" in _plain(result.output)


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
    assert "--device" in _plain(result.output)
