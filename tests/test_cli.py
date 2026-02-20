from typer.testing import CliRunner

from timmx.cli import app

runner = CliRunner()


def test_root_help_mentions_export() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "export" in result.output


def test_version_flag() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "timmx" in result.output


def test_export_help_lists_backends() -> None:
    result = runner.invoke(app, ["export", "--help"])
    assert result.exit_code == 0
    assert "onnx" in result.output
    assert "coreml" in result.output
    assert "litert" in result.output
    assert "tensorrt" in result.output
    assert "torch-export" in result.output
    assert "torchscript" in result.output


def test_export_onnx_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "onnx", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--opset" in result.output
    assert "--dynamic-batch" in result.output
    assert "--check" in result.output


def test_export_coreml_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "coreml", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--convert-to" in result.output
    assert "compute-preci" in result.output
    assert "--dynamic-batch" in result.output


def test_export_litert_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "litert", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--mode" in result.output
    assert "calibration-da" in result.output
    assert "--nhwc-input" in result.output


def test_export_tensorrt_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "tensorrt", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--mode" in result.output
    assert "--workspace" in result.output
    assert "--dynamic-batch" in result.output
    assert "calibration-da" in result.output
    assert "--keep-onnx" in result.output


def test_export_torch_export_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "torch-export", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--dynamic-batch" in result.output
    assert "--strict" in result.output


def test_export_torchscript_help_shows_options() -> None:
    result = runner.invoke(app, ["export", "torchscript", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--method" in result.output
    assert "--verify" in result.output
