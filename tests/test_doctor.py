from typer.testing import CliRunner

from timmx.cli import app

runner = CliRunner()

BACKENDS = ["coreml", "executorch", "litert", "onnx", "tensorrt", "torch-export", "torchscript"]


def test_doctor_runs_successfully() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "timmx" in result.output
    assert "Python" in result.output
    assert "torch" in result.output


def test_doctor_lists_all_backends() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    for backend in BACKENDS:
        assert backend in result.output


def test_doctor_shows_availability() -> None:
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    for backend in BACKENDS:
        lines = [line for line in result.output.splitlines() if backend in line]
        assert lines, f"backend {backend} not found in output"
        combined = " ".join(lines)
        assert "available" in combined or "missing" in combined
