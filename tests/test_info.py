import re

from typer.testing import CliRunner

from timmx.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_info_resnet18() -> None:
    result = runner.invoke(app, ["info", "resnet18"])
    assert result.exit_code == 0
    output = _plain(result.output)
    assert "resnet18" in output
    assert "ResNet" in output
    assert "Parameters" in output
    assert "Input size" in output
    assert "3 x 224 x 224" in output
    assert "Classes" in output
    assert "1000" in output
    assert "none (random init)" in output


def test_info_pretrained_flag() -> None:
    """Verify --pretrained is reflected in the weights row (no actual download)."""
    result = runner.invoke(app, ["info", "resnet18", "--pretrained"])
    assert result.exit_code == 0
    output = _plain(result.output)
    assert "pretrained" in output


def test_info_num_classes_override() -> None:
    result = runner.invoke(app, ["info", "resnet18", "--num-classes", "10"])
    assert result.exit_code == 0
    output = _plain(result.output)
    assert "10" in output


def test_info_invalid_model() -> None:
    result = runner.invoke(app, ["info", "not_a_real_model_xyz"])
    assert result.exit_code == 2
    output = _plain(result.output)
    assert "error" in output.lower()


def test_info_help() -> None:
    result = runner.invoke(app, ["info", "--help"])
    assert result.exit_code == 0
    output = _plain(result.output)
    assert "--pretrained" in output
    assert "--checkpoint" in output
    assert "--num-classes" in output
    assert "--in-chans" in output
