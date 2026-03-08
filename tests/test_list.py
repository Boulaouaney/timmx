import re
from unittest.mock import patch

from typer.testing import CliRunner

from timmx.cli import app

runner = CliRunner()

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


def test_list_resnet_returns_results() -> None:
    result = runner.invoke(app, ["list", "resnet"])
    output = _plain(result.output)
    assert result.exit_code == 0
    assert "resnet" in output.lower()
    assert "Found" in output
    # Should find at least some resnet models
    match = re.search(r"Found (\d+) models?", output)
    assert match is not None
    assert int(match.group(1)) > 0


def test_list_pretrained_only() -> None:
    result = runner.invoke(app, ["list", "--pretrained-only", "resnet18"])
    output = _plain(result.output)
    assert result.exit_code == 0
    assert "resnet18" in output.lower()
    assert "Found" in output


def test_list_nonexistent_model() -> None:
    result = runner.invoke(app, ["list", "nonexistent_xyz_model_12345"])
    output = _plain(result.output)
    assert result.exit_code == 0
    assert "Found 0 models" in output


def test_list_no_args() -> None:
    result = runner.invoke(app, ["list"])
    output = _plain(result.output)
    assert result.exit_code == 0
    match = re.search(r"Found (\d+) models?", output)
    assert match is not None
    assert int(match.group(1)) > 0


def test_list_singular_form() -> None:
    with patch("timm.list_models", return_value=["resnet18"]):
        result = runner.invoke(app, ["list", "resnet18"])
        output = _plain(result.output)
        assert result.exit_code == 0
        assert "Found 1 model" in output
        assert "Found 1 models" not in output
