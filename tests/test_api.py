from pathlib import Path

import pytest
import torch

import timmx
from timmx.errors import ConfigurationError
from timmx.export import create_builtin_registry


def test_backends_lists_registry_names() -> None:
    assert timmx.backends() == create_builtin_registry().names()


def test_export_returns_written_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = timmx.export("torchscript", "resnet18", input_size=(3, 32, 32))
    assert path == tmp_path / "resnet18.pt"
    assert isinstance(torch.jit.load(str(path)), torch.jit.ScriptModule)


def test_export_honours_output_and_string_choices(tmp_path: Path) -> None:
    path = timmx.export(
        "torchscript",
        "resnet18",
        output=tmp_path / "scripted.pt",
        method="script",
        input_size=(3, 32, 32),
        verify=False,
    )
    assert path == tmp_path / "scripted.pt"
    assert path.exists()


def test_export_rejects_unknown_backend() -> None:
    with pytest.raises(ConfigurationError, match="Unknown backend 'tflite'; available: coreai"):
        timmx.export("tflite", "resnet18")


def test_export_rejects_unknown_option() -> None:
    with pytest.raises(TypeError, match="not_a_flag"):
        timmx.export("torchscript", "resnet18", not_a_flag=True)
