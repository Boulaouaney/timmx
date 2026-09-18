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
    path = timmx.export_model("torchscript", "resnet18", input_size=(3, 32, 32))
    assert path == tmp_path / "resnet18.pt"
    assert isinstance(torch.jit.load(str(path)), torch.jit.ScriptModule)


def test_export_honours_output_and_string_choices(tmp_path: Path) -> None:
    path = timmx.export_model(
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
        timmx.export_model("tflite", "resnet18")


def test_export_rejects_unknown_option() -> None:
    with pytest.raises(
        ConfigurationError, match="Unknown option\\(s\\) not_a_flag; valid options: "
    ):
        timmx.export_model("torchscript", "resnet18", not_a_flag=True)


def test_export_rejects_model_name_as_option() -> None:
    with pytest.raises(ConfigurationError, match=r"Unknown option\(s\) model_name"):
        timmx.export_model("torchscript", "resnet18", model_name="resnet50")


def test_export_rejects_unknown_choice() -> None:
    with pytest.raises(
        ConfigurationError, match="Unknown value 'bogus' for method; choices: trace, script"
    ):
        timmx.export_model("torchscript", "resnet18", method="bogus")


def test_export_accepts_string_paths_and_choices(tmp_path: Path) -> None:
    ov = pytest.importorskip("openvino")
    # openvino checks output.suffix before prepare_export(), so a str output must already be a Path
    path = timmx.export_model(
        "openvino",
        "resnet18",
        output=str(tmp_path / "r18.xml"),
        input_size=(3, 32, 32),
        device="cpu",
        verify=False,
    )
    assert path == tmp_path / "r18.xml"
    assert ov.Core().read_model(str(path)).inputs[0].any_name == "input"


def test_import_timmx_does_not_load_the_backends() -> None:
    import subprocess
    import sys

    loaded = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, timmx; print('timmx.api' in sys.modules, 'torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert loaded == ["False", "False"]
