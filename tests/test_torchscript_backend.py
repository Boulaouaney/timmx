from pathlib import Path

import torch

from timmx.export.torchscript_backend import TorchScriptBackend


def _build_kwargs(output_path: Path, **overrides: object) -> dict:
    defaults: dict[str, object] = {
        "model_name": "resnet18",
        "output": output_path,
        "checkpoint": None,
        "pretrained": False,
        "num_classes": None,
        "in_chans": None,
        "batch_size": 1,
        "input_size": (3, 32, 32),
        "method": "trace",
        "device": "cpu",
        "verify": True,
    }
    defaults.update(overrides)
    return defaults


def test_export_torchscript_trace_and_load(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.pt"
    kwargs = _build_kwargs(output_path)

    backend = TorchScriptBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    loaded = torch.jit.load(str(output_path))
    assert isinstance(loaded, torch.jit.ScriptModule)


def test_export_torchscript_script_method(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_script.pt"
    kwargs = _build_kwargs(output_path, method="script")

    backend = TorchScriptBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    loaded = torch.jit.load(str(output_path))
    assert isinstance(loaded, torch.jit.ScriptModule)


def test_export_torchscript_verify_runs_forward_pass(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_verified.pt"
    kwargs = _build_kwargs(output_path, verify=True)

    backend = TorchScriptBackend()
    command = backend.create_command()
    command(**kwargs)

    loaded = torch.jit.load(str(output_path))
    out = loaded(torch.randn(1, 3, 32, 32))
    assert out.shape == (1, 1000)
