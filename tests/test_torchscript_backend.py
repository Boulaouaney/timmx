from pathlib import Path

import pytest
import torch

from timmx.export.common import create_timm_model, wrap_with_preprocessing
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


def test_export_torchscript_grayscale_normalize_uses_single_channel_stats(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_grayscale.pt"
    kwargs = _build_kwargs(
        output_path,
        in_chans=1,
        input_size=(1, 32, 32),
        normalize=True,
    )

    TorchScriptBackend().create_command()(**kwargs)

    loaded = torch.jit.load(str(output_path))
    out = loaded(torch.randn(1, 1, 32, 32))

    assert out.shape == (1, 1000)
    assert tuple(loaded.mean.shape) == (1, 1, 1, 1)
    assert tuple(loaded.std.shape) == (1, 1, 1, 1)


@pytest.mark.parametrize("method", ["trace", "script"])
def test_export_torchscript_wrapper_round_trips_outputs_and_config(
    tmp_path: Path, method: str
) -> None:
    seed = 789
    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    x = torch.rand(1, 3, 32, 32)

    torch.manual_seed(seed)
    reference_model = create_timm_model(
        "resnet18",
        pretrained=False,
        checkpoint=None,
        num_classes=None,
        in_chans=None,
    ).eval()
    wrapped = wrap_with_preprocessing(reference_model, softmax=True, mean=mean, std=std).eval()

    output_path = tmp_path / f"resnet18_{method}_wrapped.pt"
    kwargs = _build_kwargs(
        output_path,
        method=method,
        softmax=True,
        mean=mean,
        std=std,
    )

    torch.manual_seed(seed)
    TorchScriptBackend().create_command()(**kwargs)

    loaded = torch.jit.load(str(output_path))
    out = loaded(x)
    expected = wrapped(x)

    assert loaded.training is False
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)
    assert torch.allclose(out.sum(dim=-1), torch.ones(1), atol=1e-5)
    assert torch.allclose(loaded.mean.detach().flatten(), torch.tensor(mean), atol=1e-6)
    assert torch.allclose(loaded.std.detach().flatten(), torch.tensor(std), atol=1e-6)
