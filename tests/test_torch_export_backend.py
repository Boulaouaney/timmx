from pathlib import Path

import pytest
import torch

from timmx.errors import ConfigurationError
from timmx.export.common import create_timm_model, wrap_with_preprocessing
from timmx.export.torch_export_backend import TorchExportBackend


def _build_kwargs(
    output_path: Path,
    dynamic_batch: bool = False,
    batch_size: int = 1,
    **overrides: object,
) -> dict:
    defaults: dict[str, object] = {
        "model_name": "resnet18",
        "output": output_path,
        "checkpoint": None,
        "pretrained": False,
        "num_classes": None,
        "in_chans": None,
        "batch_size": batch_size,
        "input_size": (3, 32, 32),
        "dynamic_batch": dynamic_batch,
        "device": "cpu",
        "strict": False,
        "normalize": False,
        "softmax": False,
        "mean": None,
        "std": None,
        "verify": True,
    }
    defaults.update(overrides)
    return defaults


def test_export_torch_export_archive_and_load(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.pt2"
    kwargs = _build_kwargs(output_path)

    backend = TorchExportBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    loaded_program = torch.export.load(str(output_path))
    assert type(loaded_program).__name__ == "ExportedProgram"


def test_dynamic_batch_adds_range_constraints(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_dynamic.pt2"
    kwargs = _build_kwargs(output_path, dynamic_batch=True, batch_size=2)

    backend = TorchExportBackend()
    command = backend.create_command()
    command(**kwargs)

    loaded_program = torch.export.load(str(output_path))
    assert len(loaded_program.range_constraints) >= 1


def test_dynamic_batch_requires_batch_size_ge_2(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_dynamic_invalid.pt2"
    kwargs = _build_kwargs(output_path, dynamic_batch=True, batch_size=1)

    backend = TorchExportBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError):
        command(**kwargs)


def test_export_torch_export_softmax_and_custom_stats_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_softmax.pt2"
    checkpoint_path = tmp_path / "resnet18.pth"
    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)

    reference_model = create_timm_model(
        "resnet18",
        pretrained=False,
        checkpoint=None,
        num_classes=None,
        in_chans=None,
    )
    torch.save(reference_model.state_dict(), checkpoint_path)

    kwargs = _build_kwargs(
        output_path,
        checkpoint=checkpoint_path,
        normalize=True,
        softmax=True,
        mean=mean,
        std=std,
        verify=False,
    )

    TorchExportBackend().create_command()(**kwargs)

    loaded_program = torch.export.load(str(output_path))
    loaded_module = loaded_program.module()
    wrapped_reference = wrap_with_preprocessing(
        reference_model,
        normalize=True,
        softmax=True,
        mean=mean,
        std=std,
    ).eval()
    example_input = torch.rand(1, 3, 32, 32)

    expected = wrapped_reference(example_input)
    actual = loaded_module(example_input)

    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    assert any("softmax" in str(node.target) for node in loaded_program.graph.nodes)


def test_torch_export_rejects_mean_std_without_wrapper_flags(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_invalid.pt2"
    kwargs = _build_kwargs(
        output_path,
        mean=(0.5, 0.25, 0.75),
        std=(0.125, 0.5, 0.25),
    )

    with pytest.raises(ConfigurationError, match="--mean/--std require --normalize"):
        TorchExportBackend().create_command()(**kwargs)
