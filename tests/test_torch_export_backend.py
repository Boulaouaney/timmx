import argparse
from pathlib import Path

import pytest
import torch

from timmx.errors import ConfigurationError
from timmx.export.torch_export_backend import TorchExportBackend


def _build_args(
    output_path: Path, dynamic_batch: bool = False, batch_size: int = 1
) -> argparse.Namespace:
    return argparse.Namespace(
        model_name="resnet18",
        output=output_path,
        checkpoint=None,
        pretrained=False,
        num_classes=None,
        in_chans=None,
        batch_size=batch_size,
        input_size=[3, 32, 32],
        dynamic_batch=dynamic_batch,
        device="cpu",
        strict=False,
        verify=True,
        exportable=True,
    )


def test_export_torch_export_archive_and_load(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.pt2"
    args = _build_args(output_path)

    backend = TorchExportBackend()
    exit_code = backend.run(args)

    assert exit_code == 0
    assert output_path.exists()
    loaded_program = torch.export.load(str(output_path))
    assert type(loaded_program).__name__ == "ExportedProgram"


def test_dynamic_batch_adds_range_constraints(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_dynamic.pt2"
    args = _build_args(output_path, dynamic_batch=True, batch_size=2)

    backend = TorchExportBackend()
    exit_code = backend.run(args)

    assert exit_code == 0
    loaded_program = torch.export.load(str(output_path))
    assert len(loaded_program.range_constraints) >= 1


def test_dynamic_batch_requires_batch_size_ge_2(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_dynamic_invalid.pt2"
    args = _build_args(output_path, dynamic_batch=True, batch_size=1)

    backend = TorchExportBackend()
    with pytest.raises(ConfigurationError):
        backend.run(args)
