from pathlib import Path

import pytest
import torch

from timmx.errors import ConfigurationError
from timmx.export.calibration import resolve_calibration_batches


def test_random_calibration_defaults_to_single_batch() -> None:
    batches = resolve_calibration_batches(
        calibration_data=None,
        calibration_steps=None,
        batch_size=2,
        input_size=(3, 16, 16),
        device=torch.device("cpu"),
    )

    assert len(batches) == 1
    assert tuple(batches[0].shape) == (2, 3, 16, 16)


def test_file_calibration_respects_requested_steps(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.pt"
    calibration_tensor = torch.randn(12, 3, 16, 16)
    torch.save(calibration_tensor, calibration_path)

    batches = resolve_calibration_batches(
        calibration_data=calibration_path,
        calibration_steps=3,
        batch_size=2,
        input_size=(3, 16, 16),
        device=torch.device("cpu"),
    )

    assert len(batches) == 3
    assert torch.equal(batches[0], calibration_tensor[0:2].to(dtype=torch.float32))
    assert torch.equal(batches[2], calibration_tensor[4:6].to(dtype=torch.float32))


def test_file_calibration_rejects_input_shape_mismatch(tmp_path: Path) -> None:
    calibration_path = tmp_path / "bad_calibration.pt"
    torch.save(torch.randn(4, 3, 8, 8), calibration_path)

    with pytest.raises(ConfigurationError):
        resolve_calibration_batches(
            calibration_data=calibration_path,
            calibration_steps=None,
            batch_size=2,
            input_size=(3, 16, 16),
            device=torch.device("cpu"),
        )
