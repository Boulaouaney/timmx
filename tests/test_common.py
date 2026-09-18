import numpy as np
import pytest
import torch

from timmx.errors import ExportError
from timmx.export.common import verify_outputs


def test_verify_outputs_accepts_matching_output() -> None:
    expected = torch.randn(2, 10)
    verify_outputs(expected, expected.numpy(), backend="test")


def test_verify_outputs_reshapes_flat_output() -> None:
    expected = torch.randn(1, 4)
    verify_outputs(expected, expected.numpy().flatten().tolist(), backend="test")


def test_verify_outputs_rejects_size_mismatch() -> None:
    with pytest.raises(ExportError, match="has 3 values, PyTorch has 4"):
        verify_outputs(torch.randn(1, 4), np.zeros(3, dtype=np.float32), backend="test")


def test_verify_outputs_rejects_divergent_output() -> None:
    expected = torch.randn(2, 10)
    with pytest.raises(ExportError, match="diverges from PyTorch"):
        verify_outputs(expected, (-expected).numpy(), backend="test")
