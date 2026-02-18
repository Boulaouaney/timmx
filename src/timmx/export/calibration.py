from __future__ import annotations

import argparse
from pathlib import Path

import torch

from timmx.errors import ConfigurationError, ExportError


def add_calibration_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--calibration-data",
        type=Path,
        help=(
            "Path to a torch-saved calibration tensor with shape (N, C, H, W). "
            "Used by quantized export modes."
        ),
    )
    parser.add_argument(
        "--calibration-steps",
        type=int,
        help=(
            "Number of calibration batches to consume. "
            "Default is 1 random batch when --calibration-data is not set, "
            "or all full batches from --calibration-data when set."
        ),
    )


def resolve_calibration_batches(
    *,
    calibration_data: Path | None,
    calibration_steps: int | None,
    batch_size: int,
    input_size: tuple[int, int, int],
    device: torch.device,
) -> list[torch.Tensor]:
    if calibration_steps is not None and calibration_steps < 1:
        raise ConfigurationError("--calibration-steps must be >= 1.")

    if calibration_data is None:
        batch_count = calibration_steps or 1
        return [torch.randn(batch_size, *input_size, device=device) for _ in range(batch_count)]

    data_tensor = _load_calibration_tensor(calibration_data)
    data_tensor = _normalize_calibration_tensor(
        data_tensor=data_tensor,
        input_size=input_size,
        source_path=calibration_data,
    )

    full_batches = data_tensor.shape[0] // batch_size
    if full_batches < 1:
        raise ConfigurationError(
            f"Calibration data {calibration_data} has {data_tensor.shape[0]} samples but "
            f"needs at least --batch-size {batch_size}."
        )

    requested_batches = full_batches if calibration_steps is None else calibration_steps
    if requested_batches > full_batches:
        raise ConfigurationError(
            f"Calibration data {calibration_data} has only {full_batches} full batches for "
            f"--batch-size {batch_size}, but --calibration-steps is {requested_batches}."
        )

    batches: list[torch.Tensor] = []
    for batch_index in range(requested_batches):
        start = batch_index * batch_size
        stop = start + batch_size
        batch = data_tensor[start:stop].to(device=device, dtype=torch.float32)
        batches.append(batch)

    return batches


def _load_calibration_tensor(calibration_data: Path) -> torch.Tensor:
    resolved_path = calibration_data.expanduser().resolve()
    if not resolved_path.is_file():
        raise ConfigurationError(f"Calibration data file does not exist: {resolved_path}")

    try:
        payload = torch.load(resolved_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ExportError(f"Failed to load calibration data from {resolved_path}: {exc}") from exc

    if not torch.is_tensor(payload):
        raise ConfigurationError(
            f"Calibration data at {resolved_path} must be a torch.Tensor with shape (N, C, H, W)."
        )

    return payload


def _normalize_calibration_tensor(
    *,
    data_tensor: torch.Tensor,
    input_size: tuple[int, int, int],
    source_path: Path,
) -> torch.Tensor:
    if data_tensor.ndim == 3:
        data_tensor = data_tensor.unsqueeze(0)

    if data_tensor.ndim != 4:
        raise ConfigurationError(
            f"Calibration data {source_path} must have rank 4 (N, C, H, W), got rank "
            f"{data_tensor.ndim}."
        )

    channels, height, width = (
        int(data_tensor.shape[1]),
        int(data_tensor.shape[2]),
        int(data_tensor.shape[3]),
    )

    if (channels, height, width) != input_size:
        raise ConfigurationError(
            f"Calibration data {source_path} has per-sample shape "
            f"({channels}, {height}, {width}) but expected {input_size}."
        )

    if not data_tensor.is_floating_point():
        data_tensor = data_tensor.to(dtype=torch.float32)
    else:
        data_tensor = data_tensor.detach().to(dtype=torch.float32)

    return data_tensor.contiguous()
