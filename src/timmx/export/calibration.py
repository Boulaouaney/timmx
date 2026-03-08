from __future__ import annotations

import random
from pathlib import Path

import torch

from timmx.console import console
from timmx.errors import ConfigurationError, ExportError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}
DEFAULT_CALIBRATION_SAMPLES = 128


def resolve_calibration_batches(
    *,
    calibration_data: Path | None,
    calibration_steps: int | None,
    batch_size: int,
    input_size: tuple[int, int, int],
    device: torch.device,
    model: torch.nn.Module | None = None,
    calibration_samples: int | None = None,
    random_calibration: bool = False,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> list[torch.Tensor]:
    if calibration_steps is not None and calibration_steps < 1:
        raise ConfigurationError("--calibration-steps must be >= 1.")

    if calibration_samples is not None and calibration_samples < 1:
        raise ConfigurationError("--calibration-samples must be >= 1.")

    if calibration_data is None:
        if not random_calibration:
            raise ConfigurationError(
                "int8 quantization requires calibration data for accurate results.\n\n"
                "  Provide real data:      --calibration-data <image-dir-or-tensor.pt>\n"
                "  Or use random noise:    --random-calibration\n"
                "  Or choose a different quantization mode (e.g. fp16, dynamic-int8).\n\n"
                "Random noise calibration produces poor quantization accuracy and "
                "is not recommended for production use."
            )
        console.print(
            "[bold yellow]Warning:[/bold yellow] Using random noise for calibration. "
            "For better quantization accuracy, provide real data via "
            "[bold]--calibration-data <image-dir-or-tensor.pt>[/bold]."
        )
        batch_count = calibration_steps or 1
        return [torch.randn(batch_size, *input_size, device=device) for _ in range(batch_count)]

    resolved_path = calibration_data.expanduser().resolve()

    if resolved_path.is_dir():
        if model is None:
            raise ConfigurationError(
                "model is required when loading calibration images from a directory."
            )
        data_tensor = _load_calibration_images(
            image_dir=resolved_path,
            model=model,
            input_size=input_size,
            max_samples=calibration_samples or DEFAULT_CALIBRATION_SAMPLES,
            mean=mean,
            std=std,
        )
    elif resolved_path.is_file():
        data_tensor = _load_calibration_tensor(resolved_path)
        data_tensor = _normalize_calibration_tensor(
            data_tensor=data_tensor,
            input_size=input_size,
            source_path=resolved_path,
        )
    else:
        raise ConfigurationError(f"Calibration data path does not exist: {resolved_path}")

    return _slice_into_batches(
        data_tensor=data_tensor,
        batch_size=batch_size,
        calibration_steps=calibration_steps,
        device=device,
        source_path=resolved_path,
    )


def _slice_into_batches(
    *,
    data_tensor: torch.Tensor,
    batch_size: int,
    calibration_steps: int | None,
    device: torch.device,
    source_path: Path,
) -> list[torch.Tensor]:
    full_batches = data_tensor.shape[0] // batch_size
    if full_batches < 1:
        raise ConfigurationError(
            f"Calibration data {source_path} has {data_tensor.shape[0]} samples but "
            f"needs at least --batch-size {batch_size}."
        )

    requested_batches = full_batches if calibration_steps is None else calibration_steps
    if requested_batches > full_batches:
        raise ConfigurationError(
            f"Calibration data {source_path} has only {full_batches} full batches for "
            f"--batch-size {batch_size}, but --calibration-steps is {requested_batches}."
        )

    batches: list[torch.Tensor] = []
    for batch_index in range(requested_batches):
        start = batch_index * batch_size
        stop = start + batch_size
        batch = data_tensor[start:stop].to(device=device, dtype=torch.float32)
        batches.append(batch)

    return batches


# ---------------------------------------------------------------------------
# Image directory loading
# ---------------------------------------------------------------------------


def _collect_image_paths(image_dir: Path) -> list[Path]:
    """Recursively collect image file paths from a directory."""
    paths = [
        p
        for p in sorted(image_dir.rglob("*"))
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not paths:
        raise ConfigurationError(
            f"No image files found in {image_dir}. "
            f"Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    return paths


def _load_calibration_images(
    *,
    image_dir: Path,
    model: torch.nn.Module,
    input_size: tuple[int, int, int],
    max_samples: int,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> torch.Tensor:
    from PIL import Image
    from timm.data import create_transform, resolve_data_config

    image_paths = _collect_image_paths(image_dir)

    if len(image_paths) > max_samples:
        rng = random.Random(42)
        image_paths = rng.sample(image_paths, max_samples)

    data_config = resolve_data_config(model=model)
    data_config["input_size"] = input_size
    if mean is not None:
        data_config["mean"] = mean
    if std is not None:
        data_config["std"] = std
    transform = create_transform(**data_config, is_training=False)

    tensors: list[torch.Tensor] = []
    skipped = 0
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            tensor = transform(img)
            tensors.append(tensor)
        except Exception as exc:
            skipped += 1
            console.print(f"[dim]  Skipped {path.name}: {exc}[/dim]")
            continue

    if not tensors:
        raise ConfigurationError(
            f"Failed to load any images from {image_dir}. "
            "Check that the directory contains valid image files."
        )

    if skipped > 0:
        console.print(f"[yellow]Skipped {skipped} unreadable image(s) from {image_dir}.[/yellow]")

    console.print(
        f"Loaded [bold]{len(tensors)}[/bold] calibration image(s) from [bold]{image_dir}[/bold]."
    )

    return torch.stack(tensors)


# ---------------------------------------------------------------------------
# Torch tensor file loading (existing path)
# ---------------------------------------------------------------------------


def _load_calibration_tensor(resolved_path: Path) -> torch.Tensor:
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
