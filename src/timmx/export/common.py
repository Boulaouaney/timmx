from __future__ import annotations

from pathlib import Path

import timm
import torch
from timm.data import resolve_data_config

from timmx.errors import ConfigurationError, ExportError

DEFAULT_INPUT_SIZE = (3, 224, 224)


def create_timm_model(
    model_name: str,
    *,
    pretrained: bool,
    checkpoint: Path | None,
    num_classes: int | None,
    in_chans: int | None,
    exportable: bool,
) -> torch.nn.Module:
    create_kwargs: dict[str, object] = {
        "pretrained": pretrained,
        "exportable": exportable,
    }

    if checkpoint is not None:
        resolved_checkpoint = Path(checkpoint).expanduser()
        if not resolved_checkpoint.is_file():
            raise ConfigurationError(f"Checkpoint file does not exist: {resolved_checkpoint}")
        create_kwargs["checkpoint_path"] = str(resolved_checkpoint)

    if num_classes is not None:
        create_kwargs["num_classes"] = num_classes

    if in_chans is not None:
        create_kwargs["in_chans"] = in_chans

    try:
        return timm.create_model(model_name, **create_kwargs)
    except Exception as exc:
        raise ExportError(f"Failed to create timm model {model_name!r}: {exc}") from exc


def resolve_input_size(
    model: torch.nn.Module,
    requested: list[int] | None,
) -> tuple[int, int, int]:
    if requested is not None:
        return int(requested[0]), int(requested[1]), int(requested[2])

    config = resolve_data_config({}, model=model)
    raw_input_size = config.get("input_size")
    if isinstance(raw_input_size, tuple) and len(raw_input_size) == 3:
        return int(raw_input_size[0]), int(raw_input_size[1]), int(raw_input_size[2])
    if isinstance(raw_input_size, list) and len(raw_input_size) == 3:
        return int(raw_input_size[0]), int(raw_input_size[1]), int(raw_input_size[2])

    return DEFAULT_INPUT_SIZE


def validate_common_args(*, batch_size: int, device: str) -> None:
    if batch_size < 1:
        raise ConfigurationError("--batch-size must be >= 1.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("--device cuda was requested but CUDA is unavailable.")
