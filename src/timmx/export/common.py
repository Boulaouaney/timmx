from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import timm
import torch
import typer
from timm.data import resolve_data_config
from timm.utils import reparameterize_model

from timmx.errors import ConfigurationError, ExportError
from timmx.export.types import Device

DEFAULT_INPUT_SIZE = (3, 224, 224)

# ---------------------------------------------------------------------------
# Shared Typer type aliases for common CLI parameters
# ---------------------------------------------------------------------------

ModelNameArg = Annotated[str, typer.Argument(help="timm model name, e.g. resnet18")]
OutputOpt = Annotated[Path, typer.Option(help="Path to write the exported model.")]
CheckpointOpt = Annotated[Path | None, typer.Option(help="Path to a fine-tuned checkpoint.")]
PretrainedOpt = Annotated[bool, typer.Option("--pretrained", help="Load timm pretrained weights.")]
NumClassesOpt = Annotated[
    int | None, typer.Option(help="Override the model classifier output classes.")
]
InChansOpt = Annotated[int | None, typer.Option(help="Override model input channels.")]
BatchSizeOpt = Annotated[int, typer.Option(help="Example input batch size for export.")]
InputSizeOpt = Annotated[
    tuple[int, int, int] | None, typer.Option(help="Explicit input shape as C H W.")
]
DeviceOpt = Annotated[Device, typer.Option(help="Device used for model instantiation and tracing.")]


# ---------------------------------------------------------------------------
# PreparedExport: result of the common model-creation / input-preparation step
# ---------------------------------------------------------------------------


@dataclass
class PreparedExport:
    """Bundle returned by :func:`prepare_export` with everything backends need."""

    model: torch.nn.Module
    example_input: torch.Tensor
    resolved_input_size: tuple[int, int, int]
    output_path: Path
    torch_device: torch.device


def prepare_export(
    *,
    model_name: str,
    output: Path,
    checkpoint: Path | None,
    pretrained: bool,
    num_classes: int | None,
    in_chans: int | None,
    batch_size: int,
    input_size: tuple[int, int, int] | None,
    device: Device | str,
    output_is_dir: bool = False,
) -> PreparedExport:
    """Validate common args, create the timm model, and build an example input.

    Set *output_is_dir=True* when the backend writes to a directory rather than a
    single file (e.g. ncnn).  The resolved path is then created as a directory;
    otherwise its parent directory is created.
    """
    validate_common_args(batch_size=batch_size, device=device)

    output_path = Path(output).expanduser().resolve()
    if output_is_dir:
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    model = create_timm_model(
        model_name,
        pretrained=pretrained,
        checkpoint=checkpoint,
        num_classes=num_classes,
        in_chans=in_chans,
    )
    resolved_input_size = resolve_input_size(model, input_size)

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    example_input = torch.randn(batch_size, *resolved_input_size, device=torch_device)

    return PreparedExport(
        model=model,
        example_input=example_input,
        resolved_input_size=resolved_input_size,
        output_path=output_path,
        torch_device=torch_device,
    )


# TODO: add preprocessing (norm) & postprocessing (softmax) wrapper
def create_timm_model(
    model_name: str,
    *,
    pretrained: bool,
    checkpoint: Path | None,
    num_classes: int | None,
    in_chans: int | None,
) -> torch.nn.Module:
    create_kwargs: dict[str, object] = {
        "pretrained": pretrained,
        "exportable": True,
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
        model = timm.create_model(model_name, **create_kwargs)
    except Exception as exc:
        raise ExportError(f"Failed to create timm model {model_name!r}: {exc}") from exc

    reparameterize_model(model, inplace=True)
    return model


def resolve_input_size(
    model: torch.nn.Module,
    requested: tuple[int, int, int] | None,
) -> tuple[int, int, int]:
    if requested is not None:
        return requested

    config = resolve_data_config(model=model)
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
