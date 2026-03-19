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

# ---------------------------------------------------------------------------
# Typer type aliases for --normalize / --softmax CLI flags
# ---------------------------------------------------------------------------

NormalizeOpt = Annotated[
    bool,
    typer.Option(
        "--normalize",
        help="Wrap model with input normalization (mean/std from timm config).",
    ),
]
SoftmaxOpt = Annotated[
    bool,
    typer.Option(
        "--softmax",
        help="Add softmax output layer.",
    ),
]
MeanOpt = Annotated[
    tuple[float, float, float] | None,
    typer.Option(help="Custom normalization mean (3 RGB values; averaged for 1-channel models)."),
]
StdOpt = Annotated[
    tuple[float, float, float] | None,
    typer.Option(help="Custom normalization std (3 RGB values; averaged for 1-channel models)."),
]

DEFAULT_INPUT_SIZE = (3, 224, 224)
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)
SUPPORTED_INPUT_CHANNELS = frozenset({1, 3})

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
InChansOpt = Annotated[
    int | None,
    typer.Option(help="Override model input channels (currently 1 or 3 only)."),
]
BatchSizeOpt = Annotated[int, typer.Option(help="Example input batch size for export.")]
InputSizeOpt = Annotated[
    tuple[int, int, int] | None, typer.Option(help="Explicit input shape as C H W.")
]
DeviceOpt = Annotated[Device, typer.Option(help="Device used for model instantiation and export.")]


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
    normalize: bool = False,
    softmax: bool = False,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> PreparedExport:
    """Validate common args, create the timm model, and build an example input.

    Set *output_is_dir=True* when the backend writes to a directory rather than a
    single file (e.g. ncnn).  The resolved path is then created as a directory;
    otherwise its parent directory is created.
    """
    validate_common_args(
        batch_size=batch_size,
        device=device,
        in_chans=in_chans,
        input_size=input_size,
    )

    if (mean is not None or std is not None) and not normalize:
        raise ConfigurationError("--mean/--std require --normalize.")

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
    model_input_channels = resolve_model_input_channels(model)
    validate_supported_input_channels(model_input_channels, source="model")
    if resolved_input_size[0] != model_input_channels:
        raise ConfigurationError(
            "--input-size channel count must match the model input channels: "
            f"expected {model_input_channels}, got {resolved_input_size[0]}."
        )

    torch_device = torch.device(device)
    model = model.to(torch_device)
    model.eval()

    if softmax or normalize:
        model = wrap_with_preprocessing(
            model,
            normalize=normalize,
            softmax=softmax,
            mean=mean,
            std=std,
        )
        model = model.to(torch_device)

    example_input = torch.randn(batch_size, *resolved_input_size, device=torch_device)

    return PreparedExport(
        model=model,
        example_input=example_input,
        resolved_input_size=resolved_input_size,
        output_path=output_path,
        torch_device=torch_device,
    )


# ---------------------------------------------------------------------------
# Preprocessing / postprocessing wrapper
# ---------------------------------------------------------------------------


class PrePostWrapper(torch.nn.Module):
    """Wraps a model with optional input normalization and optional softmax output."""

    def __init__(
        self,
        model: torch.nn.Module,
        mean: tuple[float, ...] | None = None,
        std: tuple[float, ...] | None = None,
        normalize: bool = True,
        softmax: bool = False,
    ):
        super().__init__()
        self.model = model
        self.normalize = normalize
        if normalize:
            if mean is None or std is None:
                raise ConfigurationError("PrePostWrapper normalization requires mean/std values.")
            self.register_buffer("mean", torch.tensor(mean).reshape(1, -1, 1, 1))
            self.register_buffer("std", torch.tensor(std).reshape(1, -1, 1, 1))
        else:
            # Use identity-normalization tensors so TorchScript sees concrete Tensor
            # types instead of Optional[Tensor] (which fails static type-checking of
            # the normalize branch even when self.normalize is False).
            self.register_buffer("mean", torch.zeros(1))
            self.register_buffer("std", torch.ones(1))
        self.softmax = softmax
        self.train(model.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.model(x)
        if self.softmax:
            x = torch.nn.functional.softmax(x, dim=-1)
        return x


def wrap_with_preprocessing(
    model: torch.nn.Module,
    normalize: bool = True,
    softmax: bool = False,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> PrePostWrapper:
    """Wrap model with optional normalization and optional softmax.

    Uses timm's data config for normalization by default; pass *mean*/*std* to override.
    """
    effective_mean: tuple[float, ...] | None
    effective_std: tuple[float, ...] | None
    if normalize:
        effective_mean, effective_std = resolve_normalization_stats(model, mean=mean, std=std)
    else:
        effective_mean, effective_std = None, None
    return PrePostWrapper(
        model,
        mean=effective_mean,
        std=effective_std,
        normalize=normalize,
        softmax=softmax,
    )


def resolve_model_input_channels(model: torch.nn.Module) -> int:
    raw_channels = getattr(model, "in_chans", None)
    if isinstance(raw_channels, int):
        return raw_channels

    config = resolve_data_config(model=model)
    raw_input_size = config.get("input_size")
    if isinstance(raw_input_size, tuple) and len(raw_input_size) == 3:
        return int(raw_input_size[0])
    if isinstance(raw_input_size, list) and len(raw_input_size) == 3:
        return int(raw_input_size[0])

    return DEFAULT_INPUT_SIZE[0]


def resolve_normalization_stats(
    model: torch.nn.Module,
    *,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    config = resolve_data_config(model=model)
    input_channels = resolve_model_input_channels(model)
    return resolve_normalization_stats_for_channels(
        input_channels=input_channels,
        mean=mean,
        std=std,
        default_mean=config.get("mean", DEFAULT_MEAN),
        default_std=config.get("std", DEFAULT_STD),
    )


def resolve_normalization_stats_for_channels(
    *,
    input_channels: int,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    default_mean: tuple[float, ...] | list[float] | None = DEFAULT_MEAN,
    default_std: tuple[float, ...] | list[float] | None = DEFAULT_STD,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    validate_supported_input_channels(input_channels, source="input")

    raw_mean = mean if mean is not None else (default_mean or DEFAULT_MEAN)
    raw_std = std if std is not None else (default_std or DEFAULT_STD)
    return (
        _coerce_stats_for_channels(raw_mean, input_channels, stat_name="mean"),
        _coerce_stats_for_channels(raw_std, input_channels, stat_name="std"),
    )


def validate_supported_input_channels(channels: int, *, source: str) -> None:
    if channels not in SUPPORTED_INPUT_CHANNELS:
        raise ConfigurationError(f"{source} input channels must be 1 or 3 for now, got {channels}.")


def _coerce_stats_for_channels(
    stats: tuple[float, ...] | list[float],
    input_channels: int,
    *,
    stat_name: str,
) -> tuple[float, ...]:
    values = tuple(float(v) for v in stats)
    if len(values) == input_channels:
        return values
    if len(values) == 3 and input_channels == 1:
        return (sum(values) / len(values),)
    if len(values) == 1 and input_channels == 3:
        return values * 3
    raise ConfigurationError(
        f"Normalization {stat_name} has {len(values)} values but only "
        "1-value and 3-value stats are supported."
    )


def create_timm_model(
    model_name: str,
    *,
    pretrained: bool,
    checkpoint: Path | None,
    num_classes: int | None,
    in_chans: int | None,
) -> torch.nn.Module:
    if in_chans is not None:
        validate_supported_input_channels(in_chans, source="--in-chans")

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
        validate_supported_input_channels(requested[0], source="--input-size")
        return requested

    input_channels = resolve_model_input_channels(model)
    validate_supported_input_channels(input_channels, source="model")

    config = resolve_data_config(model=model)
    raw_input_size = config.get("input_size")
    if isinstance(raw_input_size, tuple) and len(raw_input_size) == 3:
        return input_channels, int(raw_input_size[1]), int(raw_input_size[2])
    if isinstance(raw_input_size, list) and len(raw_input_size) == 3:
        return input_channels, int(raw_input_size[1]), int(raw_input_size[2])

    return input_channels, DEFAULT_INPUT_SIZE[1], DEFAULT_INPUT_SIZE[2]


def validate_common_args(
    *,
    batch_size: int,
    device: str,
    in_chans: int | None = None,
    input_size: tuple[int, int, int] | None = None,
) -> None:
    if batch_size < 1:
        raise ConfigurationError("--batch-size must be >= 1.")
    if device == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("--device cuda was requested but CUDA is unavailable.")
    if in_chans is not None:
        validate_supported_input_channels(in_chans, source="--in-chans")
    if input_size is not None:
        validate_supported_input_channels(input_size[0], source="--input-size")
