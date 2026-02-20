from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import torch
import typer

from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import ExportBackend
from timmx.export.common import create_timm_model, resolve_input_size, validate_common_args
from timmx.export.types import Device


class ConvertTo(StrEnum):
    mlprogram = "mlprogram"
    neuralnetwork = "neuralnetwork"


class ComputePrecision(StrEnum):
    float16 = "float16"
    float32 = "float32"


class CoreMLBackend(ExportBackend):
    name = "coreml"
    help = "Export a timm model to Core ML."

    def create_command(self) -> Callable[..., None]:
        def command(
            model_name: Annotated[str, typer.Argument(help="timm model name, e.g. resnet18")],
            output: Annotated[
                Path, typer.Option(help="Path to write the Core ML model (.mlpackage or .mlmodel).")
            ],
            checkpoint: Annotated[
                Path | None, typer.Option(help="Path to a fine-tuned checkpoint.")
            ] = None,
            pretrained: Annotated[
                bool, typer.Option("--pretrained", help="Load timm pretrained weights.")
            ] = False,
            num_classes: Annotated[
                int | None, typer.Option(help="Override the model classifier output classes.")
            ] = None,
            in_chans: Annotated[
                int | None, typer.Option(help="Override model input channels.")
            ] = None,
            batch_size: Annotated[
                int, typer.Option(help="Example input batch size for export.")
            ] = 1,
            input_size: Annotated[
                tuple[int, int, int] | None,
                typer.Option(help="Explicit input shape as C H W."),
            ] = None,
            dynamic_batch: Annotated[
                bool,
                typer.Option(
                    "--dynamic-batch", help="Export with flexible Core ML batch dimension."
                ),
            ] = False,
            batch_upper_bound: Annotated[
                int,
                typer.Option(
                    help="Upper bound used for flexible batch when --dynamic-batch is enabled."
                ),
            ] = 8,
            device: Annotated[
                Device, typer.Option(help="Device used for model instantiation and tracing.")
            ] = Device.cpu,
            convert_to: Annotated[
                ConvertTo, typer.Option(help="Core ML model type to generate.")
            ] = ConvertTo.mlprogram,
            compute_precision: Annotated[
                ComputePrecision | None,
                typer.Option(help="Precision for mlprogram conversion."),
            ] = None,
            verify: Annotated[
                bool, typer.Option(help="Reload the saved Core ML model metadata after export.")
            ] = True,
        ) -> None:
            validate_common_args(batch_size=batch_size, device=device)
            if dynamic_batch:
                if batch_upper_bound < 1:
                    raise ConfigurationError("--batch-upper-bound must be >= 1.")
                if batch_upper_bound < batch_size:
                    raise ConfigurationError("--batch-upper-bound must be >= --batch-size.")
            if convert_to == ConvertTo.neuralnetwork and compute_precision is not None:
                raise ConfigurationError(
                    "--compute-precision is only supported when --convert-to mlprogram."
                )

            ct = _import_coremltools()

            output_path = Path(output).expanduser().resolve()
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
            with torch.no_grad():
                try:
                    traced_model = torch.jit.trace(model, example_input)
                except Exception as exc:
                    raise ExportError(f"TorchScript trace failed: {exc}") from exc

            input_type = ct.TensorType(
                name="input",
                shape=_build_input_shape(
                    batch_size=batch_size,
                    dynamic_batch=dynamic_batch,
                    batch_upper_bound=batch_upper_bound,
                    input_size=resolved_input_size,
                    ct=ct,
                ),
            )
            convert_kwargs: dict[str, object] = {
                "source": "pytorch",
                "inputs": [input_type],
                "convert_to": str(convert_to),
            }
            if compute_precision is not None:
                convert_kwargs["compute_precision"] = _map_compute_precision(
                    str(compute_precision), ct
                )

            try:
                coreml_model = ct.convert(traced_model, **convert_kwargs)
            except Exception as exc:
                raise ExportError(f"Core ML conversion failed: {exc}") from exc

            try:
                coreml_model.save(str(output_path))
            except Exception as exc:
                raise ExportError(f"Failed to save Core ML model: {exc}") from exc

            if verify:
                try:
                    ct.models.MLModel(str(output_path), skip_model_load=True)
                except Exception as exc:
                    raise ExportError(f"Saved Core ML model failed verification: {exc}") from exc

        return command


def _build_input_shape(
    *,
    batch_size: int,
    dynamic_batch: bool,
    batch_upper_bound: int,
    input_size: tuple[int, int, int],
    ct: object,
) -> tuple[object, int, int, int] | tuple[int, int, int, int]:
    if not dynamic_batch:
        return (batch_size, *input_size)

    batch_dim = ct.RangeDim(
        lower_bound=1,
        upper_bound=batch_upper_bound,
        default=batch_size,
        symbol="batch",
    )
    return (batch_dim, *input_size)


def _map_compute_precision(value: str, ct: object) -> object:
    if value == "float16":
        return ct.precision.FLOAT16
    return ct.precision.FLOAT32


def _import_coremltools() -> object:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise ExportError(
            "coremltools is required for Core ML export. Install dependencies with `uv sync`."
        ) from exc
    return ct
