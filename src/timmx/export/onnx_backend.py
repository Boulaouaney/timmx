from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import onnx
import torch
import typer

from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import ExportBackend
from timmx.export.common import create_timm_model, resolve_input_size, validate_common_args
from timmx.export.types import Device

DEFAULT_OPSET = 18


class OnnxBackend(ExportBackend):
    name = "onnx"
    help = "Export a timm model to ONNX."

    def create_command(self) -> Callable[..., None]:
        def command(
            model_name: Annotated[str, typer.Argument(help="timm model name, e.g. resnet18")],
            output: Annotated[Path, typer.Option(help="Path to write the ONNX model.")],
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
            opset: Annotated[
                int, typer.Option(help="ONNX opset version to target.")
            ] = DEFAULT_OPSET,
            dynamic_batch: Annotated[
                bool, typer.Option("--dynamic-batch", help="Mark batch axis as dynamic.")
            ] = False,
            device: Annotated[
                Device, typer.Option(help="Device used for model instantiation and tracing.")
            ] = Device.cpu,
            external_data: Annotated[
                bool, typer.Option(help="Save large model weights in external data files.")
            ] = False,
            check: Annotated[bool, typer.Option(help="Run ONNX checker after export.")] = True,
        ) -> None:
            validate_common_args(batch_size=batch_size, device=device)
            if opset < 7:
                raise ConfigurationError("--opset must be >= 7.")

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

            export_kwargs: dict[str, object] = {
                "f": str(output_path),
                "opset_version": opset,
                "input_names": ["input"],
                "output_names": ["output"],
                "dynamo": True,
                "fallback": True,
                "external_data": external_data,
            }
            if dynamic_batch:
                export_kwargs["dynamic_shapes"] = ({0: torch.export.Dim("batch")},)
                export_kwargs["dynamic_axes"] = {"input": {0: "batch"}, "output": {0: "batch"}}

            try:
                torch.onnx.export(model, (example_input,), **export_kwargs)
            except Exception as exc:
                raise ExportError(f"ONNX export failed: {exc}") from exc

            if check:
                try:
                    onnx.checker.check_model(str(output_path))
                except Exception as exc:
                    raise ExportError(f"Exported model failed ONNX check: {exc}") from exc

        return command
