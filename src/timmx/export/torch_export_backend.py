from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import torch
import typer

from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import ExportBackend
from timmx.export.common import create_timm_model, resolve_input_size, validate_common_args
from timmx.export.types import Device


class TorchExportBackend(ExportBackend):
    name = "torch-export"
    help = "Export a timm model with torch.export (.pt2)."

    def create_command(self) -> Callable[..., None]:
        def command(
            model_name: Annotated[str, typer.Argument(help="timm model name, e.g. resnet18")],
            output: Annotated[
                Path,
                typer.Option(help="Path to write the exported program archive (typically .pt2)."),
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
                bool, typer.Option("--dynamic-batch", help="Mark batch axis as dynamic.")
            ] = False,
            device: Annotated[
                Device, typer.Option(help="Device used for model instantiation and tracing.")
            ] = Device.cpu,
            strict: Annotated[
                bool, typer.Option(help="Enable strict graph capture during torch.export.")
            ] = False,
            verify: Annotated[
                bool,
                typer.Option(help="Load the saved .pt2 archive after export to validate it."),
            ] = True,
        ) -> None:
            validate_common_args(batch_size=batch_size, device=device)
            if dynamic_batch and batch_size < 2:
                raise ConfigurationError(
                    "--dynamic-batch requires --batch-size >= 2 for stable symbolic shape capture."
                )

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
            dynamic_shapes: tuple[dict[int, torch.export.Dim], ...] | None = None
            if dynamic_batch:
                dynamic_shapes = ({0: torch.export.Dim("batch")},)

            try:
                exported_program = torch.export.export(
                    model,
                    (example_input,),
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                )
            except Exception as exc:
                raise ExportError(f"torch.export capture failed: {exc}") from exc

            try:
                torch.export.save(exported_program, str(output_path))
            except Exception as exc:
                raise ExportError(f"Failed to save torch.export archive: {exc}") from exc

            if verify:
                try:
                    torch.export.load(str(output_path))
                except Exception as exc:
                    raise ExportError(f"Saved torch.export archive failed to load: {exc}") from exc

        return command
