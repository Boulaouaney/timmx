from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

import torch
import typer

from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import ExportBackend
from timmx.export.common import (
    BatchSizeOpt,
    CheckpointOpt,
    DeviceOpt,
    InChansOpt,
    InputSizeOpt,
    ModelNameArg,
    NumClassesOpt,
    OutputOpt,
    PretrainedOpt,
    prepare_export,
)
from timmx.export.types import Device


class TorchExportBackend(ExportBackend):
    name = "torch-export"
    help = "Export a timm model with torch.export (.pt2)."

    def create_command(self) -> Callable[..., None]:
        def command(
            model_name: ModelNameArg,
            output: OutputOpt,
            checkpoint: CheckpointOpt = None,
            pretrained: PretrainedOpt = False,
            num_classes: NumClassesOpt = None,
            in_chans: InChansOpt = None,
            batch_size: BatchSizeOpt = 1,
            input_size: InputSizeOpt = None,
            dynamic_batch: Annotated[
                bool, typer.Option("--dynamic-batch", help="Mark batch axis as dynamic.")
            ] = False,
            device: DeviceOpt = Device.cpu,
            strict: Annotated[
                bool, typer.Option(help="Enable strict graph capture during torch.export.")
            ] = False,
            verify: Annotated[
                bool,
                typer.Option(help="Load the saved .pt2 archive after export to validate it."),
            ] = True,
        ) -> None:
            if dynamic_batch and batch_size < 2:
                raise ConfigurationError(
                    "--dynamic-batch requires --batch-size >= 2 for stable symbolic shape capture."
                )

            prep = prepare_export(
                model_name=model_name,
                output=output,
                checkpoint=checkpoint,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_chans,
                batch_size=batch_size,
                input_size=input_size,
                device=device,
            )

            dynamic_shapes: tuple[dict[int, torch.export.Dim], ...] | None = None
            if dynamic_batch:
                dynamic_shapes = ({0: torch.export.Dim("batch")},)

            try:
                exported_program = torch.export.export(
                    prep.model,
                    (prep.example_input,),
                    dynamic_shapes=dynamic_shapes,
                    strict=strict,
                )
            except Exception as exc:
                raise ExportError(f"torch.export capture failed: {exc}") from exc

            try:
                torch.export.save(exported_program, str(prep.output_path))
            except Exception as exc:
                raise ExportError(f"Failed to save torch.export archive: {exc}") from exc

            if verify:
                try:
                    torch.export.load(str(prep.output_path))
                except Exception as exc:
                    raise ExportError(f"Saved torch.export archive failed to load: {exc}") from exc

        return command
