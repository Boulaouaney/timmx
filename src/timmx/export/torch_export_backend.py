from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
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
    MeanOpt,
    ModelNameArg,
    NormalizeOpt,
    NumClassesOpt,
    OutputOpt,
    PretrainedOpt,
    SoftmaxOpt,
    StdOpt,
    batch_dynamic_shapes,
    capture_program,
    prepare_export,
    reference_output,
    verify_outputs,
)
from timmx.export.types import Device


class TorchExportBackend(ExportBackend):
    name = "torch-export"
    help = "Export a timm model with torch.export (.pt2)."

    def create_command(self) -> Callable[..., Path]:
        def command(
            model_name: ModelNameArg,
            output: OutputOpt = None,
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
            normalize: NormalizeOpt = False,
            softmax: SoftmaxOpt = False,
            mean: MeanOpt = None,
            std: StdOpt = None,
            verify: Annotated[
                bool,
                typer.Option(
                    help="Reload the saved .pt2 archive and compare its output with PyTorch."
                ),
            ] = True,
        ) -> Path:
            if dynamic_batch and batch_size < 2:
                raise ConfigurationError(
                    "--dynamic-batch requires --batch-size >= 2 for stable symbolic shape capture."
                )

            prep = prepare_export(
                model_name=model_name,
                output=output,
                default_suffix=".pt2",
                checkpoint=checkpoint,
                pretrained=pretrained,
                num_classes=num_classes,
                in_chans=in_chans,
                batch_size=batch_size,
                input_size=input_size,
                device=device,
                normalize=normalize,
                softmax=softmax,
                mean=mean,
                std=std,
            )

            exported_program = capture_program(
                prep.model,
                prep.example_input,
                dynamic_shapes=batch_dynamic_shapes(dynamic_batch),
                strict=strict,
            )

            try:
                torch.export.save(exported_program, str(prep.output_path))
            except Exception as exc:
                raise ExportError(f"Failed to save torch.export archive: {exc}") from exc

            if verify:
                try:
                    loaded = torch.export.load(str(prep.output_path)).module()
                    with torch.no_grad():
                        actual = loaded(prep.example_input)
                except Exception as exc:
                    raise ExportError(f"Saved torch.export archive failed to load: {exc}") from exc
                expected = reference_output(prep.model, prep.example_input)
                verify_outputs(expected, actual.cpu().numpy(), backend="torch.export")

            return prep.output_path

        return command
