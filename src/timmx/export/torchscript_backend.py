from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from typing import Annotated

import torch
import typer

from timmx.errors import ExportError
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


class ScriptMethod(StrEnum):
    trace = "trace"
    script = "script"


class TorchScriptBackend(ExportBackend):
    name = "torchscript"
    help = "Export a timm model to TorchScript (.pt)."

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
            method: Annotated[
                ScriptMethod,
                typer.Option(help="Scripting method: trace (recommended) or script."),
            ] = ScriptMethod.trace,
            device: DeviceOpt = Device.cpu,
            verify: Annotated[
                bool,
                typer.Option(help="Load the saved model after export and run a forward pass."),
            ] = True,
        ) -> None:
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

            try:
                if method == ScriptMethod.trace:
                    scripted = torch.jit.trace(prep.model, prep.example_input)
                else:
                    scripted = torch.jit.script(prep.model)
            except Exception as exc:
                raise ExportError(f"TorchScript {method} failed: {exc}") from exc

            try:
                torch.jit.save(scripted, str(prep.output_path))
            except Exception as exc:
                raise ExportError(f"Failed to save TorchScript model: {exc}") from exc

            if verify:
                try:
                    loaded = torch.jit.load(str(prep.output_path), map_location=prep.torch_device)
                    loaded(prep.example_input)
                except Exception as exc:
                    raise ExportError(
                        f"Saved TorchScript model failed verification: {exc}"
                    ) from exc

        return command
