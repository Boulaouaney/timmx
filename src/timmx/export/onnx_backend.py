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

DEFAULT_OPSET = 18


class OnnxBackend(ExportBackend):
    name = "onnx"
    help = "Export a timm model to ONNX."

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
            opset: Annotated[
                int, typer.Option(help="ONNX opset version to target.")
            ] = DEFAULT_OPSET,
            dynamic_batch: Annotated[
                bool, typer.Option("--dynamic-batch", help="Mark batch axis as dynamic.")
            ] = False,
            device: DeviceOpt = Device.cpu,
            external_data: Annotated[
                bool, typer.Option(help="Save large model weights in external data files.")
            ] = False,
            check: Annotated[bool, typer.Option(help="Run ONNX checker after export.")] = True,
        ) -> None:
            if opset < 7:
                raise ConfigurationError("--opset must be >= 7.")

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

            export_kwargs: dict[str, object] = {
                "f": str(prep.output_path),
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
                torch.onnx.export(prep.model, (prep.example_input,), **export_kwargs)
            except Exception as exc:
                raise ExportError(f"ONNX export failed: {exc}") from exc

            if check:
                import onnx

                try:
                    onnx.checker.check_model(str(prep.output_path))
                except Exception as exc:
                    raise ExportError(f"Exported model failed ONNX check: {exc}") from exc

        return command
