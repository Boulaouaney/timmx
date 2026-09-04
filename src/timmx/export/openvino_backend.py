from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import DependencyStatus, ExportBackend
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
    PretrainedOpt,
    SoftmaxOpt,
    StdOpt,
    prepare_export,
)
from timmx.export.types import Device


class OpenVINOBackend(ExportBackend):
    name = "openvino"
    help = "Export a timm model to OpenVINO IR (.xml + .bin)."

    def check_dependencies(self) -> DependencyStatus:
        try:
            import openvino  # noqa: F401
        except ImportError:
            return DependencyStatus(
                available=False,
                missing_packages=["openvino"],
                install_hint="pip install 'timmx[openvino]'",
            )
        return DependencyStatus(available=True, missing_packages=[], install_hint="")

    def create_command(self) -> Callable[..., None]:
        def command(
            model_name: ModelNameArg,
            output: Annotated[
                Path,
                typer.Option(help="Path of the IR .xml file (the .bin is written alongside)."),
            ],
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
            fp16: Annotated[
                bool, typer.Option(help="Compress weights to fp16 in the saved IR.")
            ] = True,
            verify: Annotated[
                bool,
                typer.Option(help="Reload the saved IR, compile it on CPU and run a forward pass."),
            ] = True,
            normalize: NormalizeOpt = False,
            softmax: SoftmaxOpt = False,
            mean: MeanOpt = None,
            std: StdOpt = None,
        ) -> None:
            if output.suffix != ".xml":
                raise ConfigurationError("--output must be an .xml path for OpenVINO IR.")

            ov = _import_openvino()

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
                normalize=normalize,
                softmax=softmax,
                mean=mean,
                std=std,
            )

            shape = [-1 if dynamic_batch else batch_size, *prep.resolved_input_size]
            try:
                ov_model = ov.convert_model(
                    prep.model, example_input=prep.example_input, input=[shape]
                )
                ov_model.inputs[0].get_tensor().set_names({"input"})
                ov_model.outputs[0].get_tensor().set_names({"output"})
            except Exception as exc:
                raise ExportError(f"OpenVINO conversion failed: {exc}") from exc

            try:
                ov.save_model(ov_model, str(prep.output_path), compress_to_fp16=fp16)
            except Exception as exc:
                raise ExportError(f"Failed to save OpenVINO model: {exc}") from exc

            if verify:
                try:
                    compiled = ov.Core().compile_model(str(prep.output_path), "CPU")
                    compiled(prep.example_input.cpu().numpy())
                except Exception as exc:
                    raise ExportError(f"Saved OpenVINO model failed verification: {exc}") from exc

        return command


def _import_openvino() -> object:
    try:
        import openvino as ov
    except ImportError as exc:
        raise ExportError(
            "openvino is required for OpenVINO export. Install with: pip install 'timmx[openvino]'"
        ) from exc
    return ov
