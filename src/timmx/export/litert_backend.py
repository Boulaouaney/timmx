from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import torch
import typer

from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import DependencyStatus, ExportBackend
from timmx.export.calibration import resolve_calibration_batches
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


class LiteRTMode(StrEnum):
    fp32 = "fp32"
    fp16 = "fp16"
    dynamic_int8 = "dynamic-int8"
    int8 = "int8"


class LiteRTBackend(ExportBackend):
    name = "litert"
    help = "Export a timm model to LiteRT/TFLite using litert-torch."

    def check_dependencies(self) -> DependencyStatus:
        missing = []
        try:
            import litert_torch  # noqa: F401
        except ImportError:
            missing.append("litert-torch")
        return DependencyStatus(
            available=not missing,
            missing_packages=missing,
            install_hint="pip install 'timmx[litert]'",
        )

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
            device: DeviceOpt = Device.cpu,
            mode: Annotated[
                LiteRTMode, typer.Option(help="Export precision / quantization mode.")
            ] = LiteRTMode.fp32,
            calibration_data: Annotated[
                Path | None,
                typer.Option(
                    help=(
                        "Path to a torch-saved calibration tensor with shape (N, C, H, W). "
                        "Used by quantized export modes."
                    )
                ),
            ] = None,
            calibration_steps: Annotated[
                int | None,
                typer.Option(
                    help=(
                        "Number of calibration batches to consume. "
                        "Default is 1 random batch when --calibration-data is not set, "
                        "or all full batches from --calibration-data when set."
                    )
                ),
            ] = None,
            nhwc_input: Annotated[
                bool,
                typer.Option(help="Expose the first model input as NHWC instead of NCHW."),
            ] = False,
            verify: Annotated[
                bool, typer.Option(help="Load and allocate the exported TFLite model.")
            ] = True,
        ) -> None:
            int8_modes = {LiteRTMode.dynamic_int8, LiteRTMode.int8}

            if mode in int8_modes and device != Device.cpu:
                raise ConfigurationError(
                    "LiteRT int8 modes currently require --device cpu for PT2E quantization."
                )
            if mode not in int8_modes and (
                calibration_data is not None or calibration_steps is not None
            ):
                raise ConfigurationError(
                    "--calibration-data and --calibration-steps are only valid with "
                    "--mode dynamic-int8 or --mode int8."
                )

            litert_torch = _import_litert_torch()

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

            convert_module: torch.nn.Module = prep.model
            quant_config = None
            converter_flags: dict[str, object] = {}
            example_input = prep.example_input

            if mode == LiteRTMode.fp16:
                tensorflow = _import_tensorflow()
                converter_flags = {
                    "optimizations": [tensorflow.lite.Optimize.DEFAULT],
                    "target_spec": {"supported_types": [tensorflow.float16]},
                }
            elif mode in int8_modes:
                calibration_batches = resolve_calibration_batches(
                    calibration_data=calibration_data,
                    calibration_steps=calibration_steps,
                    batch_size=batch_size,
                    input_size=prep.resolved_input_size,
                    device=prep.torch_device,
                )
                example_input = calibration_batches[0]
                convert_module, quant_config = _prepare_pt2e_quantized_module(
                    prep.model,
                    example_input,
                    calibration_batches=calibration_batches,
                    is_dynamic=(mode == LiteRTMode.dynamic_int8),
                )

            if nhwc_input:
                _validate_nhwc_input_compatibility(example_input)
                convert_module = litert_torch.to_channel_last_io(convert_module, args=[0])
                example_input = _to_nhwc_input(example_input)

            try:
                edge_model = litert_torch.convert(
                    convert_module,
                    (example_input,),
                    quant_config=quant_config,
                    _ai_edge_converter_flags=converter_flags,
                )
            except Exception as exc:
                raise ExportError(f"LiteRT conversion failed: {exc}") from exc

            try:
                edge_model.export(str(prep.output_path))
            except Exception as exc:
                raise ExportError(f"Failed to save LiteRT model: {exc}") from exc

            if verify:
                _verify_tflite_model(prep.output_path)

        return command


def _prepare_pt2e_quantized_module(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    *,
    calibration_batches: list[torch.Tensor],
    is_dynamic: bool,
) -> tuple[torch.nn.Module, object]:
    from litert_torch.quantize import pt2e_quantizer, quant_config
    from torchao.quantization.pt2e import quantize_pt2e

    try:
        exported_module = torch.export.export(model, (example_input,), strict=False).module()

        quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
            pt2e_quantizer.get_symmetric_quantization_config(
                is_per_channel=False,
                is_dynamic=is_dynamic,
            )
        )
        prepared_module = quantize_pt2e.prepare_pt2e(exported_module, quantizer)

        with torch.no_grad():
            for calibration_batch in calibration_batches:
                prepared_module(calibration_batch)

        quantized_module = quantize_pt2e.convert_pt2e(prepared_module, fold_quantize=False)
    except Exception as exc:
        mode_label = "dynamic-int8" if is_dynamic else "int8"
        raise ExportError(f"Failed to prepare {mode_label} PT2E quantized model: {exc}") from exc

    return quantized_module, quant_config.QuantConfig(pt2e_quantizer=quantizer)


def _validate_nhwc_input_compatibility(example_input: torch.Tensor) -> None:
    if example_input.ndim < 3:
        raise ConfigurationError(
            "--nhwc-input requires rank >= 3 (for NHWC -> NCHW transposition)."
        )


def _to_nhwc_input(example_input: torch.Tensor) -> torch.Tensor:
    dims = [0, *range(2, example_input.ndim), 1]
    return example_input.permute(*dims).contiguous()


def _verify_tflite_model(output_path: Path) -> None:
    try:
        from ai_edge_litert import interpreter as tfl_interpreter
    except ImportError as exc:
        raise ExportError(
            "ai-edge-litert is required to verify LiteRT export. "
            "Install with: pip install ai-edge-litert"
        ) from exc

    try:
        interpreter = tfl_interpreter.Interpreter(model_path=str(output_path))
        interpreter.allocate_tensors()
    except Exception as exc:
        raise ExportError(f"Saved LiteRT model failed verification: {exc}") from exc


def _import_litert_torch() -> object:
    try:
        import litert_torch
    except ImportError as exc:
        raise ExportError(
            "litert-torch is required for LiteRT export. Install with: pip install 'timmx[litert]'"
        ) from exc
    return litert_torch


def _import_tensorflow() -> object:
    try:
        import tensorflow
    except ImportError as exc:
        raise ExportError(
            "TensorFlow is required for LiteRT fp16 conversion flags. "
            "Install with: pip install tensorflow"
        ) from exc
    return tensorflow
