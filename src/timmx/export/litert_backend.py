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
    MeanOpt,
    ModelNameArg,
    NormalizeOpt,
    NumClassesOpt,
    OutputOpt,
    PretrainedOpt,
    SoftmaxOpt,
    StdOpt,
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
                LiteRTMode,
                typer.Option(
                    help=(
                        "Export precision: fp32, fp16 (fp16 weights), dynamic-int8 "
                        "(int8 weights, fp32 activations, no calibration) or int8 "
                        "(full integer, needs calibration)."
                    )
                ),
            ] = LiteRTMode.fp32,
            calibration_data: Annotated[
                Path | None,
                typer.Option(
                    help=(
                        "Path to calibration data: an image directory or a "
                        "torch-saved tensor (N, C, H, W). Required for --mode int8 "
                        "unless --random-calibration is set."
                    )
                ),
            ] = None,
            calibration_steps: Annotated[
                int | None,
                typer.Option(
                    help=(
                        "Number of calibration batches to consume. "
                        "Default is all full batches from --calibration-data."
                    )
                ),
            ] = None,
            calibration_samples: Annotated[
                int | None,
                typer.Option(
                    help=(
                        "Max number of images to load when --calibration-data is a "
                        "directory. Default is 128."
                    )
                ),
            ] = None,
            random_calibration: Annotated[
                bool,
                typer.Option(
                    "--random-calibration",
                    help=(
                        "Use random noise for calibration instead of real data. "
                        "Not recommended for production use."
                    ),
                ),
            ] = False,
            per_channel: Annotated[
                bool,
                typer.Option(
                    help="Use per-channel weight quantization for int8. "
                    "Disable with --no-per-channel for per-tensor."
                ),
            ] = True,
            nhwc_input: Annotated[
                bool,
                typer.Option(help="Expose the first model input as NHWC instead of NCHW."),
            ] = False,
            verify: Annotated[
                bool, typer.Option(help="Load and allocate the exported TFLite model.")
            ] = True,
            normalize: NormalizeOpt = False,
            softmax: SoftmaxOpt = False,
            mean: MeanOpt = None,
            std: StdOpt = None,
        ) -> None:
            if mode == LiteRTMode.int8 and device != Device.cpu:
                raise ConfigurationError(
                    "LiteRT int8 mode currently requires --device cpu for PT2E quantization."
                )
            if mode != LiteRTMode.int8 and (mean is not None or std is not None) and not normalize:
                raise ConfigurationError(
                    "--mean/--std require --normalize unless used for --mode int8 calibration."
                )
            if mode != LiteRTMode.int8 and (
                calibration_data is not None
                or calibration_steps is not None
                or calibration_samples is not None
                or random_calibration
            ):
                raise ConfigurationError(
                    "--calibration-data, --calibration-steps, --calibration-samples, "
                    "and --random-calibration are only valid with --mode int8."
                )
            if mode != LiteRTMode.int8 and not per_channel:
                raise ConfigurationError("--no-per-channel is only valid with --mode int8.")

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
                normalize=normalize,
                softmax=softmax,
                mean=mean if normalize else None,
                std=std if normalize else None,
            )

            convert_module: torch.nn.Module = prep.model
            quant_config = None
            example_input = prep.example_input

            if mode == LiteRTMode.int8:
                calibration_batches = resolve_calibration_batches(
                    calibration_data=calibration_data,
                    calibration_steps=calibration_steps,
                    batch_size=batch_size,
                    input_size=prep.resolved_input_size,
                    device=prep.torch_device,
                    model=prep.model,
                    calibration_samples=calibration_samples,
                    random_calibration=random_calibration,
                    mean=mean,
                    std=std,
                    normalize_images=not normalize,
                )
                example_input = calibration_batches[0]
                convert_module, quant_config = _prepare_pt2e_quantized_module(
                    prep.model,
                    example_input,
                    calibration_batches=calibration_batches,
                    per_channel=per_channel,
                )

            if nhwc_input:
                _validate_nhwc_input_compatibility(example_input)
                convert_module = litert_torch.to_channel_last_io(convert_module, args=[0])
                example_input = _to_nhwc_input(example_input)

            try:
                edge_model = litert_torch.convert(
                    convert_module, (example_input,), quant_config=quant_config
                )
            except Exception as exc:
                raise ExportError(f"LiteRT conversion failed: {exc}") from exc

            try:
                edge_model.export(str(prep.output_path))
            except Exception as exc:
                raise ExportError(f"Failed to save LiteRT model: {exc}") from exc

            if mode in (LiteRTMode.fp16, LiteRTMode.dynamic_int8):
                _quantize_weights(prep.output_path, mode)

            if verify:
                _verify_tflite_model(prep.output_path)

        return command


def _prepare_pt2e_quantized_module(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    *,
    calibration_batches: list[torch.Tensor],
    per_channel: bool = True,
) -> tuple[torch.nn.Module, object]:
    """Static int8 PT2E quantization (int8 weights and activations) with calibration."""
    from litert_torch.quantize import pt2e_quantizer, quant_config
    from torchao.quantization.pt2e import quantize_pt2e

    try:
        exported_module = torch.export.export(model, (example_input,), strict=False).module()

        quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
            pt2e_quantizer.get_symmetric_quantization_config(
                is_per_channel=per_channel,
                is_dynamic=False,
            )
        )
        prepared_module = quantize_pt2e.prepare_pt2e(exported_module, quantizer)

        with torch.no_grad():
            for calibration_batch in calibration_batches:
                prepared_module(calibration_batch)

        quantized_module = quantize_pt2e.convert_pt2e(prepared_module, fold_quantize=False)
        # Suppress LiteRT training-mode warning; graph already has eval semantics.
        quantized_module.training = False
    except Exception as exc:
        raise ExportError(f"Failed to prepare int8 PT2E quantized model: {exc}") from exc

    return quantized_module, quant_config.QuantConfig(pt2e_quantizer=quantizer)


def _quantize_weights(output_path: Path, mode: LiteRTMode) -> None:
    """Post-training weight quantization of a saved .tflite with ai-edge-quantizer.

    fp16 casts weights to float16 (dequantized at runtime); dynamic-int8 stores int8
    weights and quantizes activations on the fly. Neither needs calibration data.
    """
    try:
        from ai_edge_quantizer import qtyping, quantizer, recipe
        from ai_edge_quantizer.algorithm_manager import AlgorithmName
    except ImportError as exc:
        raise ExportError(
            "ai-edge-quantizer is required for LiteRT fp16/dynamic-int8 export. "
            "Install with: pip install 'timmx[litert]'"
        ) from exc

    try:
        qt = quantizer.Quantizer(str(output_path))
        if mode == LiteRTMode.fp16:
            qt.update_quantization_recipe(
                regex=".*",
                operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                op_config=qtyping.OpQuantizationConfig(
                    weight_tensor_config=qtyping.TensorQuantizationConfig(
                        num_bits=16, dtype=qtyping.TensorDataType.FLOAT
                    ),
                    compute_precision=qtyping.ComputePrecision.FLOAT,
                    explicit_dequantize=True,
                ),
                algorithm_key=AlgorithmName.FLOAT_CASTING,
            )
        else:
            qt.load_quantization_recipe(recipe.dynamic_wi8_afp32())
        result = qt.quantize(enable_progress_report=False)
        output_path.write_bytes(result.quantized_model)
    except Exception as exc:
        raise ExportError(f"LiteRT {mode} weight quantization failed: {exc}") from exc


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
