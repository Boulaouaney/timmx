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


class ExecuTorchDelegate(StrEnum):
    xnnpack = "xnnpack"
    coreml = "coreml"


class ExecuTorchMode(StrEnum):
    fp32 = "fp32"
    int8 = "int8"


class ComputePrecision(StrEnum):
    float16 = "float16"
    float32 = "float32"


class ExecuTorchBackend(ExportBackend):
    name = "executorch"
    help = "Export a timm model to ExecuTorch (.pte) with delegate acceleration."

    def check_dependencies(self) -> DependencyStatus:
        missing = []
        try:
            import executorch  # noqa: F401
        except ImportError:
            missing.append("executorch")
        return DependencyStatus(
            available=not missing,
            missing_packages=missing,
            install_hint="pip install 'timmx[executorch]'",
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
            delegate: Annotated[
                ExecuTorchDelegate,
                typer.Option(help="ExecuTorch delegate backend for hardware acceleration."),
            ] = ExecuTorchDelegate.xnnpack,
            mode: Annotated[
                ExecuTorchMode,
                typer.Option(help="Export precision mode."),
            ] = ExecuTorchMode.fp32,
            compute_precision: Annotated[
                ComputePrecision | None,
                typer.Option(help="CoreML compute precision. Only valid with --delegate coreml."),
            ] = None,
            dynamic_batch: Annotated[
                bool,
                typer.Option("--dynamic-batch", help="Mark batch axis as dynamic."),
            ] = False,
            calibration_data: Annotated[
                Path | None,
                typer.Option(
                    help="Path to a torch-saved calibration tensor (N, C, H, W) for int8."
                ),
            ] = None,
            calibration_steps: Annotated[
                int | None,
                typer.Option(help="Number of calibration batches for int8 quantization."),
            ] = None,
            per_channel: Annotated[
                bool,
                typer.Option(
                    help="Use per-channel quantization for int8. "
                    "Disable with --no-per-channel for per-tensor."
                ),
            ] = True,
        ) -> None:
            if compute_precision is not None and delegate != ExecuTorchDelegate.coreml:
                raise ConfigurationError(
                    "--compute-precision is only supported with --delegate coreml."
                )
            if mode != ExecuTorchMode.int8 and (
                calibration_data is not None or calibration_steps is not None
            ):
                raise ConfigurationError(
                    "--calibration-data and --calibration-steps are only valid with --mode int8."
                )
            if dynamic_batch and batch_size < 2:
                raise ConfigurationError(
                    "--dynamic-batch requires --batch-size >= 2 for stable symbolic shape capture."
                )

            _import_executorch()

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

            partitioner = _build_partitioner(
                delegate=delegate,
                compute_precision=compute_precision,
                quantized=mode == ExecuTorchMode.int8,
            )

            if mode == ExecuTorchMode.int8:
                et_program = _export_quantized(
                    model=prep.model,
                    example_input=prep.example_input,
                    resolved_input_size=prep.resolved_input_size,
                    torch_device=prep.torch_device,
                    batch_size=batch_size,
                    calibration_data=calibration_data,
                    calibration_steps=calibration_steps,
                    per_channel=per_channel,
                    dynamic_batch=dynamic_batch,
                    delegate=delegate,
                    partitioner=partitioner,
                )
            else:
                et_program = _export_standard(
                    model=prep.model,
                    example_input=prep.example_input,
                    dynamic_batch=dynamic_batch,
                    partitioner=partitioner,
                )

            try:
                with open(prep.output_path, "wb") as f:
                    et_program.write_to_file(f)
            except Exception as exc:
                raise ExportError(
                    f"Failed to write ExecuTorch model to {prep.output_path}: {exc}"
                ) from exc

        return command


def _export_standard(
    *,
    model: torch.nn.Module,
    example_input: torch.Tensor,
    dynamic_batch: bool,
    partitioner: list[object],
) -> object:
    from executorch.exir import to_edge_transform_and_lower

    dynamic_shapes: tuple[dict[int, torch.export.Dim], ...] | None = None
    if dynamic_batch:
        dynamic_shapes = ({0: torch.export.Dim("batch")},)

    try:
        exported_program = torch.export.export(
            model,
            (example_input,),
            dynamic_shapes=dynamic_shapes,
        )
    except Exception as exc:
        raise ExportError(f"torch.export capture failed: {exc}") from exc

    try:
        et_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=partitioner,
        ).to_executorch()
    except Exception as exc:
        raise ExportError(f"ExecuTorch edge lowering failed: {exc}") from exc

    return et_program


def _export_quantized(
    *,
    model: torch.nn.Module,
    example_input: torch.Tensor,
    resolved_input_size: tuple[int, int, int],
    torch_device: torch.device,
    batch_size: int,
    calibration_data: Path | None,
    calibration_steps: int | None,
    per_channel: bool,
    dynamic_batch: bool,
    delegate: ExecuTorchDelegate,
    partitioner: list[object],
) -> object:
    from executorch.exir import EdgeCompileConfig, to_edge_transform_and_lower
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    calibration_batches = resolve_calibration_batches(
        calibration_data=calibration_data,
        calibration_steps=calibration_steps,
        batch_size=batch_size,
        input_size=resolved_input_size,
        device=torch_device,
    )

    quantizer = _build_quantizer(delegate=delegate, per_channel=per_channel)

    try:
        exported_module = torch.export.export(model, (example_input,)).module()

        prepared = prepare_pt2e(exported_module, quantizer)

        with torch.no_grad():
            for batch in calibration_batches:
                prepared(batch)

        quantized = convert_pt2e(prepared)
    except Exception as exc:
        raise ExportError(f"PT2E quantization failed: {exc}") from exc

    dynamic_shapes: tuple[dict[int, torch.export.Dim], ...] | None = None
    if dynamic_batch:
        dynamic_shapes = ({0: torch.export.Dim("batch")},)

    try:
        exported_program = torch.export.export(
            quantized,
            (example_input,),
            dynamic_shapes=dynamic_shapes,
        )
    except Exception as exc:
        raise ExportError(f"torch.export capture of quantized model failed: {exc}") from exc

    try:
        et_program = to_edge_transform_and_lower(
            exported_program,
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
            partitioner=partitioner,
        ).to_executorch()
    except Exception as exc:
        raise ExportError(f"ExecuTorch edge lowering of quantized model failed: {exc}") from exc

    return et_program


def _build_quantizer(*, delegate: ExecuTorchDelegate, per_channel: bool) -> object:
    if delegate == ExecuTorchDelegate.xnnpack:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        quantizer = XNNPACKQuantizer()
        quantizer.set_global(
            get_symmetric_quantization_config(
                is_per_channel=per_channel,
                is_dynamic=False,
            )
        )
        return quantizer

    if delegate == ExecuTorchDelegate.coreml:
        from coremltools.optimize.torch.quantization import LinearQuantizerConfig
        from executorch.backends.apple.coreml.quantizer import CoreMLQuantizer

        config = LinearQuantizerConfig.from_dict(
            {
                "global_config": {
                    "weight_dtype": "qint8",
                    "activation_dtype": "quint8",
                    "weight_per_channel": per_channel,
                }
            }
        )
        return CoreMLQuantizer(config)

    raise ConfigurationError(f"Unknown delegate: {delegate}")


def _build_partitioner(
    *,
    delegate: ExecuTorchDelegate,
    compute_precision: ComputePrecision | None,
    quantized: bool = False,
) -> list[object]:
    if delegate == ExecuTorchDelegate.xnnpack:
        try:
            from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
                XnnpackPartitioner,
            )
        except ImportError as exc:
            raise ExportError(
                "XnnpackPartitioner is required for --delegate xnnpack. "
                "Install with: pip install 'timmx[executorch]'"
            ) from exc
        return [XnnpackPartitioner()]

    if delegate == ExecuTorchDelegate.coreml:
        try:
            from executorch.backends.apple.coreml.partition import CoreMLPartitioner
        except ImportError as exc:
            raise ExportError(
                "CoreMLPartitioner is required for --delegate coreml. "
                "Ensure executorch is installed with CoreML support (macOS only)."
            ) from exc

        compile_specs = _build_coreml_compile_specs(compute_precision, quantized=quantized)
        if compile_specs is not None:
            return [CoreMLPartitioner(compile_specs=compile_specs)]
        return [CoreMLPartitioner()]

    raise ConfigurationError(f"Unknown delegate: {delegate}")


def _build_coreml_compile_specs(
    compute_precision: ComputePrecision | None,
    *,
    quantized: bool = False,
) -> list[object] | None:
    if compute_precision is None and not quantized:
        return None

    import coremltools as ct
    from executorch.backends.apple.coreml.compiler import CoreMLBackend

    kwargs: dict[str, object] = {}
    if compute_precision is not None:
        kwargs["compute_precision"] = (
            ct.precision.FLOAT32
            if compute_precision == ComputePrecision.float32
            else ct.precision.FLOAT16
        )
    if quantized:
        kwargs["minimum_deployment_target"] = ct.target.iOS17
    return CoreMLBackend.generate_compile_specs(**kwargs)


def _import_executorch() -> None:
    try:
        import executorch  # noqa: F401
    except ImportError as exc:
        raise ExportError(
            "executorch is required for ExecuTorch export. "
            "Install with: pip install 'timmx[executorch]'"
        ) from exc
