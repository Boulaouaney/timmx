from __future__ import annotations

import copy
import tempfile
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
    PrePostWrapper,
    PretrainedOpt,
    SoftmaxOpt,
    StdOpt,
    prepare_export,
    reference_output,
    verify_outputs,
)
from timmx.export.types import Device

DEFAULT_OPSET = 18
DEFAULT_WORKSPACE_GIB = 4


class TensorRTMode(StrEnum):
    fp32 = "fp32"
    fp16 = "fp16"
    int8 = "int8"


class TensorRTBackend(ExportBackend):
    name = "tensorrt"
    help = "Export a timm model to a TensorRT engine via ONNX."

    def check_dependencies(self) -> DependencyStatus:
        missing = []
        try:
            import tensorrt  # noqa: F401
        except ImportError:
            missing.append("tensorrt")
        try:
            import onnxscript  # noqa: F401
        except ImportError:
            missing.append("onnxscript")
        hints = []
        if "tensorrt" in missing:
            hints.append("pip install tensorrt")
        if "onnxscript" in missing:
            hints.append("pip install 'timmx[onnx]'")
        return DependencyStatus(
            available=not missing,
            missing_packages=missing,
            install_hint=" && ".join(hints) if hints else "",
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
            device: DeviceOpt = Device.cuda,
            mode: Annotated[
                TensorRTMode,
                typer.Option(
                    help=(
                        "Engine precision: fp32, fp16 (graph cast to half behind fp32 I/O) or "
                        "int8 (explicit Q/DQ quantization of conv/linear layers, needs "
                        "calibration data and torchao)."
                    )
                ),
            ] = TensorRTMode.fp32,
            workspace: Annotated[
                int, typer.Option(help="Maximum workspace memory in GiB.")
            ] = DEFAULT_WORKSPACE_GIB,
            opset: Annotated[
                int, typer.Option(help="ONNX opset version for intermediate export.")
            ] = DEFAULT_OPSET,
            dynamic_batch: Annotated[
                bool,
                typer.Option(
                    "--dynamic-batch",
                    help="Build engine with dynamic batch size using optimization profiles.",
                ),
            ] = False,
            batch_min: Annotated[
                int, typer.Option(help="Minimum batch size for dynamic batch profile.")
            ] = 1,
            batch_max: Annotated[
                int, typer.Option(help="Maximum batch size for dynamic batch profile.")
            ] = 16,
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
            keep_onnx: Annotated[
                bool,
                typer.Option(
                    "--keep-onnx", help="Keep the intermediate ONNX file alongside the engine."
                ),
            ] = False,
            verify: Annotated[
                bool,
                typer.Option(help="Run the built engine on the GPU and compare with PyTorch."),
            ] = True,
            verbose: Annotated[
                bool, typer.Option(help="Enable verbose TensorRT builder logging.")
            ] = False,
            normalize: NormalizeOpt = False,
            softmax: SoftmaxOpt = False,
            mean: MeanOpt = None,
            std: StdOpt = None,
        ) -> None:
            if device != Device.cuda:
                raise ConfigurationError("TensorRT export requires --device cuda.")

            if opset < 7:
                raise ConfigurationError("--opset must be >= 7.")

            if workspace < 1:
                raise ConfigurationError("--workspace must be >= 1 GiB.")

            if (
                mode != TensorRTMode.int8
                and (mean is not None or std is not None)
                and not normalize
            ):
                raise ConfigurationError(
                    "--mean/--std require --normalize unless used for --mode int8 calibration."
                )

            if mode != TensorRTMode.int8 and (
                calibration_data is not None
                or calibration_steps is not None
                or calibration_samples is not None
                or random_calibration
            ):
                raise ConfigurationError(
                    "--calibration-data, --calibration-steps, --calibration-samples, "
                    "and --random-calibration are only valid with --mode int8."
                )

            if dynamic_batch:
                if batch_size < 2:
                    raise ConfigurationError(
                        "--batch-size must be >= 2 with --dynamic-batch "
                        "for stable symbolic shape capture."
                    )
                if batch_min < 1:
                    raise ConfigurationError("--batch-min must be >= 1.")
                if batch_max < batch_size:
                    raise ConfigurationError("--batch-max must be >= --batch-size.")
                if batch_min > batch_size:
                    raise ConfigurationError("--batch-min must be <= --batch-size.")

            trt = _import_tensorrt()

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

            onnx_path: Path
            temp_dir: tempfile.TemporaryDirectory[str] | None = None

            if keep_onnx:
                onnx_path = prep.output_path.with_suffix(".onnx")
            else:
                temp_dir = tempfile.TemporaryDirectory()
                onnx_path = Path(temp_dir.name) / "model.onnx"

            _require_onnxscript()

            # Strongly typed TensorRT networks take their precision from the graph, so fp16 and
            # int8 are expressed in the ONNX model rather than with builder flags (removed in
            # TensorRT 11 together with implicit int8 calibration).
            dynamic_shapes: tuple[dict[int, torch.export.Dim], ...] | None = None
            if dynamic_batch:
                batch_dim = torch.export.Dim("batch", min=batch_min, max=batch_max)
                dynamic_shapes = ({0: batch_dim},)

            export_model: torch.nn.Module | torch.export.ExportedProgram = prep.model
            verify_input = prep.example_input
            if mode == TensorRTMode.fp16:
                export_model = _half_model(prep.model)
            elif mode == TensorRTMode.int8:
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
                verify_input = calibration_batches[0]
                export_model = _quantize_int8(
                    prep.model, prep.example_input, calibration_batches, dynamic_shapes
                )

            export_kwargs: dict[str, object] = {
                "opset_version": opset,
                "input_names": ["input"],
                "output_names": ["output"],
                "dynamo": True,
                "external_data": False,
            }
            if dynamic_shapes is not None and mode != TensorRTMode.int8:
                export_kwargs["dynamic_shapes"] = dynamic_shapes

            try:
                torch.onnx.export(
                    export_model,
                    (prep.example_input,),
                    f=str(onnx_path),
                    **export_kwargs,
                )
            except Exception as exc:
                if temp_dir is not None:
                    temp_dir.cleanup()
                raise ExportError(f"Intermediate ONNX export failed: {exc}") from exc

            try:
                logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
                builder = trt.Builder(logger)
                network = builder.create_network(
                    1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
                )
                parser = trt.OnnxParser(network, logger)

                with open(onnx_path, "rb") as onnx_file:
                    if not parser.parse(onnx_file.read()):
                        errors = [parser.get_error(i) for i in range(parser.num_errors)]
                        error_msgs = "\n".join(str(e) for e in errors)
                        raise ExportError(f"TensorRT ONNX parsing failed:\n{error_msgs}")

                config = builder.create_builder_config()
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 30))

                if dynamic_batch:
                    profile = builder.create_optimization_profile()
                    input_shape_min = (batch_min, *prep.resolved_input_size)
                    input_shape_opt = (batch_size, *prep.resolved_input_size)
                    input_shape_max = (batch_max, *prep.resolved_input_size)
                    profile.set_shape("input", input_shape_min, input_shape_opt, input_shape_max)
                    config.add_optimization_profile(profile)

                serialized_engine = builder.build_serialized_network(network, config)
                if serialized_engine is None:
                    raise ExportError("TensorRT engine build returned None.")

            except ExportError:
                raise
            except Exception as exc:
                raise ExportError(f"TensorRT engine build failed: {exc}") from exc
            finally:
                if temp_dir is not None:
                    temp_dir.cleanup()

            try:
                prep.output_path.write_bytes(serialized_engine)
            except Exception as exc:
                raise ExportError(
                    f"Failed to write TensorRT engine to {prep.output_path}: {exc}"
                ) from exc

            if verify:
                _verify_engine(
                    trt,
                    prep.output_path,
                    verify_input,
                    reference_output(prep.model, verify_input),
                )

        return command


def _verify_engine(
    trt: object, engine_path: Path, runtime_input: torch.Tensor, expected: torch.Tensor
) -> None:
    try:
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        context = engine.create_execution_context()
        x = runtime_input.cuda().contiguous()
        context.set_input_shape("input", tuple(x.shape))
        output_shape = tuple(context.get_tensor_shape("output"))
        y = torch.empty(output_shape, dtype=torch.float32, device="cuda")
        context.set_tensor_address("input", x.data_ptr())
        context.set_tensor_address("output", y.data_ptr())
        stream = torch.cuda.current_stream()
        if not context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError("execute_async_v3 returned False")
        stream.synchronize()
    except Exception as exc:
        raise ExportError(f"Saved TensorRT engine failed verification: {exc}") from exc
    verify_outputs(expected, y.cpu().numpy(), backend="TensorRT")


class _HalfIO(torch.nn.Module):
    """Run the model in fp16 behind fp32 inputs and outputs."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model.half()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.half()).float()


def _half_model(model: torch.nn.Module) -> torch.nn.Module:
    """Cast the network to fp16, keeping --normalize/--softmax pre/post-processing in fp32.

    A strongly typed engine runs softmax in fp16 if the graph says so, and exp() of real logits
    overflows half precision (probabilities came out with cosine 0.79 against PyTorch).
    """
    model = copy.deepcopy(model)
    if isinstance(model, PrePostWrapper):
        model.model = _HalfIO(model.model)
        return model
    return _HalfIO(model)


def _quantize_int8(
    model: torch.nn.Module,
    example_input: torch.Tensor,
    calibration_batches: list[torch.Tensor],
    dynamic_shapes: tuple[dict[int, torch.export.Dim], ...] | None,
) -> torch.export.ExportedProgram:
    """PT2E static int8 in the shape TensorRT's ONNX parser accepts.

    Symmetric (zero point 0) per-tensor activations (histogram-calibrated ranges, which survive
    activation outliers far better than min/max) and per-channel weights, quantized on the
    inputs of conv/linear layers only: TensorRT rejects asymmetric int8 and fuses conv+relu+pool
    itself, which a Q/DQ pair on the conv output would break ("could not find any implementation").
    """
    try:
        from torchao.quantization.pt2e import quantize_pt2e
        from torchao.quantization.pt2e.observer import (
            HistogramObserver,
            PerChannelMinMaxObserver,
        )
        from torchao.quantization.pt2e.quantizer import (
            QuantizationAnnotation,
            QuantizationSpec,
            Quantizer,
        )
    except ImportError as exc:
        raise ExportError(
            "torchao is required for --mode int8 (explicit Q/DQ quantization). "
            "Install with: pip install torchao"
        ) from exc

    activation = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-128,
        quant_max=127,
        qscheme=torch.per_tensor_symmetric,
        observer_or_fake_quant_ctr=HistogramObserver.with_args(eps=2**-12),
    )
    weight = QuantizationSpec(
        dtype=torch.int8,
        quant_min=-127,
        quant_max=127,
        qscheme=torch.per_channel_symmetric,
        ch_axis=0,
        observer_or_fake_quant_ctr=PerChannelMinMaxObserver.with_args(eps=2**-12),
    )
    targets = {torch.ops.aten.conv2d.default, torch.ops.aten.linear.default}

    class _TensorRTQuantizer(Quantizer):
        def annotate(self, graph_module: torch.fx.GraphModule) -> torch.fx.GraphModule:
            for node in graph_module.graph.nodes:
                if node.op == "call_function" and node.target in targets:
                    node.meta["quantization_annotation"] = QuantizationAnnotation(
                        input_qspec_map={node.args[0]: activation, node.args[1]: weight},
                        _annotated=True,
                    )
            return graph_module

        def validate(self, graph_module: torch.fx.GraphModule) -> None:
            pass

    # Capture with the batch symbol from the start: a module from a static export asserts the
    # traced batch size, which specializes a later dynamic re-export.
    try:
        exported = torch.export.export(
            model, (example_input,), dynamic_shapes=dynamic_shapes
        ).module()
        prepared = quantize_pt2e.prepare_pt2e(exported, _TensorRTQuantizer())
        with torch.no_grad():
            for batch in calibration_batches:
                prepared(batch)
        quantized = quantize_pt2e.convert_pt2e(prepared)
        quantized.training = False
        # The ONNX exporter keeps dynamic shapes from an ExportedProgram, not a GraphModule.
        return torch.export.export(quantized, (example_input,), dynamic_shapes=dynamic_shapes)
    except Exception as exc:
        raise ExportError(f"PT2E quantization failed: {exc}") from exc


def _require_onnxscript() -> None:
    try:
        import onnxscript  # noqa: F401
    except ImportError as exc:
        raise ExportError(
            "onnxscript is required for TensorRT export (dynamo-based ONNX export). "
            "Install with: pip install 'timmx[onnx]'"
        ) from exc


def _import_tensorrt() -> object:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ExportError(
            "tensorrt is required for TensorRT export. "
            "Install with: pip install tensorrt (requires CUDA)"
        ) from exc
    if not hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        raise ExportError(
            f"TensorRT >= 10 is required (found {getattr(trt, '__version__', 'unknown')})."
        )
    return trt
