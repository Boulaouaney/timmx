from __future__ import annotations

import tempfile
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import torch
import typer

from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import ExportBackend
from timmx.export.calibration import resolve_calibration_batches
from timmx.export.common import create_timm_model, resolve_input_size, validate_common_args
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

    def create_command(self) -> Callable[..., None]:
        def command(
            model_name: Annotated[str, typer.Argument(help="timm model name, e.g. resnet18")],
            output: Annotated[
                Path, typer.Option(help="Path to write the TensorRT engine (.engine).")
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
                int, typer.Option(help="Example input batch size for export and calibration.")
            ] = 1,
            input_size: Annotated[
                tuple[int, int, int] | None,
                typer.Option(help="Explicit input shape as C H W."),
            ] = None,
            device: Annotated[
                Device, typer.Option(help="Device used for model instantiation and tracing.")
            ] = Device.cuda,
            mode: Annotated[
                TensorRTMode, typer.Option(help="Engine precision mode.")
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
                        "Path to a torch-saved calibration tensor with shape (N, C, H, W). "
                        "Required for --mode int8 unless random calibration is acceptable."
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
            calibration_cache: Annotated[
                Path | None,
                typer.Option(help="Path to read/write TensorRT INT8 calibration cache."),
            ] = None,
            keep_onnx: Annotated[
                bool,
                typer.Option(
                    "--keep-onnx", help="Keep the intermediate ONNX file alongside the engine."
                ),
            ] = False,
            verbose: Annotated[
                bool, typer.Option(help="Enable verbose TensorRT builder logging.")
            ] = False,
            exportable: Annotated[
                bool,
                typer.Option(help="Use timm export-friendly layer variants when available."),
            ] = True,
        ) -> None:
            validate_common_args(batch_size=batch_size, device=device)

            if device != Device.cuda:
                raise ConfigurationError("TensorRT export requires --device cuda.")

            if opset < 7:
                raise ConfigurationError("--opset must be >= 7.")

            if workspace < 1:
                raise ConfigurationError("--workspace must be >= 1 GiB.")

            if mode != TensorRTMode.int8 and (
                calibration_data is not None
                or calibration_steps is not None
                or calibration_cache is not None
            ):
                raise ConfigurationError(
                    "--calibration-data, --calibration-steps, and --calibration-cache "
                    "are only valid with --mode int8."
                )

            if dynamic_batch:
                if batch_min < 1:
                    raise ConfigurationError("--batch-min must be >= 1.")
                if batch_max < batch_size:
                    raise ConfigurationError("--batch-max must be >= --batch-size.")
                if batch_min > batch_size:
                    raise ConfigurationError("--batch-min must be <= --batch-size.")

            trt = _import_tensorrt()

            output_path = Path(output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            model = create_timm_model(
                model_name,
                pretrained=pretrained,
                checkpoint=checkpoint,
                num_classes=num_classes,
                in_chans=in_chans,
                exportable=exportable,
            )
            resolved_input_size = resolve_input_size(model, input_size)

            torch_device = torch.device(device)
            model = model.to(torch_device)
            model.eval()

            onnx_path: Path
            temp_dir: tempfile.TemporaryDirectory[str] | None = None

            if keep_onnx:
                onnx_path = output_path.with_suffix(".onnx")
            else:
                temp_dir = tempfile.TemporaryDirectory()
                onnx_path = Path(temp_dir.name) / "model.onnx"

            example_input = torch.randn(batch_size, *resolved_input_size, device=torch_device)

            try:
                torch.onnx.export(
                    model,
                    (example_input,),
                    f=str(onnx_path),
                    opset_version=opset,
                    input_names=["input"],
                    output_names=["output"],
                    dynamo=True,
                    fallback=True,
                )
            except Exception as exc:
                if temp_dir is not None:
                    temp_dir.cleanup()
                raise ExportError(f"Intermediate ONNX export failed: {exc}") from exc

            try:
                logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
                builder = trt.Builder(logger)
                network = builder.create_network(
                    1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                )
                parser = trt.OnnxParser(network, logger)

                with open(onnx_path, "rb") as onnx_file:
                    if not parser.parse(onnx_file.read()):
                        errors = [parser.get_error(i) for i in range(parser.num_errors)]
                        error_msgs = "\n".join(str(e) for e in errors)
                        raise ExportError(f"TensorRT ONNX parsing failed:\n{error_msgs}")

                config = builder.create_builder_config()
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace * (1 << 30))

                if mode == TensorRTMode.fp16:
                    config.set_flag(trt.BuilderFlag.FP16)
                elif mode == TensorRTMode.int8:
                    config.set_flag(trt.BuilderFlag.INT8)
                    calibrator = _create_calibrator(
                        trt=trt,
                        calibration_data=calibration_data,
                        calibration_steps=calibration_steps,
                        calibration_cache=calibration_cache,
                        batch_size=batch_size,
                        input_size=resolved_input_size,
                        device=torch_device,
                    )
                    config.int8_calibrator = calibrator

                if dynamic_batch:
                    profile = builder.create_optimization_profile()
                    input_shape_min = (batch_min, *resolved_input_size)
                    input_shape_opt = (batch_size, *resolved_input_size)
                    input_shape_max = (batch_max, *resolved_input_size)
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
                output_path.write_bytes(serialized_engine)
            except Exception as exc:
                raise ExportError(
                    f"Failed to write TensorRT engine to {output_path}: {exc}"
                ) from exc

        return command


def _make_calibrator_class(trt: object) -> type:
    class _Calibrator(trt.IInt8MinMaxCalibrator):
        def __init__(self, batches: list[torch.Tensor], cache_path: Path) -> None:
            super().__init__()
            self._batches = batches
            self._batch_iter = iter(batches)
            self._batch_size = batches[0].shape[0]
            self._cache_path = cache_path
            self._current_batch: torch.Tensor | None = None

        def get_batch_size(self) -> int:
            return self._batch_size

        def get_batch(self, names: list[str]) -> list[int] | None:
            try:
                batch = next(self._batch_iter)
                self._current_batch = batch.cuda().contiguous()
                return [self._current_batch.data_ptr()]
            except StopIteration:
                return None

        def read_calibration_cache(self) -> bytes | None:
            if self._cache_path.exists():
                return self._cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache: bytes) -> None:
            self._cache_path.write_bytes(cache)

    return _Calibrator


def _create_calibrator(
    *,
    trt: object,
    calibration_data: Path | None,
    calibration_steps: int | None,
    calibration_cache: Path | None,
    batch_size: int,
    input_size: tuple[int, int, int],
    device: torch.device,
) -> object:
    batches = resolve_calibration_batches(
        calibration_data=calibration_data,
        calibration_steps=calibration_steps,
        batch_size=batch_size,
        input_size=input_size,
        device=device,
    )
    cache_path = calibration_cache or Path("calibration.cache")
    calibrator_cls = _make_calibrator_class(trt)
    return calibrator_cls(batches=batches, cache_path=cache_path)


def _import_tensorrt() -> object:
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise ExportError(
            "tensorrt is required for TensorRT export. "
            "Install with `uv pip install tensorrt` on a system with CUDA."
        ) from exc
    return trt
