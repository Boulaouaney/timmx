from __future__ import annotations

import random
import sys
import tempfile
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import torch
import typer

from timmx.console import console
from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import DependencyStatus, ExportBackend
from timmx.export.calibration import (
    DEFAULT_CALIBRATION_SAMPLES,
    collect_image_paths,
)
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
    resolve_normalization_stats,
)
from timmx.export.types import Device

DEFAULT_OPSET = 19
MAX_OPSET = 19


class RknnMode(StrEnum):
    fp32 = "fp32"
    fp16 = "fp16"
    int8 = "int8"


class RknnQuantAlgorithm(StrEnum):
    normal = "normal"
    mmse = "mmse"
    kl_divergence = "kl_divergence"


class RknnQuantMethod(StrEnum):
    channel = "channel"
    layer = "layer"


class RknnBackend(ExportBackend):
    name = "rknn"
    help = "Export a timm model to RKNN for Rockchip NPUs via ONNX."

    def check_dependencies(self) -> DependencyStatus:
        if sys.platform != "linux":
            return DependencyStatus(
                available=False,
                missing_packages=["rknn-toolkit2"],
                install_hint=(
                    "rknn-toolkit2 is only available on Linux (x86_64, aarch64). "
                    "pip install rknn-toolkit2"
                ),
            )
        try:
            from rknn.api import RKNN  # noqa: F401
        except ImportError:
            return DependencyStatus(
                available=False,
                missing_packages=["rknn-toolkit2"],
                install_hint="pip install rknn-toolkit2",
            )
        return DependencyStatus(available=True, missing_packages=[], install_hint="")

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
            target_platform: Annotated[
                str,
                typer.Option(
                    help=(
                        "Rockchip target platform "
                        "(e.g. rk3588, rk3576, rk3566, rk3568, rk3562, "
                        "rv1103, rv1106)."
                    )
                ),
            ] = "rk3588",
            mode: Annotated[RknnMode, typer.Option(help="Quantization mode.")] = RknnMode.fp16,
            quant_algorithm: Annotated[
                RknnQuantAlgorithm,
                typer.Option(
                    help="Quantization algorithm (int8 only). "
                    "normal=fast min/max, mmse=better accuracy, "
                    "kl_divergence=information-theoretic."
                ),
            ] = RknnQuantAlgorithm.normal,
            quant_method: Annotated[
                RknnQuantMethod,
                typer.Option(
                    help="Quantization granularity (int8 only). "
                    "channel=per-channel, layer=per-tensor."
                ),
            ] = RknnQuantMethod.channel,
            calibration_data: Annotated[
                Path | None,
                typer.Option(
                    help=("Path to a directory of calibration images. Required for --mode int8.")
                ),
            ] = None,
            calibration_samples: Annotated[
                int | None,
                typer.Option(
                    help=(
                        "Max number of images to use from --calibration-data. "
                        f"Default is {DEFAULT_CALIBRATION_SAMPLES}."
                    )
                ),
            ] = None,
            random_calibration: Annotated[
                bool,
                typer.Option(
                    "--random-calibration",
                    help="Not supported for RKNN (RKNN requires real image paths).",
                ),
            ] = False,
            opset: Annotated[
                int,
                typer.Option(help="ONNX opset version for intermediate export."),
            ] = DEFAULT_OPSET,
            keep_onnx: Annotated[
                bool,
                typer.Option(
                    "--keep-onnx",
                    help="Keep the intermediate ONNX file alongside the output.",
                ),
            ] = False,
            normalize: NormalizeOpt = False,
            softmax: SoftmaxOpt = False,
            mean: MeanOpt = None,
            std: StdOpt = None,
        ) -> None:
            # ----------------------------------------------------------
            # Validation
            # ----------------------------------------------------------
            if sys.platform != "linux":
                raise ConfigurationError(
                    "RKNN export is only supported on Linux (x86_64, aarch64). "
                    f"Current platform: {sys.platform}."
                )

            if random_calibration:
                raise ConfigurationError(
                    "RKNN does not support --random-calibration. "
                    "Provide real images via --calibration-data <image-directory>."
                )

            if opset < 7 or opset > MAX_OPSET:
                raise ConfigurationError(f"--opset must be between 7 and {MAX_OPSET}.")

            if mode != RknnMode.int8 and (mean is not None or std is not None) and not normalize:
                raise ConfigurationError(
                    "--mean/--std require --normalize unless used for --mode int8 calibration."
                )

            if mode != RknnMode.int8 and (
                calibration_data is not None or calibration_samples is not None
            ):
                raise ConfigurationError(
                    "--calibration-data and --calibration-samples are only valid with --mode int8."
                )

            if mode != RknnMode.int8 and (
                quant_algorithm != RknnQuantAlgorithm.normal
                or quant_method != RknnQuantMethod.channel
            ):
                raise ConfigurationError(
                    "--quant-algorithm and --quant-method are only valid with --mode int8."
                )

            if mode == RknnMode.int8 and calibration_data is None:
                raise ConfigurationError(
                    "int8 quantization requires calibration data.\n\n"
                    "  Provide an image directory: "
                    "--calibration-data <image-directory>\n"
                    "  Or choose a different mode (e.g. fp16, fp32)."
                )

            if calibration_data is not None:
                resolved_cal = calibration_data.expanduser().resolve()
                if not resolved_cal.is_dir():
                    raise ConfigurationError(
                        "RKNN calibration requires an image directory, "
                        "not a file. Provide --calibration-data <image-directory>."
                    )

            if calibration_samples is not None and calibration_samples < 1:
                raise ConfigurationError("--calibration-samples must be >= 1.")

            # ----------------------------------------------------------
            # Model preparation
            # ----------------------------------------------------------
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

            # ----------------------------------------------------------
            # Intermediate ONNX export
            # ----------------------------------------------------------
            onnx_path: Path
            temp_dir: tempfile.TemporaryDirectory[str] | None = None

            if keep_onnx:
                onnx_path = prep.output_path.with_suffix(".onnx")
            else:
                temp_dir = tempfile.TemporaryDirectory()
                onnx_path = Path(temp_dir.name) / "model.onnx"

            try:
                torch.onnx.export(
                    prep.model,
                    (prep.example_input,),
                    f=str(onnx_path),
                    opset_version=opset,
                    input_names=["input"],
                    output_names=["output"],
                )
            except Exception as exc:
                if temp_dir is not None:
                    temp_dir.cleanup()
                raise ExportError(f"Intermediate ONNX export failed: {exc}") from exc

            # ----------------------------------------------------------
            # RKNN conversion
            # ----------------------------------------------------------
            RKNN = _import_rknn()
            rknn = RKNN()
            calibration_txt: Path | None = None
            cal_temp_dir: tempfile.TemporaryDirectory[str] | None = None

            try:
                # --- Normalization config ---
                rknn_config_kwargs: dict[str, object] = {
                    "target_platform": target_platform,
                }

                if normalize:
                    # Normalization embedded in graph; RKNN just converts
                    # uint8 [0,255] -> float [0,1] for the model.
                    channels = prep.resolved_input_size[0]
                    rknn_config_kwargs["mean_values"] = [[0.0] * channels]
                    rknn_config_kwargs["std_values"] = [[255.0] * channels]
                    if mode == RknnMode.int8:
                        console.print(
                            "[bold yellow]Warning:[/bold yellow] Using --normalize with "
                            "--mode int8 may affect calibration accuracy. For best results, "
                            "omit --normalize and let RKNN handle normalization natively."
                        )
                else:
                    # Let RKNN handle normalization via config (scaled to [0,255])
                    resolved_mean, resolved_std = resolve_normalization_stats(
                        prep.model,
                        mean=mean if mode == RknnMode.int8 and mean else None,
                        std=std if mode == RknnMode.int8 and std else None,
                    )
                    rknn_config_kwargs["mean_values"] = [[m * 255.0 for m in resolved_mean]]
                    rknn_config_kwargs["std_values"] = [[s * 255.0 for s in resolved_std]]

                if mode == RknnMode.int8:
                    rknn_config_kwargs["quantized_algorithm"] = str(quant_algorithm)
                    rknn_config_kwargs["quantized_method"] = str(quant_method)

                ret = rknn.config(**rknn_config_kwargs)
                if ret != 0:
                    raise ExportError(f"RKNN config failed (code {ret}).")

                # --- Load ONNX ---
                input_shape = [batch_size, *prep.resolved_input_size]
                ret = rknn.load_onnx(model=str(onnx_path), input_size_list=[input_shape])
                if ret != 0:
                    raise ExportError(f"RKNN failed to load ONNX model (code {ret}).")

                # --- Build ---
                build_kwargs: dict[str, object] = {}
                if mode == RknnMode.int8:
                    build_kwargs["do_quantization"] = True
                    # Prepare calibration text file
                    cal_dir = calibration_data.expanduser().resolve()  # type: ignore[union-attr]
                    image_paths = collect_image_paths(cal_dir)
                    max_samples = calibration_samples or DEFAULT_CALIBRATION_SAMPLES
                    if len(image_paths) > max_samples:
                        rng = random.Random(42)
                        image_paths = rng.sample(image_paths, max_samples)
                    cal_temp_dir = tempfile.TemporaryDirectory()
                    calibration_txt = Path(cal_temp_dir.name) / "calibration.txt"
                    _write_calibration_file(image_paths, calibration_txt)
                    build_kwargs["dataset"] = str(calibration_txt)
                    console.print(
                        f"Using [bold]{len(image_paths)}[/bold] calibration image(s) "
                        f"from [bold]{cal_dir}[/bold]."
                    )
                else:
                    build_kwargs["do_quantization"] = False

                ret = rknn.build(**build_kwargs)
                if ret != 0:
                    raise ExportError(f"RKNN model build failed (code {ret}).")

                # --- Export ---
                ret = rknn.export_rknn(str(prep.output_path))
                if ret != 0:
                    raise ExportError(
                        f"Failed to export RKNN model to {prep.output_path} (code {ret})."
                    )

            except ExportError:
                raise
            except Exception as exc:
                raise ExportError(f"RKNN export failed: {exc}") from exc
            finally:
                rknn.release()
                if cal_temp_dir is not None:
                    cal_temp_dir.cleanup()
                if temp_dir is not None:
                    temp_dir.cleanup()

        return command


def _import_rknn() -> type:
    try:
        from rknn.api import RKNN
    except ImportError as exc:
        raise ExportError(
            "rknn-toolkit2 is required for RKNN export. Install with: pip install rknn-toolkit2"
        ) from exc
    return RKNN


def _write_calibration_file(image_paths: list[Path], output_file: Path) -> None:
    with open(output_file, "w") as f:
        for path in image_paths:
            f.write(f"{path}\n")
