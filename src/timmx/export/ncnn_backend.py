from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from timmx.errors import ExportError
from timmx.export.base import DependencyStatus, ExportBackend
from timmx.export.common import (
    BatchSizeOpt,
    CheckpointOpt,
    DeviceOpt,
    InChansOpt,
    InputSizeOpt,
    ModelNameArg,
    NumClassesOpt,
    PretrainedOpt,
    prepare_export,
)
from timmx.export.types import Device


class NcnnBackend(ExportBackend):
    name = "ncnn"
    help = "Export a timm model to ncnn (.param/.bin) via pnnx."

    def check_dependencies(self) -> DependencyStatus:
        try:
            __import__("pnnx")
            return DependencyStatus(available=True, missing_packages=[], install_hint="")
        except ImportError:
            return DependencyStatus(
                available=False,
                missing_packages=["pnnx"],
                install_hint="pip install 'timmx[ncnn]'",
            )

    def create_command(self) -> Callable[..., None]:
        def command(
            model_name: ModelNameArg,
            output: Annotated[
                Path, typer.Option(help="Output directory to write the exported ncnn model files.")
            ],
            checkpoint: CheckpointOpt = None,
            pretrained: PretrainedOpt = False,
            num_classes: NumClassesOpt = None,
            in_chans: InChansOpt = None,
            batch_size: BatchSizeOpt = 1,
            input_size: InputSizeOpt = None,
            device: DeviceOpt = Device.cpu,
            fp16: Annotated[
                bool,
                typer.Option(help="Export weights in fp16 precision."),
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
                output_is_dir=True,
            )
            output_dir = prep.output_path

            try:
                import pnnx
            except ImportError as exc:
                raise ExportError(
                    "pnnx is required for ncnn export. Install with: pip install 'timmx[ncnn]'"
                ) from exc

            ptpath = output_dir / "model.pt"
            pnnxparam = output_dir / "model.pnnx.param"
            pnnxbin = output_dir / "model.pnnx.bin"
            pnnxpy = output_dir / "model_pnnx.py"
            pnnxonnx = output_dir / "model.pnnx.onnx"
            ncnnparam = output_dir / "model.ncnn.param"
            ncnnbin = output_dir / "model.ncnn.bin"
            ncnnpy = output_dir / "model_ncnn.py"

            try:
                pnnx.export(
                    prep.model,
                    inputs=prep.example_input,
                    ptpath=str(ptpath),
                    pnnxparam=str(pnnxparam),
                    pnnxbin=str(pnnxbin),
                    pnnxpy=str(pnnxpy),
                    pnnxonnx=str(pnnxonnx),
                    ncnnparam=str(ncnnparam),
                    ncnnbin=str(ncnnbin),
                    ncnnpy=str(ncnnpy),
                    fp16=fp16,
                    device=str(device),
                )
            except Exception as exc:
                raise ExportError(f"ncnn export via pnnx failed: {exc}") from exc

            # Remove pnnx intermediate files and the torchscript checkpoint
            for path in (ptpath, pnnxparam, pnnxbin, pnnxpy, pnnxonnx):
                path.unlink(missing_ok=True)

            # Remove __pycache__ that pnnx writes alongside the Python wrappers
            pycache = output_dir / "__pycache__"
            if pycache.exists():
                shutil.rmtree(pycache, ignore_errors=True)

        return command
