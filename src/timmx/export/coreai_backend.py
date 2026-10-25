from __future__ import annotations

import asyncio
import copy
import platform
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import torch
import typer

from timmx.console import console
from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import DependencyStatus, ExportBackend
from timmx.export.common import (
    OUTPUT_DEFAULT_HELP,
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
    batch_dynamic_shapes,
    capture_program,
    prepare_export,
    reference_output,
    verify_outputs,
)
from timmx.export.types import Device

# coreai-torch names the converted graph "main" unless told otherwise; the runtime looks it up
# by that name.
ENTRYPOINT = "main"


class CoreAIBackend(ExportBackend):
    name = "coreai"
    help = "Export a timm model to a Core AI asset (.aimodel)."

    def check_dependencies(self) -> DependencyStatus:
        try:
            import coreai_torch  # noqa: F401
        except ImportError:
            return DependencyStatus(
                available=False,
                missing_packages=["coreai-torch"],
                install_hint="pip install 'timmx[coreai]'",
            )
        return DependencyStatus(available=True, missing_packages=[], install_hint="")

    def create_command(self) -> Callable[..., Path]:
        def command(
            model_name: ModelNameArg,
            output: Annotated[
                Path | None,
                typer.Option(
                    help=f"Path of the .aimodel asset directory to write ({OUTPUT_DEFAULT_HELP})."
                ),
            ] = None,
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
            half: Annotated[
                bool,
                typer.Option(
                    "--half",
                    help="Convert the model, its input and its output to float16, which the "
                    "Neural Engine requires; verification then runs on the Neural Engine.",
                ),
            ] = False,
            verify: Annotated[
                bool,
                typer.Option(help="Reload the saved asset and compare its output with PyTorch."),
            ] = True,
            normalize: NormalizeOpt = False,
            softmax: SoftmaxOpt = False,
            mean: MeanOpt = None,
            std: StdOpt = None,
        ) -> Path:
            if output is not None and output.suffix != ".aimodel":
                raise ConfigurationError("--output must be an .aimodel path for Core AI.")
            if dynamic_batch and batch_size < 2:
                raise ConfigurationError(
                    "--dynamic-batch requires --batch-size >= 2 for stable symbolic shape capture."
                )

            coreai_torch = _import_coreai_torch()

            prep = prepare_export(
                model_name=model_name,
                output=output,
                default_suffix=".aimodel",
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

            export_model, example_input = prep.model, prep.example_input
            if half:
                export_model = copy.deepcopy(prep.model).half()
                example_input = prep.example_input.half()
                # create_timm_model's exportable=True turns timm's fused attention off (old ONNX
                # export could not take SDPA). get_decomp_table() keeps SDPA whole for the
                # runtime's own kernel, without which the Neural Engine runs ViT ~10x slower.
                for module in export_model.modules():
                    if hasattr(module, "fused_attn"):
                        module.fused_attn = True

            exported_program = capture_program(
                export_model, example_input, dynamic_shapes=batch_dynamic_shapes(dynamic_batch)
            )
            try:
                # get_decomp_table() keeps composite ops (SDPA, instance_norm, pixel_shuffle)
                # intact so the Core AI runtime can pick its own kernels for them.
                exported_program = exported_program.run_decompositions(
                    coreai_torch.get_decomp_table()
                )
            except Exception as exc:
                raise ExportError(f"torch.export decomposition failed: {exc}") from exc

            try:
                program = (
                    coreai_torch.TorchConverter()
                    .add_exported_program(
                        exported_program,
                        input_names=["input"],
                        output_names=["output"],
                        entrypoint_name=ENTRYPOINT,
                    )
                    .to_coreai()
                )
            except Exception as exc:
                raise ExportError(f"Core AI conversion failed: {exc}") from exc

            try:
                program.save_asset(prep.output_path)
            except Exception as exc:
                raise ExportError(f"Failed to save Core AI asset: {exc}") from exc

            if verify:
                _verify_asset(
                    prep.output_path,
                    example_input,
                    reference_output(prep.model, prep.example_input),
                    neural_engine=half,
                )

            return prep.output_path

        return command


def _import_coreai_torch() -> object:
    try:
        import coreai_torch
    except ImportError as exc:
        raise ExportError(
            "coreai-torch is required for Core AI export. Install with: pip install 'timmx[coreai]'"
        ) from exc
    return coreai_torch


def _verify_asset(
    output_path: Path, example_input: torch.Tensor, expected: torch.Tensor, *, neural_engine: bool
) -> None:
    """Reload the asset and, on macOS, run it against the PyTorch reference.

    With `neural_engine` the runtime is asked to prefer the Neural Engine, which only takes
    float16 graphs and which the default specialization does not prefer.

    The Core AI runtime only executes on Apple platforms (loading an asset on Linux fails with
    an opaque "No such file or directory"), so elsewhere we can only check that what we wrote
    reads back as a valid asset.
    """
    try:
        from coreai.authoring import AIModelAsset

        if platform.system() != "Darwin":
            AIModelAsset.load(output_path)
            console.print("[dim]verify: asset only (Core AI inference needs macOS)[/dim]")
            return
        actual = asyncio.run(_run_asset(output_path, example_input, neural_engine))
    except Exception as exc:
        raise ExportError(f"Saved Core AI asset failed verification: {exc}") from exc
    verify_outputs(expected, actual, backend="Core AI")


async def _run_asset(output_path: Path, example_input: torch.Tensor, neural_engine: bool) -> object:
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

    options = None
    if neural_engine:
        # Apple's Neural Engine compiler aborts the process on some attention layouts (twins,
        # hiera); say what is running so the abort is not a mystery.
        console.print("[dim]verify: specializing for the Neural Engine[/dim]")
        options = SpecializationOptions.from_preferred_compute_unit_kind(
            ComputeUnitKind.neural_engine()
        )
    model = await AIModel.load(output_path, options)
    function = model.load_function(ENTRYPOINT)
    outputs = await function({"input": NDArray(example_input.detach().cpu())})
    return outputs["output"].numpy()
