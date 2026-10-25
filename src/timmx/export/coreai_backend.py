from __future__ import annotations

import asyncio
import platform
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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
                    "Neural Engine requires; verification then requests the Neural Engine "
                    "(macOS 27+).",
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

            # Computed first so --half can convert prep.model in place.
            expected = reference_output(prep.model, prep.example_input) if verify else None
            model, example_input = prep.model, prep.example_input
            if half:
                model, example_input = model.half(), example_input.half()
                # create_timm_model's exportable=True turns timm's fused attention off (old ONNX
                # export could not take SDPA). get_decomp_table() keeps SDPA whole for the
                # runtime's own kernel, without which the Neural Engine runs ViT ~10x slower.
                _set_fused_attn(model, True)

            def capture() -> torch.export.ExportedProgram:
                program = capture_program(
                    model, example_input, dynamic_shapes=batch_dynamic_shapes(dynamic_batch)
                )
                try:
                    # get_decomp_table() keeps composite ops (SDPA, instance_norm, pixel_shuffle)
                    # intact so the Core AI runtime can pick its own kernels for them.
                    return program.run_decompositions(coreai_torch.get_decomp_table())
                except Exception as exc:
                    raise ExportError(f"torch.export decomposition failed: {exc}") from exc

            exported_program = capture()
            if half and _has_sdpa_over_4d(exported_program):
                # Apple's Neural Engine compiler aborts the whole process on SDPA over 5-D
                # tensors (twins, hiera), so those models keep timm's unfused attention.
                _set_fused_attn(model, False)
                exported_program = capture()

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
                _verify_asset(prep.output_path, example_input, expected, neural_engine=half)

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


def _set_fused_attn(model: torch.nn.Module, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "fused_attn"):
            module.fused_attn = enabled


def _has_sdpa_over_4d(program: torch.export.ExportedProgram) -> bool:
    return any(
        node.target is torch.ops.aten.scaled_dot_product_attention.default
        and node.args[0].meta["val"].dim() > 4
        for node in program.graph.nodes
    )


def _neural_engine_options() -> object | None:
    """Load options preferring the Neural Engine, or None where the runtime takes no preference.

    The Neural Engine only runs float16 graphs, and the default specialization does not pick it.
    The in-package runtime used before macOS 27 (`is_supported()` is False) fails any load that
    is given options.
    """
    from coreai.runtime import ComputeUnitKind, SpecializationOptions

    if not SpecializationOptions.is_supported():
        return None
    return SpecializationOptions.from_preferred_compute_unit_kind(ComputeUnitKind.neural_engine())


def _verify_asset(
    output_path: Path, example_input: torch.Tensor, expected: torch.Tensor, *, neural_engine: bool
) -> None:
    """Reload the asset and, on macOS, run it against the PyTorch reference.

    With `neural_engine` the runtime is asked to prefer the Neural Engine.

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
        # A worker thread has no running event loop, so asyncio.run() also works when the
        # caller is inside one (Jupyter, async code).
        with ThreadPoolExecutor(max_workers=1) as pool:
            run = _run_asset(output_path, example_input, neural_engine)
            actual = pool.submit(asyncio.run, run).result()
    except Exception as exc:
        raise ExportError(f"Saved Core AI asset failed verification: {exc}") from exc
    verify_outputs(expected, actual, backend="Core AI")


async def _run_asset(output_path: Path, example_input: torch.Tensor, neural_engine: bool) -> object:
    from coreai.runtime import AIModel, NDArray

    options = _neural_engine_options() if neural_engine else None
    if neural_engine:
        console.print(
            "[dim]verify: requesting the Neural Engine[/dim]"
            if options is not None
            else "[dim]verify: this Core AI runtime takes no compute unit preference (needs "
            "macOS 27), running with its default[/dim]"
        )
    model = await AIModel.load(output_path, options)
    function = model.load_function(ENTRYPOINT)
    outputs = await function({"input": NDArray(example_input.detach().cpu())})
    return outputs["output"].numpy()
