from __future__ import annotations

import platform
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import numpy as np
import torch
import typer
from PIL import Image

from timmx.console import console
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
    OutputOpt,
    PretrainedOpt,
    SoftmaxOpt,
    StdOpt,
    prepare_export,
    reference_output,
    verify_outputs,
)
from timmx.export.types import Device


class ExportSource(StrEnum):
    trace = "trace"
    torch_export = "torch-export"


class ConvertTo(StrEnum):
    mlprogram = "mlprogram"
    neuralnetwork = "neuralnetwork"


class ComputePrecision(StrEnum):
    float16 = "float16"
    float32 = "float32"


class CoreMLBackend(ExportBackend):
    name = "coreml"
    help = "Export a timm model to Core ML."

    def check_dependencies(self) -> DependencyStatus:
        missing = []
        try:
            import coremltools  # noqa: F401
        except ImportError:
            missing.append("coremltools")
        return DependencyStatus(
            available=not missing,
            missing_packages=missing,
            install_hint="pip install 'timmx[coreml]'",
        )

    def create_command(self) -> Callable[..., Path]:
        def command(
            model_name: ModelNameArg,
            output: OutputOpt = None,
            checkpoint: CheckpointOpt = None,
            pretrained: PretrainedOpt = False,
            num_classes: NumClassesOpt = None,
            in_chans: InChansOpt = None,
            batch_size: BatchSizeOpt = 1,
            input_size: InputSizeOpt = None,
            dynamic_batch: Annotated[
                bool,
                typer.Option(
                    "--dynamic-batch", help="Export with flexible Core ML batch dimension."
                ),
            ] = False,
            batch_upper_bound: Annotated[
                int,
                typer.Option(
                    help="Upper bound used for flexible batch when --dynamic-batch is enabled."
                ),
            ] = 8,
            device: DeviceOpt = Device.cpu,
            source: Annotated[
                ExportSource,
                typer.Option(
                    help="Model capture method: torch-export (default) or trace (torch.jit.trace)."
                ),
            ] = ExportSource.torch_export,
            convert_to: Annotated[
                ConvertTo, typer.Option(help="Core ML model type to generate.")
            ] = ConvertTo.mlprogram,
            compute_precision: Annotated[
                ComputePrecision | None,
                typer.Option(help="Precision for mlprogram conversion."),
            ] = None,
            half: Annotated[
                bool,
                typer.Option("--half", help="Quantize weights to float16."),
            ] = False,
            int8: Annotated[
                bool,
                typer.Option("--int8", help="Quantize weights to 8-bit (linear, per-channel)."),
            ] = False,
            int4: Annotated[
                bool,
                typer.Option(
                    "--int4", help="Palettize weights to 4-bit (k-means, mlpackage only)."
                ),
            ] = False,
            normalize: NormalizeOpt = False,
            softmax: SoftmaxOpt = False,
            mean: MeanOpt = None,
            std: StdOpt = None,
            image_input: Annotated[
                bool,
                typer.Option(
                    "--image-input",
                    help="Take an image (CVPixelBuffer, pixels scaled by 1/255) instead of a "
                    "float tensor so Vision and the Xcode preview can feed the model; requires "
                    "--normalize and --batch-size 1.",
                ),
            ] = False,
            class_labels: Annotated[
                Path | None,
                typer.Option(
                    help="Text file with one class label per line; makes the model a Core ML "
                    "classifier (outputs classLabel and classLabel_probs)."
                ),
            ] = None,
            verify: Annotated[
                bool,
                typer.Option(help="Reload the saved model and compare its output with PyTorch."),
            ] = True,
        ) -> Path:
            expected_suffix = ".mlpackage" if convert_to == ConvertTo.mlprogram else ".mlmodel"
            if output is not None and output.suffix != expected_suffix:
                raise ConfigurationError(
                    f"--output must be a {expected_suffix} path for --convert-to {convert_to}."
                )
            if image_input and not normalize:
                raise ConfigurationError(
                    "--image-input requires --normalize (the model receives [0, 1] pixels)."
                )
            if image_input and (batch_size != 1 or dynamic_batch):
                raise ConfigurationError(
                    "--image-input requires --batch-size 1 without --dynamic-batch."
                )
            labels = _read_class_labels(class_labels) if class_labels is not None else None
            if convert_to == ConvertTo.neuralnetwork and compute_precision is not None:
                raise ConfigurationError(
                    "--compute-precision is only supported when --convert-to mlprogram."
                )
            if dynamic_batch:
                if batch_upper_bound < 1:
                    raise ConfigurationError("--batch-upper-bound must be >= 1.")
                if batch_upper_bound < batch_size:
                    raise ConfigurationError("--batch-upper-bound must be >= --batch-size.")
            if source == ExportSource.torch_export and dynamic_batch and batch_size < 2:
                raise ConfigurationError(
                    "--dynamic-batch with --source torch-export requires "
                    "--batch-size >= 2 for stable symbolic shape capture."
                )
            if sum([half, int8, int4]) > 1:
                raise ConfigurationError(
                    "Only one of --half, --int8, --int4 can be specified at a time."
                )
            if int4 and convert_to != ConvertTo.mlprogram:
                raise ConfigurationError("--int4 is only supported with --convert-to mlprogram.")

            ct = _import_coremltools()

            prep = prepare_export(
                model_name=model_name,
                output=output,
                default_suffix=expected_suffix,
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

            if labels is not None:
                num_outputs = reference_output(prep.model, prep.example_input).shape[-1]
                if len(labels) != num_outputs:
                    raise ConfigurationError(
                        f"--class-labels has {len(labels)} labels but the model has "
                        f"{num_outputs} outputs."
                    )

            convert_kwargs: dict[str, object] = {"convert_to": str(convert_to)}
            if compute_precision is not None:
                convert_kwargs["compute_precision"] = _map_compute_precision(
                    str(compute_precision), ct
                )
            if labels is not None:
                convert_kwargs["classifier_config"] = ct.ClassifierConfig(labels)
            if image_input:
                convert_kwargs["inputs"] = [_image_input_type(prep.resolved_input_size, ct)]

            if source == ExportSource.torch_export:
                dynamic_shapes: tuple[dict[int, torch.export.Dim], ...] | None = None
                if dynamic_batch:
                    dynamic_shapes = ({0: torch.export.Dim("batch", min=1, max=batch_upper_bound)},)

                try:
                    exported_program = torch.export.export(
                        prep.model,
                        (prep.example_input,),
                        dynamic_shapes=dynamic_shapes,
                    )
                    # coremltools requires ATEN dialect, not TRAINING
                    exported_program = exported_program.run_decompositions({})
                except Exception as exc:
                    raise ExportError(f"torch.export capture failed: {exc}") from exc

                try:
                    coreml_model = ct.convert(exported_program, **convert_kwargs)
                except Exception as exc:
                    raise ExportError(
                        f"Core ML conversion failed: {exc} (try --source trace)"
                    ) from exc
            else:
                with torch.no_grad():
                    try:
                        traced_model = torch.jit.trace(prep.model, prep.example_input)
                    except Exception as exc:
                        raise ExportError(f"TorchScript trace failed: {exc}") from exc

                convert_kwargs["source"] = "pytorch"
                if not image_input:
                    convert_kwargs["inputs"] = [
                        ct.TensorType(
                            name="input",
                            shape=_build_input_shape(
                                batch_size=batch_size,
                                dynamic_batch=dynamic_batch,
                                batch_upper_bound=batch_upper_bound,
                                input_size=prep.resolved_input_size,
                                ct=ct,
                            ),
                        )
                    ]

                try:
                    coreml_model = ct.convert(traced_model, **convert_kwargs)
                except Exception as exc:
                    raise ExportError(
                        f"Core ML conversion failed: {exc} (try --source torch-export)"
                    ) from exc

            try:
                coreml_model = _name_io(coreml_model, ct, rename_output=labels is None)
            except Exception as exc:
                raise ExportError(f"Failed to name Core ML inputs/outputs: {exc}") from exc

            bits = 4 if int4 else 8 if int8 else 16 if half else 32
            if bits < 32:
                is_mlprogram = convert_to == ConvertTo.mlprogram
                try:
                    coreml_model = _quantize_weights(
                        coreml_model, bits=bits, is_mlprogram=is_mlprogram, ct=ct
                    )
                except Exception as exc:
                    raise ExportError(f"Weight quantization failed: {exc}") from exc

            try:
                coreml_model.save(str(prep.output_path))
            except Exception as exc:
                raise ExportError(f"Failed to save Core ML model: {exc}") from exc

            if verify:
                _verify_coreml_model(
                    prep.output_path,
                    prep.model,
                    prep.example_input,
                    ct=ct,
                    image_input=image_input,
                    labels=labels,
                )

            return prep.output_path

        return command


def _name_io(coreml_model: object, ct: object, *, rename_output: bool = True) -> object:
    """Name the single input/output "input"/"output" regardless of the capture source.

    Classifiers keep coremltools' `classLabel`/`classLabel_probs` outputs (*rename_output=False*).
    """
    spec = coreml_model.get_spec()
    renames = {spec.description.input[0].name: "input"}
    if rename_output:
        renames[spec.description.output[0].name] = "output"
    for old, new in renames.items():
        if old != new:
            ct.utils.rename_feature(
                spec, old, new, rename_inputs=new == "input", rename_outputs=new == "output"
            )
    if spec.WhichOneof("Type") == "neuralNetwork":
        # rename_feature updates the interface but not the layer blobs of a neuralnetwork spec.
        for layer in spec.neuralNetwork.layers:
            for blobs in (layer.input, layer.output):
                for index, name in enumerate(blobs):
                    if name in renames:
                        blobs[index] = renames[name]
    return ct.models.MLModel(spec, weights_dir=coreml_model.weights_dir, skip_model_load=True)


def _verify_coreml_model(
    output_path: Path,
    model: torch.nn.Module,
    example_input: torch.Tensor,
    *,
    ct: object,
    image_input: bool = False,
    labels: list[str] | None = None,
) -> None:
    feed: object = example_input.cpu().numpy()
    if image_input:
        # Feed the same pixels to both: an 8-bit image to Core ML (scaled by 1/255 inside the
        # model) and its [0, 1] float version to PyTorch.
        pixels = (torch.rand_like(example_input) * 255).round()
        example_input = pixels / 255
        feed = _to_pil_image(pixels)
    expected = reference_output(model, example_input)
    try:
        if platform.system() != "Darwin":
            ct.models.MLModel(str(output_path), skip_model_load=True)
            console.print("[dim]verify: metadata only (Core ML inference needs macOS)[/dim]")
            return
        loaded = ct.models.MLModel(str(output_path))
        prediction = loaded.predict({"input": feed})
        if labels is None:
            actual = prediction["output"]
        else:
            probabilities = prediction["classLabel_probs"]
            actual = np.array([[probabilities[label] for label in labels]])
    except Exception as exc:
        raise ExportError(f"Saved Core ML model failed verification: {exc}") from exc
    verify_outputs(expected, actual, backend="Core ML")


def _to_pil_image(pixels: torch.Tensor) -> Image.Image:
    """(1, C, H, W) float tensor holding 0-255 values → PIL image (L for C=1, RGB for C=3)."""
    array = pixels[0].cpu().numpy().astype(np.uint8)
    if array.shape[0] == 1:
        return Image.fromarray(array[0], mode="L")
    return Image.fromarray(array.transpose(1, 2, 0), mode="RGB")


def _image_input_type(input_size: tuple[int, int, int], ct: object) -> object:
    """ImageType feeding 8-bit pixels scaled to [0, 1]; --normalize then applies mean/std."""
    layout = ct.colorlayout.GRAYSCALE if input_size[0] == 1 else ct.colorlayout.RGB
    return ct.ImageType(name="input", shape=(1, *input_size), scale=1 / 255, color_layout=layout)


def _read_class_labels(path: Path) -> list[str]:
    try:
        labels = [line.strip() for line in Path(path).expanduser().read_text().splitlines()]
    except OSError as exc:
        raise ConfigurationError(f"Cannot read --class-labels file {path}: {exc}") from exc
    labels = [label for label in labels if label]
    if not labels:
        raise ConfigurationError(f"--class-labels file {path} has no labels.")
    return labels


def _build_input_shape(
    *,
    batch_size: int,
    dynamic_batch: bool,
    batch_upper_bound: int,
    input_size: tuple[int, int, int],
    ct: object,
) -> tuple[object, int, int, int] | tuple[int, int, int, int]:
    if not dynamic_batch:
        return (batch_size, *input_size)

    batch_dim = ct.RangeDim(
        lower_bound=1,
        upper_bound=batch_upper_bound,
        default=batch_size,
        symbol="batch",
    )
    return (batch_dim, *input_size)


def _map_compute_precision(value: str, ct: object) -> object:
    if value == "float16":
        return ct.precision.FLOAT16
    return ct.precision.FLOAT32


def _quantize_weights(coreml_model: object, *, bits: int, is_mlprogram: bool, ct: object) -> object:
    if not is_mlprogram:
        mode = "linear" if bits == 16 else "linear_symmetric"
        console.print(f"[bold]Quantizing weights to {bits}-bit ({mode})...[/bold]")
        return ct.models.neural_network.quantization_utils.quantize_weights(
            coreml_model, bits, mode
        )

    if bits == 16:
        console.print(
            "[bold yellow]note:[/bold yellow] mlprogram models already use float16 weights"
            " — skipping --half.",
            highlight=False,
        )
        return coreml_model

    import coremltools.optimize.coreml as cto

    if bits == 8:
        # mlprogram int8: per-channel symmetric linear quantization
        op_config = cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=512
        )
        console.print("[bold]Quantizing weights to 8-bit (linear symmetric, per-channel)...[/bold]")
        return cto.linear_quantize_weights(
            coreml_model, cto.OptimizationConfig(global_config=op_config)
        )

    # mlprogram int4: k-means palettization (4-bit LUT); needs scikit-learn
    op_config = cto.OpPalettizerConfig(mode="kmeans", nbits=bits, weight_threshold=512)
    console.print(f"[bold]Palettizing weights to {bits}-bit (k-means)...[/bold]")
    return cto.palettize_weights(coreml_model, cto.OptimizationConfig(global_config=op_config))


def _import_coremltools() -> object:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise ExportError(
            "coremltools is required for Core ML export. Install with: pip install 'timmx[coreml]'"
        ) from exc
    return ct
