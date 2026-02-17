from __future__ import annotations

import argparse
from pathlib import Path

import torch

from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import ExportBackend
from timmx.export.common import create_timm_model, resolve_input_size, validate_common_args


class CoreMLBackend(ExportBackend):
    name = "coreml"
    help = "Export a timm model to Core ML."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("model_name", help="timm model name, e.g. resnet18")
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Path to write the Core ML model (.mlpackage or .mlmodel).",
        )
        parser.add_argument(
            "--checkpoint",
            type=Path,
            help="Path to a fine-tuned checkpoint to load into the model.",
        )
        parser.add_argument(
            "--pretrained",
            action="store_true",
            help="Load timm pretrained weights.",
        )
        parser.add_argument(
            "--num-classes",
            type=int,
            help="Override the model classifier output classes.",
        )
        parser.add_argument(
            "--in-chans",
            type=int,
            help="Override model input channels.",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="Example input batch size for export.",
        )
        parser.add_argument(
            "--input-size",
            type=int,
            nargs=3,
            metavar=("C", "H", "W"),
            help="Explicit input shape as channels height width.",
        )
        parser.add_argument(
            "--dynamic-batch",
            action="store_true",
            help="Export with flexible Core ML batch dimension.",
        )
        parser.add_argument(
            "--batch-upper-bound",
            type=int,
            default=8,
            help="Upper bound used for flexible batch when --dynamic-batch is enabled.",
        )
        parser.add_argument(
            "--device",
            choices=("cpu", "cuda"),
            default="cpu",
            help="Device used for model instantiation and tracing.",
        )
        parser.add_argument(
            "--convert-to",
            choices=("mlprogram", "neuralnetwork"),
            default="mlprogram",
            help="Core ML model type to generate.",
        )
        parser.add_argument(
            "--compute-precision",
            choices=("float16", "float32"),
            help="Precision for mlprogram conversion.",
        )
        parser.add_argument(
            "--verify",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Reload the saved Core ML model metadata after export.",
        )
        parser.add_argument(
            "--exportable",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use timm export-friendly layer variants when available.",
        )

    def run(self, args: argparse.Namespace) -> int:
        self._validate_args(args)
        ct = _import_coremltools()

        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = create_timm_model(
            args.model_name,
            pretrained=args.pretrained,
            checkpoint=args.checkpoint,
            num_classes=args.num_classes,
            in_chans=args.in_chans,
            exportable=args.exportable,
        )
        input_size = resolve_input_size(model, args.input_size)

        device = torch.device(args.device)
        model = model.to(device)
        model.eval()

        example_input = torch.randn(args.batch_size, *input_size, device=device)
        with torch.no_grad():
            try:
                traced_model = torch.jit.trace(model, example_input)
            except Exception as exc:
                raise ExportError(f"TorchScript trace failed: {exc}") from exc

        input_type = ct.TensorType(
            name="input", shape=self._build_input_shape(args, input_size, ct)
        )
        convert_kwargs: dict[str, object] = {
            "source": "pytorch",
            "inputs": [input_type],
            "convert_to": args.convert_to,
        }
        if args.compute_precision is not None:
            convert_kwargs["compute_precision"] = _map_compute_precision(args.compute_precision, ct)

        try:
            coreml_model = ct.convert(traced_model, **convert_kwargs)
        except Exception as exc:
            raise ExportError(f"Core ML conversion failed: {exc}") from exc

        try:
            coreml_model.save(str(output_path))
        except Exception as exc:
            raise ExportError(f"Failed to save Core ML model: {exc}") from exc

        if args.verify:
            try:
                ct.models.MLModel(str(output_path), skip_model_load=True)
            except Exception as exc:
                raise ExportError(f"Saved Core ML model failed verification: {exc}") from exc

        return 0

    def _validate_args(self, args: argparse.Namespace) -> None:
        validate_common_args(batch_size=args.batch_size, device=args.device)
        if args.dynamic_batch:
            if args.batch_upper_bound < 1:
                raise ConfigurationError("--batch-upper-bound must be >= 1.")
            if args.batch_upper_bound < args.batch_size:
                raise ConfigurationError("--batch-upper-bound must be >= --batch-size.")
        if args.convert_to == "neuralnetwork" and args.compute_precision is not None:
            raise ConfigurationError(
                "--compute-precision is only supported when --convert-to mlprogram."
            )

    def _build_input_shape(
        self, args: argparse.Namespace, input_size: tuple[int, int, int], ct: object
    ) -> tuple[object, int, int, int] | tuple[int, int, int, int]:
        if not args.dynamic_batch:
            return (args.batch_size, *input_size)

        batch_dim = ct.RangeDim(
            lower_bound=1,
            upper_bound=args.batch_upper_bound,
            default=args.batch_size,
            symbol="batch",
        )
        return (batch_dim, *input_size)


def _map_compute_precision(value: str, ct: object) -> object:
    if value == "float16":
        return ct.precision.FLOAT16
    return ct.precision.FLOAT32


def _import_coremltools() -> object:
    try:
        import coremltools as ct
    except ImportError as exc:
        raise ExportError(
            "coremltools is required for Core ML export. Install dependencies with `uv sync`."
        ) from exc
    return ct
