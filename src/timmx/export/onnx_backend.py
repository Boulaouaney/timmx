from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import timm
import torch
from timm.data import resolve_data_config

from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import ExportBackend

DEFAULT_INPUT_SIZE = (3, 224, 224)
DEFAULT_OPSET = 18


class OnnxBackend(ExportBackend):
    name = "onnx"
    help = "Export a timm model to ONNX."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("model_name", help="timm model name, e.g. resnet18")
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Path to write the ONNX model.",
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
            "--opset",
            type=int,
            default=DEFAULT_OPSET,
            help="ONNX opset version to target.",
        )
        parser.add_argument(
            "--dynamic-batch",
            action="store_true",
            help="Mark batch axis as dynamic.",
        )
        parser.add_argument(
            "--device",
            choices=("cpu", "cuda"),
            default="cpu",
            help="Device used for model instantiation and tracing.",
        )
        parser.add_argument(
            "--external-data",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Save large model weights in external data files.",
        )
        parser.add_argument(
            "--check",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Run ONNX checker after export.",
        )
        parser.add_argument(
            "--exportable",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use timm export-friendly layer variants when available.",
        )

    def run(self, args: argparse.Namespace) -> int:
        self._validate_args(args)

        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model = self._create_model(args)
        input_size = _resolve_input_size(model, args.input_size)

        device = torch.device(args.device)
        model = model.to(device)
        model.eval()

        example_input = torch.randn(args.batch_size, *input_size, device=device)

        export_kwargs: dict[str, object] = {
            "f": str(output_path),
            "opset_version": args.opset,
            "input_names": ["input"],
            "output_names": ["output"],
            "dynamo": True,
            "fallback": True,
            "external_data": args.external_data,
        }
        if args.dynamic_batch:
            export_kwargs["dynamic_shapes"] = ({0: torch.export.Dim("batch")},)
            export_kwargs["dynamic_axes"] = {"input": {0: "batch"}, "output": {0: "batch"}}

        try:
            torch.onnx.export(model, (example_input,), **export_kwargs)
        except Exception as exc:
            raise ExportError(f"ONNX export failed: {exc}") from exc

        if args.check:
            try:
                onnx.checker.check_model(str(output_path))
            except Exception as exc:
                raise ExportError(f"Exported model failed ONNX check: {exc}") from exc

        return 0

    def _create_model(self, args: argparse.Namespace) -> torch.nn.Module:
        create_kwargs: dict[str, object] = {
            "pretrained": args.pretrained,
            "exportable": args.exportable,
        }

        if args.checkpoint is not None:
            checkpoint = Path(args.checkpoint).expanduser()
            if not checkpoint.is_file():
                raise ConfigurationError(f"Checkpoint file does not exist: {checkpoint}")
            create_kwargs["checkpoint_path"] = str(checkpoint)

        if args.num_classes is not None:
            create_kwargs["num_classes"] = args.num_classes

        if args.in_chans is not None:
            create_kwargs["in_chans"] = args.in_chans

        try:
            return timm.create_model(args.model_name, **create_kwargs)
        except Exception as exc:
            raise ExportError(f"Failed to create timm model {args.model_name!r}: {exc}") from exc

    def _validate_args(self, args: argparse.Namespace) -> None:
        if args.batch_size < 1:
            raise ConfigurationError("--batch-size must be >= 1.")
        if args.opset < 7:
            raise ConfigurationError("--opset must be >= 7.")
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ConfigurationError("--device cuda was requested but CUDA is unavailable.")


def _resolve_input_size(
    model: torch.nn.Module, requested: list[int] | None
) -> tuple[int, int, int]:
    if requested is not None:
        return int(requested[0]), int(requested[1]), int(requested[2])

    config = resolve_data_config({}, model=model)
    raw_input_size = config.get("input_size")
    if isinstance(raw_input_size, tuple) and len(raw_input_size) == 3:
        return int(raw_input_size[0]), int(raw_input_size[1]), int(raw_input_size[2])
    if isinstance(raw_input_size, list) and len(raw_input_size) == 3:
        return int(raw_input_size[0]), int(raw_input_size[1]), int(raw_input_size[2])

    return DEFAULT_INPUT_SIZE
