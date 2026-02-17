from __future__ import annotations

import argparse
from pathlib import Path

import torch

from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import ExportBackend
from timmx.export.common import create_timm_model, resolve_input_size, validate_common_args


class TorchExportBackend(ExportBackend):
    name = "torch-export"
    help = "Export a timm model with torch.export (.pt2)."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("model_name", help="timm model name, e.g. resnet18")
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Path to write the exported program archive (typically .pt2).",
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
            help="Mark batch axis as dynamic.",
        )
        parser.add_argument(
            "--device",
            choices=("cpu", "cuda"),
            default="cpu",
            help="Device used for model instantiation and tracing.",
        )
        parser.add_argument(
            "--strict",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Enable strict graph capture during torch.export.",
        )
        parser.add_argument(
            "--verify",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Load the saved .pt2 archive after export to validate it.",
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
        dynamic_shapes: tuple[dict[int, torch.export.Dim], ...] | None = None
        if args.dynamic_batch:
            dynamic_shapes = ({0: torch.export.Dim("batch")},)

        try:
            exported_program = torch.export.export(
                model,
                (example_input,),
                dynamic_shapes=dynamic_shapes,
                strict=args.strict,
            )
        except Exception as exc:
            raise ExportError(f"torch.export capture failed: {exc}") from exc

        try:
            torch.export.save(exported_program, str(output_path))
        except Exception as exc:
            raise ExportError(f"Failed to save torch.export archive: {exc}") from exc

        if args.verify:
            try:
                torch.export.load(str(output_path))
            except Exception as exc:
                raise ExportError(f"Saved torch.export archive failed to load: {exc}") from exc

        return 0

    def _validate_args(self, args: argparse.Namespace) -> None:
        validate_common_args(batch_size=args.batch_size, device=args.device)
        if args.dynamic_batch and args.batch_size < 2:
            raise ConfigurationError(
                "--dynamic-batch requires --batch-size >= 2 for stable symbolic shape capture."
            )
