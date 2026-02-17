from __future__ import annotations

import argparse
from pathlib import Path

import torch

from timmx.errors import ConfigurationError, ExportError
from timmx.export.base import ExportBackend
from timmx.export.common import create_timm_model, resolve_input_size, validate_common_args


class LiteRTBackend(ExportBackend):
    name = "litert"
    help = "Export a timm model to LiteRT/TFLite using litert-torch."

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("model_name", help="timm model name, e.g. resnet18")
        parser.add_argument(
            "--output",
            type=Path,
            required=True,
            help="Path to write the LiteRT model (.tflite).",
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
            help="Example input batch size for export and calibration.",
        )
        parser.add_argument(
            "--input-size",
            type=int,
            nargs=3,
            metavar=("C", "H", "W"),
            help="Explicit input shape as channels height width.",
        )
        parser.add_argument(
            "--device",
            choices=("cpu", "cuda"),
            default="cpu",
            help="Device used for model instantiation and tracing.",
        )
        parser.add_argument(
            "--mode",
            choices=("fp32", "fp16", "dynamic-int8", "int8"),
            default="fp32",
            help="Export precision / quantization mode.",
        )
        parser.add_argument(
            "--nhwc-output",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Transpose the first model output from NCHW to NHWC.",
        )
        parser.add_argument(
            "--verify",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Load and allocate the exported TFLite model.",
        )
        parser.add_argument(
            "--exportable",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Use timm export-friendly layer variants when available.",
        )

    def run(self, args: argparse.Namespace) -> int:
        self._validate_args(args)

        litert_torch = _import_litert_torch()

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

        convert_module: torch.nn.Module = model
        quant_config = None
        converter_flags: dict[str, object] = {}

        if args.mode == "fp16":
            tensorflow = _import_tensorflow()
            converter_flags = {
                "optimizations": [tensorflow.lite.Optimize.DEFAULT],
                "target_spec": {"supported_types": [tensorflow.float16]},
            }
        elif args.mode in {"dynamic-int8", "int8"}:
            convert_module, quant_config = _prepare_pt2e_quantized_module(
                model,
                example_input,
                is_dynamic=(args.mode == "dynamic-int8"),
            )

        if args.nhwc_output:
            _validate_nhwc_output_compatibility(convert_module, example_input)
            convert_module = litert_torch.to_channel_last_io(convert_module, outputs=[0])

        try:
            edge_model = litert_torch.convert(
                convert_module,
                (example_input,),
                quant_config=quant_config,
                _ai_edge_converter_flags=converter_flags,
            )
        except Exception as exc:
            raise ExportError(f"LiteRT conversion failed: {exc}") from exc

        try:
            edge_model.export(str(output_path))
        except Exception as exc:
            raise ExportError(f"Failed to save LiteRT model: {exc}") from exc

        if args.verify:
            _verify_tflite_model(output_path)

        return 0

    def _validate_args(self, args: argparse.Namespace) -> None:
        validate_common_args(batch_size=args.batch_size, device=args.device)
        if args.mode in {"dynamic-int8", "int8"} and args.device != "cpu":
            raise ConfigurationError(
                "LiteRT int8 modes currently require --device cpu for PT2E quantization."
            )


def _prepare_pt2e_quantized_module(
    model: torch.nn.Module, example_input: torch.Tensor, *, is_dynamic: bool
) -> tuple[torch.nn.Module, object]:
    from litert_torch.quantize import pt2e_quantizer, quant_config
    from torchao.quantization.pt2e import quantize_pt2e

    try:
        exported_module = torch.export.export(model, (example_input,), strict=False).module()

        quantizer = pt2e_quantizer.PT2EQuantizer().set_global(
            pt2e_quantizer.get_symmetric_quantization_config(
                is_per_channel=False,
                is_dynamic=is_dynamic,
            )
        )
        prepared_module = quantize_pt2e.prepare_pt2e(exported_module, quantizer)

        with torch.no_grad():
            prepared_module(example_input)

        quantized_module = quantize_pt2e.convert_pt2e(prepared_module, fold_quantize=False)
    except Exception as exc:
        mode_label = "dynamic-int8" if is_dynamic else "int8"
        raise ExportError(f"Failed to prepare {mode_label} PT2E quantized model: {exc}") from exc

    return quantized_module, quant_config.QuantConfig(pt2e_quantizer=quantizer)


def _validate_nhwc_output_compatibility(
    model: torch.nn.Module,
    example_input: torch.Tensor,
) -> None:
    with torch.no_grad():
        outputs = model(example_input)

    if isinstance(outputs, (tuple, list)):
        if not outputs:
            raise ConfigurationError("--nhwc-output requested but model returned no outputs.")
        output = outputs[0]
    else:
        output = outputs

    if not torch.is_tensor(output):
        raise ConfigurationError("--nhwc-output currently requires tensor outputs.")
    if output.ndim < 3:
        raise ConfigurationError(
            "--nhwc-output requires output rank >= 3 (for NCHW -> NHWC transposition)."
        )


def _verify_tflite_model(output_path: Path) -> None:
    try:
        from ai_edge_litert import interpreter as tfl_interpreter
    except ImportError as exc:
        raise ExportError(
            "ai-edge-litert is required to verify LiteRT export. Install dependencies with `uv sync`."
        ) from exc

    try:
        interpreter = tfl_interpreter.Interpreter(model_path=str(output_path))
        interpreter.allocate_tensors()
    except Exception as exc:
        raise ExportError(f"Saved LiteRT model failed verification: {exc}") from exc


def _import_litert_torch() -> object:
    try:
        import litert_torch
    except ImportError as exc:
        raise ExportError(
            "litert-torch is required for LiteRT export. Install dependencies with `uv sync`."
        ) from exc
    return litert_torch


def _import_tensorflow() -> object:
    try:
        import tensorflow
    except ImportError as exc:
        raise ExportError(
            "TensorFlow is required for LiteRT fp16 conversion flags. Install dependencies with `uv sync`."
        ) from exc
    return tensorflow
