from __future__ import annotations

from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import torch
import typer

from timmx.errors import ExportError
from timmx.export.base import ExportBackend
from timmx.export.common import create_timm_model, resolve_input_size, validate_common_args
from timmx.export.types import Device


class ScriptMethod(StrEnum):
    trace = "trace"
    script = "script"


class TorchScriptBackend(ExportBackend):
    name = "torchscript"
    help = "Export a timm model to TorchScript (.pt)."

    def create_command(self) -> Callable[..., None]:
        def command(
            model_name: Annotated[str, typer.Argument(help="timm model name, e.g. resnet18")],
            output: Annotated[
                Path,
                typer.Option(help="Path to write the TorchScript model."),
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
                int, typer.Option(help="Example input batch size for tracing.")
            ] = 1,
            input_size: Annotated[
                tuple[int, int, int] | None,
                typer.Option(help="Explicit input shape as C H W."),
            ] = None,
            method: Annotated[
                ScriptMethod,
                typer.Option(help="Scripting method: trace (recommended) or script."),
            ] = ScriptMethod.trace,
            device: Annotated[
                Device, typer.Option(help="Device used for model instantiation and tracing.")
            ] = Device.cpu,
            verify: Annotated[
                bool,
                typer.Option(help="Load the saved model after export and run a forward pass."),
            ] = True,
        ) -> None:
            validate_common_args(batch_size=batch_size, device=device)

            output_path = Path(output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            model = create_timm_model(
                model_name,
                pretrained=pretrained,
                checkpoint=checkpoint,
                num_classes=num_classes,
                in_chans=in_chans,
            )
            resolved_input_size = resolve_input_size(model, input_size)

            torch_device = torch.device(device)
            model = model.to(torch_device)
            model.eval()

            example_input = torch.randn(batch_size, *resolved_input_size, device=torch_device)

            try:
                if method == ScriptMethod.trace:
                    scripted = torch.jit.trace(model, example_input)
                else:
                    scripted = torch.jit.script(model)
            except Exception as exc:
                raise ExportError(f"TorchScript {method} failed: {exc}") from exc

            try:
                torch.jit.save(scripted, str(output_path))
            except Exception as exc:
                raise ExportError(f"Failed to save TorchScript model: {exc}") from exc

            if verify:
                try:
                    loaded = torch.jit.load(str(output_path), map_location=torch_device)
                    loaded(example_input)
                except Exception as exc:
                    raise ExportError(
                        f"Saved TorchScript model failed verification: {exc}"
                    ) from exc

        return command
