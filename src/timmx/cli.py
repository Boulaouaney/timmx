from __future__ import annotations

import functools
import logging
import os
import warnings

import typer

from timmx import __version__
from timmx.console import console
from timmx.errors import TimmxError
from timmx.export import BackendRegistry, DependencyStatus, create_builtin_registry
from timmx.export.base import ExportBackend
from timmx.export.common import (
    CheckpointOpt,
    InChansOpt,
    ModelNameArg,
    NumClassesOpt,
    PretrainedOpt,
)

app = typer.Typer(
    name="timmx", help="Export timm models to deployment formats.", no_args_is_help=True
)
export_app = typer.Typer(help="Export a model to a deployment format.", no_args_is_help=True)
app.add_typer(export_app, name="export")


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"timmx {__version__}")
        raise typer.Exit()


@app.callback()
def _root_callback(
    version: bool = typer.Option(
        False,
        "--version",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    pass


def build_export_app(registry: BackendRegistry | None = None) -> None:
    active_registry = registry or create_builtin_registry()

    for name, backend in active_registry.items():
        command_fn = backend.create_command()

        @functools.wraps(command_fn)
        def wrapped(*args: object, _fn: object = command_fn, **kwargs: object) -> None:
            try:
                _fn(*args, **kwargs)
            except TimmxError as exc:
                console.print(f"[bold red]error:[/bold red] {exc}", highlight=False)
                raise typer.Exit(code=2) from exc

        export_app.command(name=name, help=backend.help)(wrapped)


build_export_app()


def _quiet_check(backend: ExportBackend) -> DependencyStatus:
    """Run check_dependencies with noisy third-party warnings suppressed."""
    prev_tf = os.environ.get("TF_CPP_MIN_LOG_LEVEL")
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    prev_level = logging.root.level
    logging.root.setLevel(logging.ERROR)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return backend.check_dependencies()
    finally:
        logging.root.setLevel(prev_level)
        if prev_tf is None:
            os.environ.pop("TF_CPP_MIN_LOG_LEVEL", None)
        else:
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = prev_tf


@app.command()
def doctor() -> None:
    """Check timmx installation and backend availability."""
    import platform

    import torch
    from rich.table import Table

    console.print(f"[bold]timmx[/bold] v{__version__}")
    console.print(f"Python {platform.python_version()} | torch {torch.__version__}")
    console.print()

    table = Table(title="Backends", show_lines=False)
    table.add_column("Backend", style="bold")
    table.add_column("Status")
    table.add_column("Install")

    registry = create_builtin_registry()
    all_available = True

    for name, backend in registry.items():
        status = _quiet_check(backend)
        if status.available:
            table.add_row(name, "[green]:white_check_mark: available[/green]", "")
        else:
            all_available = False
            missing_str = ", ".join(status.missing_packages)
            table.add_row(
                name,
                f"[red]:x: missing[/red] ({missing_str})",
                "[dim]{}[/dim]".format(status.install_hint.replace("[", r"\[")),
            )

    console.print(table)

    if not all_available:
        console.print()
        console.print(
            "[dim]Tip: pip install 'timmx\\[all]' installs all"
            " non-platform-specific backends.[/dim]"
        )


def _format_param_count(n: int) -> str:
    """Format a parameter count like '11.69M (11,689,512)'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M ({n:,})"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K ({n:,})"
    return f"{n:,}"


@app.command()
def info(
    model_name: ModelNameArg,
    pretrained: PretrainedOpt = False,
    checkpoint: CheckpointOpt = None,
    num_classes: NumClassesOpt = None,
    in_chans: InChansOpt = None,
) -> None:
    """Show model metadata without exporting."""
    from rich.table import Table

    from timmx.export.common import create_timm_model, resolve_input_size

    try:
        model = create_timm_model(
            model_name,
            pretrained=pretrained,
            checkpoint=checkpoint,
            num_classes=num_classes,
            in_chans=in_chans,
        )
    except TimmxError as exc:
        console.print(f"[bold red]error:[/bold red] {exc}", highlight=False)
        raise typer.Exit(code=2) from exc

    input_size = resolve_input_size(model, None)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Determine weights status
    if checkpoint is not None:
        weights_info = f"checkpoint ({checkpoint.name})"
    elif pretrained:
        weights_info = "pretrained"
    else:
        weights_info = "none (random init)"

    table = Table(show_header=False, show_edge=False, pad_edge=False)
    table.add_column(style="bold")
    table.add_column()

    table.add_row("Model", model_name)
    table.add_row("Architecture", model.__class__.__name__)
    table.add_row("Parameters", _format_param_count(total_params))
    if trainable_params != total_params:
        table.add_row("Trainable", _format_param_count(trainable_params))
    table.add_row("Input size", f"{input_size[0]} x {input_size[1]} x {input_size[2]}")
    num_cls = getattr(model, "num_classes", None)
    if num_cls is not None:
        table.add_row("Classes", str(num_cls))
    num_feat = getattr(model, "num_features", None)
    if num_feat is not None:
        table.add_row("Feature dim", str(num_feat))
    table.add_row("Weights", weights_info)

    console.print(table)


@app.command(name="list")
def list_models(
    query: str = typer.Argument("", help="Filter pattern (substring or glob)."),
    pretrained_only: bool = typer.Option(
        False, "--pretrained-only", help="Only show models with pretrained weights."
    ),
) -> None:
    """List available timm models."""
    import timm

    pattern = query if any(c in query for c in "*?[") else f"*{query}*"
    models = timm.list_models(pattern, pretrained=pretrained_only)
    for name in models:
        console.print(name, highlight=False)
    console.print()
    label = "model" if len(models) == 1 else "models"
    console.print(f"Found {len(models)} {label}")


def main() -> None:
    app()
