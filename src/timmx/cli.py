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


def main() -> None:
    app()
