from __future__ import annotations

import functools

import typer

from timmx import __version__
from timmx.console import console
from timmx.errors import TimmxError
from timmx.export import BackendRegistry, create_builtin_registry

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
        status = backend.check_dependencies()
        if status.available:
            table.add_row(name, "[green]\u2705 available[/green]", "")
        else:
            all_available = False
            missing_str = ", ".join(status.missing_packages)
            table.add_row(
                name,
                f"[red]\u274c missing[/red] ({missing_str})",
                f"[dim]{status.install_hint}[/dim]",
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
