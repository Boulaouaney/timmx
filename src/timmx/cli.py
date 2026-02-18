from __future__ import annotations

import functools

import typer

from timmx import __version__
from timmx.errors import TimmxError
from timmx.export import BackendRegistry, create_builtin_registry

app = typer.Typer(
    name="timmx", help="Export timm models to deployment formats.", no_args_is_help=True
)
export_app = typer.Typer(help="Export a model to a deployment format.", no_args_is_help=True)
app.add_typer(export_app, name="export")


def _version_callback(value: bool) -> None:
    if value:
        print(f"timmx {__version__}")
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
                typer.echo(f"error: {exc}", err=True)
                raise typer.Exit(code=2) from exc

        export_app.command(name=name, help=backend.help)(wrapped)


build_export_app()


def main() -> None:
    app()
