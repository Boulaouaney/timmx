from __future__ import annotations

import argparse
import sys

from timmx.errors import TimmxError
from timmx.export import BackendRegistry, create_builtin_registry
from timmx.export.base import ExportBackend


def build_parser(registry: BackendRegistry | None = None) -> argparse.ArgumentParser:
    active_registry = registry or create_builtin_registry()

    parser = argparse.ArgumentParser(
        prog="timmx",
        description="Export timm models to deployment formats.",
    )
    root_subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = root_subparsers.add_parser(
        "export",
        help="Export a model to a deployment format.",
        description="Export a model to a deployment format.",
    )
    export_subparsers = export_parser.add_subparsers(dest="format", required=True)

    for name, backend in active_registry.items():
        _add_backend_parser(export_subparsers, name, backend)

    return parser


def run(argv: list[str] | None = None, registry: BackendRegistry | None = None) -> int:
    parser = build_parser(registry=registry)
    args = parser.parse_args(argv)
    backend = getattr(args, "_backend", None)

    if backend is None:
        parser.print_help(sys.stderr)
        return 2

    try:
        return backend.run(args)
    except TimmxError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    return run(argv)


def _add_backend_parser(
    export_subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    backend: ExportBackend,
) -> None:
    backend_parser = export_subparsers.add_parser(
        name,
        help=backend.help,
        description=backend.help,
    )
    backend.add_arguments(backend_parser)
    backend_parser.set_defaults(_backend=backend)
