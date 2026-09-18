"""Python API: the same exports as the CLI, as a function call.

The function is `export_model`, not `export`: `timmx.export` is the backend subpackage, and rebinding
it on the package would shadow the module for `timmx.export.<module>` attribute lookups.
"""

from __future__ import annotations

import inspect
import types
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Union, get_args, get_origin

from timmx.errors import ConfigurationError
from timmx.export import create_builtin_registry


def backends() -> list[str]:
    """Names of the registered export backends (the `timmx export <name>` subcommands)."""
    return create_builtin_registry().names()


def export_model(backend: str, model_name: str, **options: object) -> Path:
    """Export a timm model with one backend and return the written path.

    *options* are the backend's CLI flags as keyword arguments (``--dynamic-batch`` becomes
    ``dynamic_batch=True``; paths may be strings and choices are their plain string values, as on
    the command line)::

        timmx.export_model("onnx", "resnet18", pretrained=True, dynamic_batch=True, output="r18.onnx")

    ``output`` defaults like the CLI (``<model name>.<ext>`` in the current directory).  Unknown
    options, unknown choices and export failures raise :class:`timmx.errors.TimmxError` subclasses.
    """
    registry = create_builtin_registry()
    try:
        implementation = registry.get(backend)
    except KeyError:
        raise ConfigurationError(
            f"Unknown backend {backend!r}; available: {', '.join(registry.names())}."
        ) from None
    command = implementation.create_command()
    return command(model_name, **_coerce_options(command, options))


def _coerce_options(command: object, options: dict[str, object]) -> dict[str, object]:
    """Apply the conversions Typer does on the command line: str → Path, str → StrEnum choice."""
    parameters = inspect.signature(command, eval_str=True).parameters
    unknown = sorted(set(options) - set(parameters))
    if unknown:
        raise ConfigurationError(
            f"Unknown option(s) {', '.join(unknown)}; valid options: "
            f"{', '.join(name for name in parameters if name != 'model_name')}."
        )
    return {
        name: _coerce(name, value, parameters[name].annotation) for name, value in options.items()
    }


def _coerce(name: str, value: object, annotation: object) -> object:
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    members = (
        get_args(annotation)
        if get_origin(annotation) in (types.UnionType, Union)
        else (annotation,)
    )
    if not isinstance(value, str):
        return value
    for member in members:
        if member is Path:
            return Path(value)
        if isinstance(member, type) and issubclass(member, StrEnum):
            try:
                return member(value)
            except ValueError:
                choices = ", ".join(choice.value for choice in member)
                raise ConfigurationError(
                    f"Unknown value {value!r} for {name}; choices: {choices}."
                ) from None
    return value
