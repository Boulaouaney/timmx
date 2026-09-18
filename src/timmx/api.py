"""Python API: the same exports as the CLI, as a function call.

The function is `export_model`, not `export`: `timmx.export` is the backend subpackage, and rebinding
it on the package would shadow the module for `timmx.export.<module>` attribute lookups.
"""

from __future__ import annotations

from pathlib import Path

from timmx.errors import ConfigurationError
from timmx.export import create_builtin_registry


def backends() -> list[str]:
    """Names of the registered export backends (the `timmx export <name>` subcommands)."""
    return create_builtin_registry().names()


def export_model(backend: str, model_name: str, **options: object) -> Path:
    """Export a timm model with one backend and return the written path.

    *options* are the backend's CLI flags as keyword arguments (``--dynamic-batch`` becomes
    ``dynamic_batch=True``, choices are plain strings such as ``mode="int8"``)::

        timmx.export_model("onnx", "resnet18", pretrained=True, dynamic_batch=True, output="r18.onnx")

    ``output`` defaults like the CLI (``<model name>.<ext>`` in the current directory).
    Failures raise :class:`timmx.errors.TimmxError` subclasses; an unknown option raises
    ``TypeError`` from the backend function itself.
    """
    registry = create_builtin_registry()
    try:
        implementation = registry.get(backend)
    except KeyError:
        raise ConfigurationError(
            f"Unknown backend {backend!r}; available: {', '.join(registry.names())}."
        ) from None
    return implementation.create_command()(model_name, **options)
