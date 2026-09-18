"""timmx package."""

from importlib.metadata import version

from timmx.errors import ConfigurationError, ExportError, TimmxError

__all__ = [
    "ConfigurationError",
    "ExportError",
    "TimmxError",
    "__version__",
    "backends",
    "export_model",
]
__version__ = version("timmx")


def __getattr__(name: str) -> object:
    # The API pulls in every backend (and torch); load it on first use so `import timmx` stays cheap.
    if name in ("backends", "export_model"):
        from timmx import api

        return getattr(api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
