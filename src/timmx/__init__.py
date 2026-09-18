"""timmx package."""

from importlib.metadata import version

from timmx.api import backends, export
from timmx.errors import ConfigurationError, ExportError, TimmxError

__all__ = [
    "ConfigurationError",
    "ExportError",
    "TimmxError",
    "__version__",
    "backends",
    "export",
]
__version__ = version("timmx")
