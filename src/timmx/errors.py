class TimmxError(Exception):
    """Base exception for user-facing timmx errors."""


class ConfigurationError(TimmxError):
    """Raised when CLI arguments are invalid or incompatible."""


class ExportError(TimmxError):
    """Raised when an export operation fails."""
