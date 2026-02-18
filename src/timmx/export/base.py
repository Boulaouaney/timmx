from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable


class ExportBackend(ABC):
    """Contract implemented by each export backend."""

    name: str
    help: str

    @abstractmethod
    def create_command(self) -> Callable[..., None]:
        """Return a Typer-compatible command function."""
