from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class DependencyStatus:
    """Result of checking whether a backend's dependencies are importable."""

    available: bool
    missing_packages: list[str]
    install_hint: str


class ExportBackend(ABC):
    """Contract implemented by each export backend."""

    name: str
    help: str

    @abstractmethod
    def create_command(self) -> Callable[..., None]:
        """Return a Typer-compatible command function."""

    def check_dependencies(self) -> DependencyStatus:
        """Check if this backend's required dependencies are importable.

        Backends with optional dependencies should override this method.
        The default returns available=True (no extra deps beyond torch).
        """
        return DependencyStatus(available=True, missing_packages=[], install_hint="")
