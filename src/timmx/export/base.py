from __future__ import annotations

import argparse
from abc import ABC, abstractmethod


class ExportBackend(ABC):
    """Contract implemented by each export backend."""

    name: str
    help: str

    @abstractmethod
    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Register backend-specific CLI arguments."""

    @abstractmethod
    def run(self, args: argparse.Namespace) -> int:
        """Execute the export and return a process exit code."""
