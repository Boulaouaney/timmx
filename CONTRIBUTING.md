# Contributing — Adding a New Export Backend

`timmx` uses a registry-based plugin architecture. Each export backend is a self-contained module that owns its CLI flags and export logic. The CLI itself never needs modification when adding a new format.

## Project Layout

```
src/timmx/
├── cli.py                      # Typer-based CLI dispatcher
├── errors.py                   # TimmxError → ConfigurationError / ExportError
└── export/
    ├── base.py                 # ExportBackend ABC
    ├── registry.py             # BackendRegistry + create_builtin_registry()
    ├── common.py               # Shared helpers (create_timm_model, resolve_input_size, ...)
    ├── calibration.py          # Calibration data loading for quantized exports
    ├── types.py                # Shared enums (Device, etc.)
    ├── onnx_backend.py         # ONNX backend
    ├── coreml_backend.py       # Core ML backend
    ├── litert_backend.py       # LiteRT / TFLite backend
    ├── tensorrt_backend.py     # TensorRT backend
    └── torch_export_backend.py # torch.export backend
```

## Steps

### 1. Create the backend module

Create `src/timmx/export/<format>_backend.py` and implement the `ExportBackend` interface:

```python
from timmx.export.base import ExportBackend


class MyFormatBackend(ExportBackend):
    @property
    def name(self) -> str:
        return "my-format"

    @property
    def help(self) -> str:
        return "Export a timm model to MyFormat."

    def create_command(self):
        # Return a Typer-compatible function with Annotated parameters
        ...
```

### 2. Register in the backend registry

In `src/timmx/export/registry.py`, import your backend and add it to `create_builtin_registry()`:

```python
from timmx.export.my_format_backend import MyFormatBackend

def create_builtin_registry() -> BackendRegistry:
    registry = BackendRegistry()
    # ... existing backends ...
    registry.register(MyFormatBackend())
    return registry
```

### 3. Add tests

At minimum, add:

- **CLI help output** — verify the subcommand appears and flags are documented (via `typer.testing.CliRunner`)
- **Registry** — verify the backend is present in the default registry
- **Smoke test** — at least one end-to-end call to `backend.create_command()(**kwargs)`

### 4. Update the README

Add the new format to the supported formats table, include usage examples, and check off the roadmap entry if applicable.

## Design Conventions

- **CLI flags**: use `typer.Option("--flag-name")` with explicit `param_decls` for store-true flags (no `--no-` form). Use plain `bool` defaults for `--flag/--no-flag` toggles.
- **Choice types**: use `StrEnum` (e.g., `Device`, `LiteRTMode`).
- **Input size**: `tuple[int, int, int] | None`.
- **Errors**: raise `TimmxError` subclasses (`ConfigurationError`, `ExportError`) for user-facing failures. The CLI catches these and exits with code 2.
- **Typing**: Python `>=3.11` built-in syntax only (`list[str]`, `A | B`). No `typing` imports unless strictly necessary.
- **Line length**: 100 characters.
