# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`timmx` is an extensible CLI and Python package for exporting [timm](https://github.com/huggingface/pytorch-image-models) models to deployment formats. Built-in backends: `onnx`, `coreml`, `litert`, `torch-export`.

## Tooling

- All execution via `uv`: `uv run timmx ...`, `uv run python`, `uv run -m module_name`
- Format/lint: `uvx ruff format . && uvx ruff check .` (import sorting is included via `pyproject.toml`)
- Build: `uv build`

## Development Commands

```bash
uv sync --group dev          # Install deps including pytest
uv run pytest                # Run all tests
uv run pytest tests/test_cli.py  # Run a single test file
uv run pytest tests/test_cli.py::test_name  # Run a single test
```

### Quality gates (run before shipping)

```bash
uv sync --group dev
uvx ruff format .
uvx ruff check .
uv run pytest
uv build
```

## Architecture

The codebase uses a **registry-based plugin architecture**:

- `src/timmx/export/base.py` — `ExportBackend` ABC: each backend implements `name`, `help`, `create_command() -> Callable[..., None]`
- `src/timmx/export/registry.py` — `BackendRegistry` dict + `create_builtin_registry()` factory that registers all built-in backends
- `src/timmx/export/types.py` — shared `Device` enum used by all backends
- `src/timmx/cli.py` — Typer-based dispatcher; dynamically registers backend commands on an `export` sub-app; catches `TimmxError` and exits with code 2
- `src/timmx/export/common.py` — shared helpers: `create_timm_model()`, `resolve_input_size()`, `validate_common_args()`
- `src/timmx/export/calibration.py` — calibration data loading/slicing for quantized exports
- `src/timmx/errors.py` — `TimmxError` → `ConfigurationError` / `ExportError`

Each backend (`onnx_backend.py`, `coreml_backend.py`, `litert_backend.py`, `torch_export_backend.py`) owns all its format-specific CLI flags and export logic. The CLI never needs modification when adding a new backend.

## Adding a New Backend

1. Create `src/timmx/export/<format>_backend.py` implementing `ExportBackend`
2. Register in `create_builtin_registry()` in `src/timmx/export/registry.py`
3. Add tests: CLI arg parsing, registry, and at least one smoke/unit test
4. Update `README.md` with usage examples

## Python / Typing Rules

- Python `>=3.11`; use modern built-in typing syntax (`list[str]`, `dict[str, int]`, `A | B`)
- Do not import from `typing` unless a built-in equivalent is unavailable
- Line length: 100 characters

## Backend-Specific Notes

- **coreml**: `--compute-precision` is only valid with `--convert-to mlprogram`
- **litert**: quantization modes are `fp32`, `fp16`, `dynamic-int8`, `int8`; `--calibration-data` expects a torch-saved tensor of shape `(N, C, H, W)`; `--nhwc-input` exposes channel-last input layout
- **torch-export**: dynamic batch capture requires `--batch-size >= 2` for stable symbolic shapes
