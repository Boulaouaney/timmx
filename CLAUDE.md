# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`timmx` is an extensible CLI and Python package for exporting [timm](https://github.com/huggingface/pytorch-image-models) models to deployment formats. Built-in backends: `onnx`, `coreml`, `litert`, `tensorrt`, `executorch`, `torch-export`, `torchscript`.

## Tooling

- All execution via `uv`: `uv run timmx ...`, `uv run python`, `uv run -m module_name`
- Format/lint: `uvx ruff format . && uvx ruff check .` (import sorting is included via `pyproject.toml`)
- Build: `uv build`

## Development Commands

```bash
uv sync --extra all --group dev           # Install non-conflicting extras + pytest
uv run pytest                             # Run all tests
uv run pytest tests/test_cli.py           # Run a single test file
uv run pytest tests/test_cli.py::test_name  # Run a single test
```

### Quality gates (run before shipping)

```bash
uv sync --extra all --group dev
uvx ruff format .
uvx ruff check .
uv run pytest
uv build
```

## Architecture

The codebase uses a **registry-based plugin architecture**:

- `src/timmx/export/base.py` — `ExportBackend` ABC: each backend implements `name`, `help`, `create_command()`, and optionally `check_dependencies() -> DependencyStatus`
- `src/timmx/export/registry.py` — `BackendRegistry` dict + `create_builtin_registry()` factory that registers all built-in backends
- `src/timmx/export/types.py` — shared `Device` enum used by all backends
- `src/timmx/console.py` — shared `rich.console.Console` instance used for all terminal output
- `src/timmx/cli.py` — Typer-based dispatcher; dynamically registers backend commands on an `export` sub-app; catches `TimmxError` and exits with code 2; also provides the `doctor` diagnostic command
- `src/timmx/export/common.py` — shared helpers: `create_timm_model()`, `resolve_input_size()`, `validate_common_args()`
- `src/timmx/export/calibration.py` — calibration data loading/slicing for quantized exports
- `src/timmx/errors.py` — `TimmxError` → `ConfigurationError` / `ExportError`

Each backend (`onnx_backend.py`, `coreml_backend.py`, `litert_backend.py`, `tensorrt_backend.py`, `executorch_backend.py`, `torch_export_backend.py`, `torchscript_backend.py`) owns all its format-specific CLI flags and export logic. The CLI never needs modification when adding a new backend.

## Adding a New Backend

1. Create `src/timmx/export/<format>_backend.py` implementing `ExportBackend`
2. If the backend has optional dependencies, override `check_dependencies()` returning `DependencyStatus` and add the extra to `[project.optional-dependencies]` in `pyproject.toml`
3. Register in `create_builtin_registry()` in `src/timmx/export/registry.py`
4. Add tests: CLI arg parsing, registry, and at least one smoke/unit test
5. Update `README.md` with usage examples

## Python / Typing Rules

- Python `>=3.11`; use modern built-in typing syntax (`list[str]`, `dict[str, int]`, `A | B`)
- Do not import from `typing` unless a built-in equivalent is unavailable
- Line length: 100 characters

## Dependencies

Core dependencies (`timm`, `torch`, `typer`, `rich`) are in `[project.dependencies]`. Backend-specific deps are optional extras in `[project.optional-dependencies]`: `onnx`, `coreml`, `litert`, `executorch`. TensorRT is not an extra (can't be resolved cross-platform) — users install it directly with `pip install tensorrt`. The `executorch` and `litert` extras conflict on torch version requirements (`torch>=2.10.0` vs `torch<2.10.0`) and cannot be installed together; this is declared via `[tool.uv] conflicts`.

## Backend-Specific Notes

- **coreml**: `--compute-precision` is only valid with `--convert-to mlprogram`
- **litert**: quantization modes are `fp32`, `fp16`, `dynamic-int8`, `int8`; `--calibration-data` expects a torch-saved tensor of shape `(N, C, H, W)`; `--nhwc-input` exposes channel-last input layout
- **tensorrt**: requires `--device cuda`, `pip install tensorrt` (Linux/Windows with CUDA only), and `onnxscript` (install via `pip install 'timmx[onnx]'`) for dynamo-based ONNX intermediate export; `external_data=False` embeds weights inline; `--dynamic-batch` requires `--batch-size >= 2` and uses `torch.export.Dim` for dynamic shape capture; quantization modes are `fp32`, `fp16`, `int8`
- **executorch**: delegates via `--delegate xnnpack` (default) or `--delegate coreml`; `--mode fp16` only valid with `--delegate coreml`; `--mode int8` only valid with `--delegate xnnpack` (uses PT2E quantization via `torchao`); `--per-channel/--no-per-channel` controls quantization granularity; `--dynamic-batch` requires `--batch-size >= 2`; output is `.pte`
- **torch-export**: dynamic batch capture requires `--batch-size >= 2` for stable symbolic shapes
- **torchscript**: `--method` selects `trace` (default, recommended) or `script`
