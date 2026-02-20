# AGENTS.md

## Purpose

This repository provides `timmx`, an extensible CLI/package for exporting `timm` models to
deployment formats. Keep architecture backend-oriented so new formats can be added with minimal
touch points.

Current built-in backends:
- `coreml`
- `litert`
- `onnx`
- `tensorrt`
- `torch-export`

## Non-Negotiable Tooling Rules

- Use `uv` for dependency management, execution, and builds.
- Script execution: `uv run script.py`
- Python interpreter: `uv run python`
- Module execution: `uv run -m module_name`
- CLI tools (including this project): `uv run timmx ...`
- Build packages with `uv build`
- Run Ruff only through `uvx ruff`

## Python and Typing Rules

- Minimum Python version is `>=3.11`.
- Use modern built-in typing syntax (`list[str]`, `dict[str, int]`, `A | B`).
- Do not import from `typing` unless a genuinely missing built-in type feature is required.

## Project Layout

- Source package: `src/timmx/`
- Export backend interface: `src/timmx/export/base.py`
- Backend registry: `src/timmx/export/registry.py`
- Backend implementations: `src/timmx/export/<format>_backend.py`
- Shared model helpers: `src/timmx/export/common.py`
- CLI entrypoint: `src/timmx/cli.py`
- Tests: `tests/`

## Backend Design Contract

Every backend must:
- Implement `ExportBackend` (`name`, `help`, `create_command`)
- `create_command()` returns a Typer-compatible function with `Annotated` type parameters
- Own all format-specific CLI flags (as Typer-annotated params) in its own module
- Raise `timmx.errors.TimmxError` subclasses for user-facing failures (the CLI wrapper catches these and exits with code 2)
- Use `typer.Option("--flag-name")` with explicit param_decls for store-true flags (no `--no-` form)
- Use plain `bool` defaults (no explicit param_decls) for `--flag/--no-flag` toggles
- Use `StrEnum` types for choices (e.g., `Device`, `LiteRTMode`, `ConvertTo`)
- Use `tuple[int, int, int] | None` for `--input-size`

The CLI must remain format-agnostic and dispatch through the registry.

Runtime nuance:
- For `coreml`, `--compute-precision` is valid only when `--convert-to mlprogram`.
- For `litert`, supported modes are `fp32`, `fp16`, `dynamic-int8`, and `int8`.
- For `litert`, `--nhwc-input` exposes the first model input as NHWC (channel-last).
- For `litert` quantized modes, calibration data can be provided via `--calibration-data` as a
  torch-saved tensor with shape `(N, C, H, W)`.
- For `tensorrt`, `--device cuda` and the `tensorrt` pip package are required (not in core deps).
- For `tensorrt`, ONNX intermediate export uses `external_data=False` to embed weights inline.
- For `tensorrt`, `--dynamic-batch` requires `--batch-size >= 2` and uses `torch.export.Dim` for
  dynamic shape capture. Supported precision modes are `fp32`, `fp16`, `int8`.
- For `torch-export`, dynamic batch capture is only stable with sample `--batch-size >= 2`.

## Adding a New Export Backend

1. Create `src/timmx/export/<format>_backend.py`.
2. Implement `ExportBackend` with `create_command()` returning a Typer-compatible function.
3. Register the backend in `create_builtin_registry()` in
   `src/timmx/export/registry.py`.
4. Add tests:
   - CLI help output coverage (via `typer.testing.CliRunner`)
   - Registry coverage
   - At least one end-to-end export smoke test calling `backend.create_command()(**kwargs)`
5. Update `README.md` format support and usage examples.

## Quality Gates Before Shipping

Run these from repo root:

```bash
uv sync --group dev
uvx ruff format .
uvx ruff check .
uv run pytest
uv build
```

## Scope Discipline

- Keep changes surgical.
- Avoid speculative abstractions.
- Add configuration only when a real backend requires it.
