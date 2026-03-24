# AGENTS.md

## Purpose

This repository provides `timmx`, an extensible CLI/package for exporting `timm` models to
deployment formats. Keep architecture backend-oriented so new formats can be added with minimal
touch points.

Current built-in backends:
- `coreml`
- `executorch`
- `litert`
- `ncnn`
- `onnx`
- `rknn`
- `tensorrt`
- `torch-export`
- `torchscript`

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
- Export backend interface: `src/timmx/export/base.py` (`ExportBackend` ABC, `DependencyStatus`)
- Backend registry: `src/timmx/export/registry.py`
- Backend implementations: `src/timmx/export/<format>_backend.py`
- Shared model helpers: `src/timmx/export/common.py` (includes `PrePostWrapper` for preprocessing/postprocessing wrapping, `wrap_with_preprocessing()` helper, and `MeanOpt`/`StdOpt`/`NormalizeOpt`/`SoftmaxOpt` Typer type aliases)
- Shared console: `src/timmx/console.py` (rich `Console` instance for all terminal output)
- CLI entrypoint: `src/timmx/cli.py` (includes `info` model inspection, `list` model search, and `doctor` diagnostic commands)
- Tests: `tests/`

## Backend Design Contract

Every backend must:
- Implement `ExportBackend` (`name`, `help`, `create_command`)
- If the backend has optional dependencies, override `check_dependencies()` returning `DependencyStatus`
- `create_command()` returns a Typer-compatible function with `Annotated` type parameters
- Own all format-specific CLI flags (as Typer-annotated params) in its own module
- Raise `timmx.errors.TimmxError` subclasses for user-facing failures (the CLI wrapper catches these and exits with code 2)
- Use `typer.Option("--flag-name")` with explicit param_decls for store-true flags (no `--no-` form)
- Use plain `bool` defaults (no explicit param_decls) for `--flag/--no-flag` toggles
- Use `StrEnum` types for choices (e.g., `Device`, `LiteRTMode`, `ConvertTo`)
- Use `tuple[int, int, int] | None` for `--input-size`

The CLI must remain format-agnostic and dispatch through the registry.

Runtime nuance:
- `--in-chans` currently supports only `1` or `3`. For 1-channel models, shared
  normalization/calibration helpers average RGB mean/std values down to a single grayscale value.
- For `onnx`, `--slim` (default `True`) runs onnxslim after export for graph optimization (constant
  folding, dead-code elimination, operator fusion); disable with `--no-slim`.
- For `coreml`, `--source` selects model capture: `trace` (default, `torch.jit.trace`) or
  `torch-export` (beta, `torch.export.export()` → `run_decompositions({})` → `ct.convert()`).
  With `torch-export`, `ct.convert()` auto-infers shapes from the `ExportedProgram` (no `inputs=`
  needed), and `--dynamic-batch` requires `--batch-size >= 2`. `--batch-upper-bound` applies to
  both sources (sets `max=` on `torch.export.Dim` for torch-export, `ct.RangeDim.upper_bound`
  for trace).
- For `coreml`, `--compute-precision` is valid only when `--convert-to mlprogram`.
- For `litert`, supported modes are `fp32`, `fp16`, `dynamic-int8`, and `int8`.
- For `litert`, `--nhwc-input` exposes the first model input as NHWC (channel-last).
- For `litert` (and `tensorrt`, `executorch`) int8 modes, `--calibration-data` accepts either an
  image directory (timm transforms applied automatically, `--calibration-samples` limits count,
  default 128) or a torch-saved tensor `(N, C, H, W)`. Int8 requires `--calibration-data` or
  the explicit `--random-calibration` escape hatch (random noise, not recommended for production).
  `--mean`/`--std` override the timm data config for calibration image normalization (useful for
  fine-tuned models trained with custom normalization).
- For `rknn`, exports via TorchScript or ONNX intermediate → RKNN conversion using rknn-toolkit2
  (Linux x86_64/aarch64 only, Python 3.11-3.12). `--source` selects intermediate format:
  `torchscript` (default, uses `torch.jit.trace` → `rknn.load_pytorch()`, no onnx dep) or `onnx`
  (uses `torch.onnx.export(dynamo=False)` → `rknn.load_onnx()`, requires `onnx>=1.16,<1.19`).
  `--target-platform` selects the Rockchip SoC (default rk3588). Modes are `fp32`, `fp16`, `int8`.
  INT8 requires `--calibration-data <image-directory>` — tensor files and `--random-calibration` are
  not supported (RKNN loads/preprocesses images internally). `--quant-algorithm` selects
  normal/mmse/kl_divergence; `--quant-method` selects channel/layer. No dynamic batch support (RKNN
  compiles to static NPU graphs). `--opset` (ONNX opset, capped at 19) and `--keep-onnx` only apply
  to `--source onnx`. Normalization is handled by RKNN config by default (mean/std scaled to
  [0,255]); `--normalize` embeds it in the graph instead, which may affect int8 calibration accuracy.
- For `ncnn`, `--output` is a directory (not a file); pnnx intermediate files (`model.pt`, `model.pnnx.*`,
  `model_pnnx.py`) and `__pycache__` are removed automatically after export. `--fp16` defaults to `True`.
  Requires `pip install 'timmx[ncnn]'` (installs `pnnx` only; the `ncnn` Python package is not needed
  for export — the conversion is handled internally by `pnnx`).
- For `tensorrt`, `--device cuda`, `pip install tensorrt`, and `onnxscript` (via `pip install 'timmx[onnx]'`)
  are required. TensorRT export uses dynamo-based ONNX as an intermediate step.
- For `tensorrt`, ONNX intermediate export uses `external_data=False` to embed weights inline.
- For `tensorrt`, `--dynamic-batch` requires `--batch-size >= 2` and uses `torch.export.Dim` for
  dynamic shape capture. Supported precision modes are `fp32`, `fp16`, `int8`.
- For `executorch`, delegates are selected via `--delegate xnnpack` (default) or `--delegate coreml`.
- For `executorch`, modes are `fp32` and `int8`. INT8 uses PT2E quantization with the appropriate
  quantizer per delegate (`XNNPACKQuantizer` for xnnpack, `CoreMLQuantizer` for coreml).
- For `executorch`, `--compute-precision float16|float32` controls CoreML compute precision (only
  valid with `--delegate coreml`, defaults to float16). CoreML int8 auto-sets iOS 17 deployment target.
- For `executorch`, `--dynamic-batch` requires `--batch-size >= 2`.
- For `torch-export`, dynamic batch capture is only stable with sample `--batch-size >= 2`.
- For `torchscript`, `--method` selects `trace` (default, recommended) or `script`.
- For `onnx`, `torchscript`, `coreml`, `torch-export`, `ncnn`, `executorch`, `litert`, `rknn`,
  and `tensorrt`, `--normalize` wraps the model with timm's mean/std normalization (via
  `PrePostWrapper` in `common.py`), so exported models accept unnormalized `[0, 1]` float input.
  `--softmax` adds a softmax output layer independently; combine it with `--normalize` when you want
  both embedded preprocessing and probability outputs, or use it alone if your runtime already feeds
  normalized tensors. `--mean`/`--std` override embedded normalization and therefore require
  `--normalize`; for `litert`, `tensorrt`, and `executorch` int8 calibration, they also override
  calibration image preprocessing.

## Adding a New Export Backend

1. Create `src/timmx/export/<format>_backend.py`.
2. Implement `ExportBackend` with `create_command()` returning a Typer-compatible function.
3. If the backend has optional dependencies, override `check_dependencies()` returning
   `DependencyStatus` and add the extra to `[project.optional-dependencies]` in `pyproject.toml`.
4. Register the backend in `create_builtin_registry()` in
   `src/timmx/export/registry.py`.
5. Add tests:
   - CLI help output coverage (via `typer.testing.CliRunner`)
   - Registry coverage
   - At least one end-to-end export smoke test calling `backend.create_command()(**kwargs)`
6. Update `README.md` format support and usage examples.

## Quality Gates Before Shipping

Run these from repo root:

```bash
uv sync --extra onnx --extra coreml --extra ncnn --group dev
uvx ruff format .
uvx ruff check .
uv run pytest
uv build
```

## Dependencies

Core dependencies (`timm`, `torch`, `typer`, `rich`) are in `[project.dependencies]`. Backend-specific
deps are optional extras in `[project.optional-dependencies]`: `onnx`, `coreml`, `litert`, `ncnn`, `executorch`.
TensorRT and RKNN cannot be resolved cross-platform so they are not extras — users install them directly
with `pip install tensorrt` or `pip install rknn-toolkit2`. RKNN requires Linux (x86_64/aarch64) and
Python 3.11-3.12; with `--source onnx` it also requires `onnx >= 1.16, < 1.19` (rknn-toolkit2 uses
`onnx.mapping` which was removed in onnx 1.19) and `setuptools` (for `pkg_resources`); the default
`--source torchscript` has no onnx dependency. The `executorch` and `litert` extras conflict on torch version
requirements (`torch>=2.10.0` vs `torch<2.10.0`) and cannot be installed together — this is declared
via `[tool.uv] conflicts` in `pyproject.toml`.

## Scope Discipline

- Keep changes surgical.
- Avoid speculative abstractions.
- Add configuration only when a real backend requires it.
