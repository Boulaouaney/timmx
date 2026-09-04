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
- `openvino`
- `tensorrt`
- `torch-export`
- `torchscript`

## Development Commands

```bash
uv sync --extra onnx --extra openvino --extra coreml --extra ncnn --group dev  # install extras + pytest
uv run pytest                               # all tests
uv run pytest tests/test_cli.py::test_name  # one test
uvx ruff format . && uvx ruff check .       # format + lint (import sorting included)
uv build
```

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
- `torch.onnx.export` is always called with `dynamo=True`; torch `>=2.11` removed the `fallback`
  kwarg, so never pass it.
- For `coreml`, `--source` selects model capture: `trace` (default, `torch.jit.trace`) or
  `torch-export` (beta, `torch.export.export()` → `run_decompositions({})` → `ct.convert()`).
  With `torch-export`, `ct.convert()` auto-infers shapes from the `ExportedProgram` (no `inputs=`
  needed), and `--dynamic-batch` requires `--batch-size >= 2`. `--batch-upper-bound` applies to
  both sources (sets `max=` on `torch.export.Dim` for torch-export, `ct.RangeDim.upper_bound`
  for trace).
- For `coreml`, `--compute-precision` is valid only when `--convert-to mlprogram`.
- For `openvino`, `--output` must be an `.xml` path (the `.bin` is written alongside); `--fp16`
  (default `True`) compresses weights via `ov.save_model(compress_to_fp16=...)`; `--dynamic-batch`
  sets the batch dim to `-1`; verification compiles the IR on the OpenVINO `CPU` device and runs a
  forward pass.
- For `litert`, supported modes are `fp32`, `fp16`, `dynamic-int8`, and `int8`. `fp16` and
  `dynamic-int8` are post-training weight quantization of the saved `.tflite` via
  `ai-edge-quantizer` (no calibration); `int8` is static PT2E quantization (per-channel by
  default, `--no-per-channel` for per-tensor) and needs calibration data.
- For `coreml`, `--half`/`--int8`/`--int4` are mutually exclusive weight quantization flags.
  neuralnetwork uses `quantize_weights()` (`linear` for fp16, `linear_symmetric` for int8);
  mlprogram uses `linear_quantize_weights()` (per-channel int8) and `palettize_weights()`
  (k-means int4, needs scikit-learn). `--half` is a no-op on mlprogram (already fp16).
- For `litert`, `--nhwc-input` exposes the first model input as NHWC (channel-last).
- Known test caveat: with every extra installed, the two `litert` static-int8 tests can fail in a
  full in-process `pytest` run with `No module named 'litert_converter.mlir.dialects.quant'`
  (litert-torch ships no Python `quant` dialect; b/362798610). They pass in isolation and the CLI
  export works, since a real export is a fresh process.
- For `litert`, `tensorrt` and `executorch` `--mode int8`, `--calibration-data` accepts either an
  image directory (timm transforms applied automatically, `--calibration-samples` limits count,
  default 128) or a torch-saved tensor `(N, C, H, W)`. Int8 requires `--calibration-data` or
  the explicit `--random-calibration` escape hatch (random noise, not recommended for production).
  `--mean`/`--std` override the timm data config for calibration image normalization (useful for
  fine-tuned models trained with custom normalization).
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
- For `onnx`, `openvino`, `torchscript`, `coreml`, `torch-export`, `ncnn`, `executorch`, `litert`,
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
uv sync --extra onnx --extra openvino --extra coreml --extra ncnn --group dev
uvx ruff format .
uvx ruff check .
uv run pytest
uv build
```

## Dependencies

Core dependencies (`timm`, `torch`, `typer`, `rich`) are in `[project.dependencies]`. Backend-specific
deps are optional extras in `[project.optional-dependencies]`: `onnx`, `openvino`, `coreml`, `litert`,
`ncnn`, `executorch`.
TensorRT cannot be resolved cross-platform (CUDA-only wheels) so it is not an extra — users install it
directly with `pip install tensorrt`. Core requires `torch>=2.9` and Python `>=3.11,<3.15`.
`litert-torch` requires `torch<2.14`, so the `litert` extra pins torch; `coremltools` has no
Python 3.14 wheels yet, so the `coreml` extra needs Python `<=3.13`.

## Scope Discipline

- Keep changes surgical.
- Avoid speculative abstractions.
- Add configuration only when a real backend requires it.
