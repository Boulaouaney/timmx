# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`timmx` is an extensible CLI and Python package for exporting [timm](https://github.com/huggingface/pytorch-image-models) models to deployment formats. Built-in backends: `onnx`, `coreml`, `litert`, `ncnn`, `tensorrt`, `rknn`, `executorch`, `torch-export`, `torchscript`.

## Tooling

- All execution via `uv`: `uv run timmx ...`, `uv run python`, `uv run -m module_name`
- Format/lint: `uvx ruff format . && uvx ruff check .` (import sorting is included via `pyproject.toml`)
- Build: `uv build`

## Development Commands

```bash
uv sync --extra onnx --extra coreml --extra ncnn --group dev  # Install extras + pytest
uv run pytest                             # Run all tests
uv run pytest tests/test_cli.py           # Run a single test file
uv run pytest tests/test_cli.py::test_name  # Run a single test
```

### Quality gates (run before shipping)

```bash
uv sync --extra onnx --extra coreml --extra ncnn --group dev
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
- `src/timmx/cli.py` — Typer-based dispatcher; dynamically registers backend commands on an `export` sub-app; catches `TimmxError` and exits with code 2; also provides the `info` and `doctor` top-level commands and the `list` command for searching available timm models
- `src/timmx/export/common.py` — shared helpers: `create_timm_model()`, `resolve_input_size()`, `validate_common_args()`, `PrePostWrapper` (preprocessing/postprocessing `nn.Module` wrapper), `wrap_with_preprocessing()`
- `src/timmx/export/calibration.py` — calibration data loading/slicing for quantized exports; supports image directories (with timm transforms), torch-saved tensors, and random noise (opt-in via `--random-calibration`)
- `src/timmx/errors.py` — `TimmxError` → `ConfigurationError` / `ExportError`

Each backend (`onnx_backend.py`, `coreml_backend.py`, `litert_backend.py`, `ncnn_backend.py`, `tensorrt_backend.py`, `executorch_backend.py`, `torch_export_backend.py`, `torchscript_backend.py`) owns all its format-specific CLI flags and export logic. The CLI never needs modification when adding a new backend.

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

Core dependencies (`timm`, `torch`, `typer`, `rich`) are in `[project.dependencies]`. Backend-specific deps are optional extras in `[project.optional-dependencies]`: `onnx`, `coreml`, `litert`, `ncnn`, `executorch`. TensorRT and RKNN are not extras (can't be resolved cross-platform) — users install them directly with `pip install tensorrt` or `pip install rknn-toolkit2` respectively. RKNN requires Linux (x86_64/aarch64), Python 3.11-3.12; with `--source onnx` it also requires `onnx >= 1.16, < 1.19` (rknn-toolkit2 uses `onnx.mapping` which was removed in onnx 1.19) and `setuptools` (for `pkg_resources`); the default `--source torchscript` has no onnx dependency. The `executorch` and `litert` extras conflict on torch version requirements (`torch>=2.10.0` vs `torch<2.10.0`) and cannot be installed together; this is declared via `[tool.uv] conflicts`.

## Backend-Specific Notes

- **shared export helpers**: `--in-chans` currently supports only `1` or `3`; for 1-channel models, shared normalization and calibration logic averages RGB mean/std values down to a single grayscale value; all backends support `--normalize` (embeds timm mean/std normalization via `PrePostWrapper`), `--softmax` (adds softmax output layer — independent of `--normalize`), and `--mean`/`--std` (override timm config, require `--normalize`; for quantization backends they also override calibration preprocessing)
- **onnx**: `--slim` (default `True`) runs onnxslim after export for graph optimization (constant folding, dead-code elimination, operator fusion); disable with `--no-slim`
- **coreml**: `--source` selects model capture: `trace` (default, `torch.jit.trace`) or `torch-export` (beta, `torch.export.export()`); with `torch-export`, `ct.convert()` auto-infers shapes from the `ExportedProgram` and `--dynamic-batch` requires `--batch-size >= 2`; `--batch-upper-bound` applies to both sources (sets `max=` on `torch.export.Dim` for torch-export, `ct.RangeDim.upper_bound` for trace); `--compute-precision` is only valid with `--convert-to mlprogram`; weight quantization via `--half` (fp16), `--int8` (8-bit), `--int4` (4-bit, mlpackage only) — mutually exclusive; neuralnetwork uses `ct.models.neural_network.quantization_utils.quantize_weights()` with linear mode, mlprogram uses `ct.optimize.coreml.palettize_weights()` k-means for int8/int4 (--half is a no-op on mlprogram since weights are already fp16); k-means palettization requires `scikit-learn` (included in the `coreml` extra)
- **litert**: quantization modes are `fp32`, `fp16`, `dynamic-int8`, `int8`; `--calibration-data` accepts an image directory or a torch-saved tensor `(N, C, H, W)`; int8 modes require `--calibration-data` or `--random-calibration`; `--calibration-samples` limits images loaded from a directory (default 128); `--nhwc-input` exposes channel-last input layout; `--mean`/`--std` override timm config for calibration normalization
- **ncnn**: `--output` is a directory (not a file); writes `model.ncnn.param`, `model.ncnn.bin`, `model_ncnn.py`; pnnx intermediate files and `__pycache__` are cleaned up automatically; `--fp16` defaults to `True`; uses `pnnx.export()` internally (traces via TorchScript then converts); only `pnnx` is required — the `ncnn` Python package is not needed for export
- **tensorrt**: requires `--device cuda`, `pip install tensorrt` (Linux/Windows with CUDA only), and `onnxscript` (install via `pip install 'timmx[onnx]'`) for dynamo-based ONNX intermediate export; `external_data=False` embeds weights inline; `--dynamic-batch` requires `--batch-size >= 2` and uses `torch.export.Dim` for dynamic shape capture; quantization modes are `fp32`, `fp16`, `int8`; int8 requires `--calibration-data` or `--random-calibration`; `--calibration-data` accepts an image directory or torch tensor; `--calibration-samples` limits images loaded from a directory (default 128); `--mean`/`--std` override timm config for calibration normalization
- **rknn**: exports via TorchScript or ONNX intermediate → RKNN conversion using rknn-toolkit2 (Linux x86_64/aarch64 only, Python 3.11-3.12); `--source` selects intermediate format: `torchscript` (default, uses `torch.jit.trace` → `rknn.load_pytorch()`, no onnx dep) or `onnx` (uses `torch.onnx.export(dynamo=False)` → `rknn.load_onnx()`, requires `onnx>=1.16,<1.19`); `--target-platform` selects the Rockchip SoC (default rk3588); modes are `fp32`, `fp16`, `int8`; int8 requires `--calibration-data <image-directory>` (tensor files and `--random-calibration` are not supported because RKNN handles image loading internally); `--quant-algorithm` selects normal/mmse/kl_divergence; `--quant-method` selects channel/layer granularity; no dynamic batch support (RKNN compiles to static NPU graphs); `--opset` (ONNX opset, capped at 19) and `--keep-onnx` only apply to `--source onnx`; normalization is handled by RKNN config by default (using `--normalize` embeds it in the graph instead, which may affect calibration accuracy); `--mean`/`--std` override timm config for RKNN normalization and calibration
- **executorch**: delegates via `--delegate xnnpack` (default) or `--delegate coreml`; modes are `fp32` (default) and `int8` (PT2E quantization — uses `XNNPACKQuantizer` for xnnpack, `CoreMLQuantizer` for coreml); int8 requires `--calibration-data` or `--random-calibration`; `--calibration-data` accepts an image directory or torch tensor; `--calibration-samples` limits images loaded from a directory (default 128); `--compute-precision float16|float32` controls CoreML compute precision (only with `--delegate coreml`, defaults to float16); CoreML int8 auto-sets `minimum_deployment_target=iOS17`; `--per-channel/--no-per-channel` controls quantization granularity; `--dynamic-batch` requires `--batch-size >= 2`; output is `.pte`; `--mean`/`--std` override timm config for calibration normalization
- **torch-export**: dynamic batch capture requires `--batch-size >= 2` for stable symbolic shapes
- **torchscript**: `--method` selects `trace` (default, recommended) or `script`
