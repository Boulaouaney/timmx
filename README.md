# timmx
[![PyPI version](https://img.shields.io/pypi/v/timmx)](https://pypi.org/project/timmx) ![Python versions](https://img.shields.io/pypi/pyversions/timmx) ![License](https://img.shields.io/pypi/l/timmx) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Boulaouaney/timmx)

An extensible CLI and Python package for exporting [timm](https://github.com/huggingface/pytorch-image-models) models to various deployment formats. Born out of having too many one-off export scripts for fine-tuned timm models — `timmx` unifies them behind a single command-line interface with a plugin-based backend system.

## Supported Formats

| Format | Command | Output |
|--------|---------|--------|
| ONNX | `timmx export onnx` | `.onnx` |
| Core ML | `timmx export coreml` | `.mlpackage` / `.mlmodel` |
| LiteRT / TFLite | `timmx export litert` | `.tflite` |
| ncnn | `timmx export ncnn` | directory (`.param` + `.bin`) |
| TensorRT | `timmx export tensorrt` | `.engine` |
| RKNN | `timmx export rknn` | `.rknn` |
| ExecuTorch | `timmx export executorch` | `.pte` |
| torch.export | `timmx export torch-export` | `.pt2` |
| TorchScript | `timmx export torchscript` | `.pt` |

## Requirements

- Python `>=3.11`
- [`uv`](https://docs.astral.sh/uv/)

## Installation

Core install (includes `timm`, `torch`, `typer`, `rich`):

```bash
pip install timmx
```

Install with specific backend extras:

```bash
pip install 'timmx[onnx]'           # ONNX export
pip install 'timmx[coreml]'         # Core ML export
pip install 'timmx[litert]'         # LiteRT/TFLite export
pip install 'timmx[ncnn]'           # ncnn export (via pnnx)
pip install 'timmx[executorch]'     # ExecuTorch export (XNNPack, CoreML delegates)
pip install 'timmx[onnx,coreml]'    # multiple backends
```

TensorRT requires CUDA and must be installed separately:

```bash
pip install tensorrt  # Linux/Windows with CUDA only
```

RKNN requires Linux and must be installed separately:

```bash
pip install rknn-toolkit2  # Linux x86_64/aarch64 only, Python 3.11-3.12
```

> **Note:** The `executorch` and `litert` extras have conflicting torch version
> requirements (`executorch` needs `torch>=2.10.0`, `litert` needs `torch<2.10.0`)
> and cannot be installed in the same environment.

Check which backends are available:

```bash
timmx doctor
```

## Quick Start

```bash
uv sync --extra onnx --extra coreml --extra ncnn --group dev
uv run timmx doctor
uv run timmx --help
```

## Model Info

Inspect a model's metadata (parameter count, input size, number of classes, etc.) without exporting:

```bash
uv run timmx info resnet18 --pretrained
```

This displays architecture details, parameter counts, default input size, and whether weights are loaded.

## Listing Models

Browse and search available timm models:

```bash
uv run timmx list resnet                        # search by substring
uv run timmx list "resnet*"                     # search by glob pattern
uv run timmx list --pretrained-only resnet      # only models with pretrained weights
```

## Usage Examples

### ONNX

```bash
uv run timmx export onnx resnet18 --pretrained --output ./artifacts/resnet18.onnx
```

Export a fine-tuned checkpoint with dynamic batching:

```bash
uv run timmx export onnx resnet18 \
  --checkpoint ./checkpoints/model.pth \
  --input-size 3 224 224 \
  --dynamic-batch \
  --output ./artifacts/resnet18_finetuned.onnx
```

Export with built-in normalization and softmax (the model will expect unnormalized `[0, 1]` float input and output probabilities):

```bash
uv run timmx export onnx resnet18 \
  --pretrained \
  --normalize --softmax \
  --output ./artifacts/resnet18_with_preprocess.onnx
```

> `--normalize` embeds the timm model's mean/std normalization into the graph. `--softmax` adds a softmax layer on the output. Use both flags together if you want a self-contained export that accepts raw `[0, 1]` float input and outputs probabilities; use `--softmax` alone if your inputs are already normalized. `--mean` / `--std` override the embedded normalization and therefore require `--normalize`. `--in-chans` currently supports only `1` or `3`; for grayscale (`--in-chans 1`) exports, RGB mean/std values are averaged down to a single channel.

Exported models are automatically optimized with [onnxslim](https://github.com/inisis/OnnxSlim) (constant folding, dead-code elimination, operator fusion). To skip optimization:

```bash
uv run timmx export onnx resnet18 --pretrained --no-slim --output ./artifacts/resnet18.onnx
```

### Core ML

```bash
uv run timmx export coreml resnet18 \
  --pretrained \
  --convert-to mlprogram \
  --compute-precision float16 \
  --output ./artifacts/resnet18.mlpackage
```

Using `torch.export` as source (beta):

```bash
uv run timmx export coreml resnet18 \
  --pretrained \
  --source torch-export \
  --convert-to mlprogram \
  --compute-precision float16 \
  --output ./artifacts/resnet18_te.mlpackage
```

Flexible batch size:

```bash
uv run timmx export coreml resnet18 \
  --dynamic-batch \
  --batch-size 2 \
  --batch-upper-bound 8 \
  --output ./artifacts/resnet18_dynamic.mlpackage
```

Weight quantization (post-conversion, applied to model weights):

```bash
# 8-bit k-means quantization (mlpackage)
uv run timmx export coreml resnet18 \
  --pretrained \
  --convert-to mlprogram \
  --compute-precision float16 \
  --int8 \
  --output ./artifacts/resnet18_int8.mlpackage

# 4-bit k-means quantization (mlpackage only)
uv run timmx export coreml resnet18 \
  --pretrained \
  --convert-to mlprogram \
  --compute-precision float16 \
  --int4 \
  --output ./artifacts/resnet18_int4.mlpackage

# fp16 weight quantization (neuralnetwork)
uv run timmx export coreml resnet18 \
  --pretrained \
  --convert-to neuralnetwork \
  --half \
  --output ./artifacts/resnet18_half.mlmodel

# 8-bit k-means quantization (neuralnetwork)
uv run timmx export coreml resnet18 \
  --pretrained \
  --convert-to neuralnetwork \
  --int8 \
  --output ./artifacts/resnet18_int8.mlmodel
```

### LiteRT / TFLite

Supported modes: `fp32`, `fp16`, `dynamic-int8`, `int8`.

```bash
uv run timmx export litert resnet18 \
  --mode fp16 \
  --output ./artifacts/resnet18_fp16.tflite
```

INT8 with calibration data (point to an image directory — timm transforms are applied automatically):

```bash
uv run timmx export litert resnet18 \
  --mode int8 \
  --calibration-data ./my-images/ \
  --output ./artifacts/resnet18_int8.tflite
```

Limit the number of calibration images loaded:

```bash
uv run timmx export litert resnet18 \
  --mode int8 \
  --calibration-data ./my-images/ \
  --calibration-samples 64 \
  --output ./artifacts/resnet18_int8.tflite
```

For fine-tuned models with custom normalization, override calibration preprocessing with `--mean` / `--std`:

```bash
uv run timmx export litert resnet18 \
  --mode int8 \
  --calibration-data ./my-images/ \
  --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 \
  --output ./artifacts/resnet18_int8.tflite
```

For image-directory calibration, `--in-chans` currently supports only `1` or `3`; grayscale models
average RGB mean/std values down to one channel automatically.

A pre-saved torch tensor `(N, C, H, W)` is also accepted:

```bash
uv run timmx export litert resnet18 \
  --mode int8 \
  --calibration-data ./calibration.pt \
  --calibration-steps 8 \
  --output ./artifacts/resnet18_int8.tflite
```

Use `--random-calibration` to skip providing real data (not recommended for production):

```bash
uv run timmx export litert resnet18 \
  --mode int8 \
  --random-calibration \
  --output ./artifacts/resnet18_int8.tflite
```

NHWC input layout:

```bash
uv run timmx export litert resnet18 \
  --mode fp32 \
  --nhwc-input \
  --output ./artifacts/resnet18_nhwc.tflite
```

### ncnn

Exports via [pnnx](https://github.com/pnnx/pnnx) and writes a deployment-ready ncnn model directory containing `model.ncnn.param`, `model.ncnn.bin`, and `model_ncnn.py`. pnnx intermediate files are removed automatically.

```bash
uv run timmx export ncnn resnet18 \
  --pretrained \
  --output ./artifacts/resnet18_ncnn
```

Export without fp16 weight quantization:

```bash
uv run timmx export ncnn resnet18 \
  --pretrained \
  --no-fp16 \
  --output ./artifacts/resnet18_ncnn_fp32
```

### TensorRT

Requires an NVIDIA GPU with CUDA and the `tensorrt` package (`pip install tensorrt`).

```bash
uv run timmx export tensorrt resnet18 \
  --pretrained \
  --mode fp16 \
  --output ./artifacts/resnet18_fp16.engine
```

INT8 with calibration (image directory or torch tensor):

```bash
uv run timmx export tensorrt resnet18 \
  --pretrained \
  --mode int8 \
  --calibration-data ./my-images/ \
  --output ./artifacts/resnet18_int8.engine
```

Override calibration normalization for fine-tuned models with `--mean` / `--std`:

```bash
uv run timmx export tensorrt resnet18 \
  --pretrained \
  --mode int8 \
  --calibration-data ./my-images/ \
  --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 \
  --output ./artifacts/resnet18_int8.engine
```

Dynamic batch size:

```bash
uv run timmx export tensorrt resnet18 \
  --pretrained \
  --dynamic-batch \
  --batch-size 4 \
  --batch-min 1 \
  --batch-max 32 \
  --output ./artifacts/resnet18_dynamic.engine
```

### RKNN

Requires Linux (x86_64 or aarch64) and the `rknn-toolkit2` package. Exports via an ONNX intermediate for Rockchip NPUs.

Basic INT8 export for RK3588:

```bash
uv run timmx export rknn resnet18 \
  --pretrained \
  --target-platform rk3588 \
  --mode int8 \
  --calibration-data ./my-images/ \
  --output ./artifacts/resnet18.rknn
```

FP16 export (no quantization):

```bash
uv run timmx export rknn resnet18 \
  --pretrained \
  --target-platform rk3588 \
  --mode fp16 \
  --output ./artifacts/resnet18_fp16.rknn
```

INT8 with MMSE quantization algorithm (better accuracy, slower):

```bash
uv run timmx export rknn resnet18 \
  --pretrained \
  --target-platform rk3566 \
  --mode int8 \
  --quant-algorithm mmse \
  --calibration-data ./my-images/ \
  --output ./artifacts/resnet18_int8_mmse.rknn
```

> **Note:** Unlike other backends, RKNN handles normalization natively via its runtime config. For best INT8 calibration accuracy, omit `--normalize` and let RKNN handle normalization internally. RKNN does not support dynamic batch sizes — models are compiled with a fixed batch size. Calibration only accepts image directories (not tensor files or `--random-calibration`).

### ExecuTorch

Export with XNNPack delegation (default, runs on CPU across all platforms):

```bash
uv run timmx export executorch resnet18 \
  --pretrained \
  --output ./artifacts/resnet18.pte
```

CoreML delegation (macOS — targets Apple Neural Engine / GPU / CPU):

```bash
uv run timmx export executorch resnet18 \
  --pretrained \
  --delegate coreml \
  --output ./artifacts/resnet18_coreml.pte
```

CoreML with explicit fp32 compute precision (default is fp16):

```bash
uv run timmx export executorch resnet18 \
  --pretrained \
  --delegate coreml \
  --compute-precision float32 \
  --output ./artifacts/resnet18_coreml_fp32.pte
```

INT8 quantized with XNNPack:

```bash
uv run timmx export executorch resnet18 \
  --pretrained \
  --mode int8 \
  --calibration-data ./my-images/ \
  --output ./artifacts/resnet18_int8.pte
```

Override calibration normalization for fine-tuned models with `--mean` / `--std`:

```bash
uv run timmx export executorch resnet18 \
  --pretrained \
  --mode int8 \
  --calibration-data ./my-images/ \
  --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 \
  --output ./artifacts/resnet18_int8.pte
```

INT8 quantized with CoreML:

```bash
uv run timmx export executorch resnet18 \
  --pretrained \
  --delegate coreml \
  --mode int8 \
  --random-calibration \
  --output ./artifacts/resnet18_coreml_int8.pte
```

Dynamic batch size:

```bash
uv run timmx export executorch resnet18 \
  --pretrained \
  --dynamic-batch \
  --batch-size 2 \
  --output ./artifacts/resnet18_dynamic.pte
```

### torch.export

```bash
uv run timmx export torch-export resnet18 \
  --pretrained \
  --dynamic-batch \
  --batch-size 2 \
  --output ./artifacts/resnet18.pt2
```

> When using `--dynamic-batch`, set `--batch-size` to at least `2` so PyTorch can capture a symbolic batch dimension.

### TorchScript

```bash
uv run timmx export torchscript resnet18 \
  --pretrained \
  --output ./artifacts/resnet18.pt
```

Export with built-in normalization (model accepts unnormalized `[0, 1]` float input):

```bash
uv run timmx export torchscript resnet18 \
  --pretrained \
  --normalize \
  --output ./artifacts/resnet18_normalized.pt
```

For fine-tuned models with custom normalization, override with `--mean` / `--std`:

```bash
uv run timmx export torchscript resnet18 \
  --pretrained \
  --normalize \
  --mean 0.5 0.5 0.5 --std 0.5 0.5 0.5 \
  --output ./artifacts/resnet18_custom_norm.pt
```

Grayscale TorchScript export behaves the same as ONNX here: `--in-chans` is currently limited to
`1` or `3`, and RGB mean/std values are averaged down to one channel for `--in-chans 1`.

Use `torch.jit.script` instead of the default `trace`:

```bash
uv run timmx export torchscript resnet18 \
  --pretrained \
  --method script \
  --output ./artifacts/resnet18_scripted.pt
```

## Diagnostics

Run `timmx info <model>` to inspect any model's metadata, or `timmx doctor` to check your installation and see which backends are available:

```bash
timmx doctor
```

This shows the timmx version, Python/torch versions, and a table of backend availability with install hints for any missing dependencies.

## Roadmap

- [x] ONNX
- [x] Core ML
- [x] LiteRT / TFLite
- [x] ncnn
- [x] torch.export
- [x] TensorRT
- [x] TorchScript
- [x] ExecuTorch (XNNPack + CoreML delegates)
- [ ] OpenVINO
- [ ] TensorFlow (SavedModel / .pb)
- [ ] TensorFlow.js
- [ ] TFLite Edge TPU
- [x] RKNN
- [ ] MNN
- [ ] PaddlePaddle

## Development

```bash
uv sync --extra onnx --extra coreml --extra ncnn --group dev  # install extras + pytest
uvx ruff format .                                              # format
uvx ruff check .                                               # lint
uv run pytest                                                  # test
uv build                                                       # build
```

## Adding a New Backend

See [CONTRIBUTING.md](CONTRIBUTING.md) for a step-by-step guide on implementing and registering a new export backend.

## AI Disclaimer

This project is developed with the assistance of AI tools. The original export logic comes from various standalone scripts I wrote for exporting fine-tuned timm models to different deployment formats. The process of consolidating these scripts into a unified CLI tool has been aided by AI, with my oversight at every step, reviewing generated code, manually fixing issues during backend porting, and validating that exports produce correct results.
