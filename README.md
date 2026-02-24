# timmx

An extensible CLI and Python package for exporting [timm](https://github.com/huggingface/pytorch-image-models) models to various deployment formats. Born out of having too many one-off export scripts for fine-tuned timm models — `timmx` unifies them behind a single command-line interface with a plugin-based backend system.

## Supported Formats

| Format | Command | Output |
|--------|---------|--------|
| ONNX | `timmx export onnx` | `.onnx` |
| Core ML | `timmx export coreml` | `.mlpackage` / `.mlmodel` |
| LiteRT / TFLite | `timmx export litert` | `.tflite` |
| ncnn | `timmx export ncnn` | directory (`.param` + `.bin`) |
| TensorRT | `timmx export tensorrt` | `.engine` |
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
pip install 'timmx[onnx,coreml]'    # multiple backends
pip install 'timmx[all]'            # all non-platform-specific backends
```

TensorRT requires CUDA and must be installed separately:

```bash
pip install tensorrt  # Linux/Windows with CUDA only
```

Check which backends are available:

```bash
timmx doctor
```

## Quick Start

```bash
uv sync --all-extras --group dev
uv run timmx doctor
uv run timmx --help
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

### Core ML

```bash
uv run timmx export coreml resnet18 \
  --pretrained \
  --convert-to mlprogram \
  --compute-precision float16 \
  --output ./artifacts/resnet18.mlpackage
```

Flexible batch size:

```bash
uv run timmx export coreml resnet18 \
  --dynamic-batch \
  --batch-size 2 \
  --batch-upper-bound 8 \
  --output ./artifacts/resnet18_dynamic.mlpackage
```

### LiteRT / TFLite

Supported modes: `fp32`, `fp16`, `dynamic-int8`, `int8`.

```bash
uv run timmx export litert resnet18 \
  --mode fp16 \
  --output ./artifacts/resnet18_fp16.tflite
```

INT8 with calibration data:

```bash
# generate a calibration tensor
uv run python -c "import torch; torch.save(torch.randn(64, 3, 224, 224), 'calibration.pt')"

uv run timmx export litert resnet18 \
  --mode int8 \
  --calibration-data ./calibration.pt \
  --calibration-steps 8 \
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

INT8 with calibration:

```bash
uv run timmx export tensorrt resnet18 \
  --pretrained \
  --mode int8 \
  --calibration-data ./calibration.pt \
  --calibration-steps 8 \
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

Use `torch.jit.script` instead of the default `trace`:

```bash
uv run timmx export torchscript resnet18 \
  --pretrained \
  --method script \
  --output ./artifacts/resnet18_scripted.pt
```

## Diagnostics

Run `timmx doctor` to check your installation and see which backends are available:

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
- [ ] ExecuTorch (XNNPACK + more delegates TBD)
- [ ] OpenVINO
- [ ] TensorFlow (SavedModel / .pb)
- [ ] TensorFlow.js
- [ ] TFLite Edge TPU
- [ ] RKNN
- [ ] MNN
- [ ] PaddlePaddle

## Development

```bash
uv sync --all-extras --group dev  # install all extras + pytest
uvx ruff format .                 # format
uvx ruff check .                  # lint
uv run pytest                     # test
uv build                          # build
```

## Adding a New Backend

See [CONTRIBUTING.md](CONTRIBUTING.md) for a step-by-step guide on implementing and registering a new export backend.

## AI Disclaimer

This project is developed with the assistance of AI tools. The original export logic comes from various standalone scripts I wrote for exporting fine-tuned timm models to different deployment formats. The process of consolidating these scripts into a unified CLI tool has been aided by AI, with my oversight at every step, reviewing generated code, manually fixing issues during backend porting, and validating that exports produce correct results.
