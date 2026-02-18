# timmx

`timmx` is an extensible CLI and Python package for exporting `timm` models to deployment
formats.

Current format support:
- Core ML (`timmx export coreml ...`)
- LiteRT / TFLite (`timmx export litert ...`)
- ONNX (`timmx export onnx ...`)
- torch.export archive (`timmx export torch-export ...`)

## Requirements

- Python `>=3.11`
- [`uv`](https://docs.astral.sh/uv/)

## Quick Start

```bash
uv sync --group dev
uv run timmx --help
```

### Export a pretrained model to ONNX

```bash
uv run timmx export onnx resnet18 --pretrained --output ./artifacts/resnet18.onnx
```

### Export a model to Core ML

```bash
uv run timmx export coreml resnet18 \
  --pretrained \
  --convert-to mlprogram \
  --compute-precision float16 \
  --output ./artifacts/resnet18.mlpackage
```

For flexible batch in Core ML:

```bash
uv run timmx export coreml resnet18 \
  --dynamic-batch \
  --batch-size 2 \
  --batch-upper-bound 8 \
  --output ./artifacts/resnet18_dynamic.mlpackage
```

### Export a model to LiteRT / TFLite

```bash
uv run timmx export litert resnet18 \
  --mode fp32 \
  --output ./artifacts/resnet18.tflite
```

LiteRT modes:
- `fp32`
- `fp16`
- `dynamic-int8`
- `int8`

Example `fp16`:

```bash
uv run timmx export litert resnet18 \
  --mode fp16 \
  --output ./artifacts/resnet18_fp16.tflite
```

Example `int8`:

```bash
uv run timmx export litert resnet18 \
  --mode int8 \
  --output ./artifacts/resnet18_int8.tflite
```

Use a calibration tensor file for quantized modes:

```bash
uv run python - <<'PY'
import torch
torch.save(torch.randn(64, 3, 224, 224), "calibration.pt")
PY

uv run timmx export litert resnet18 \
  --mode int8 \
  --calibration-data ./calibration.pt \
  --calibration-steps 8 \
  --output ./artifacts/resnet18_int8_calibrated.tflite
```

Enable NHWC input layout (first input only):

```bash
uv run timmx export litert resnet18 \
  --mode fp32 \
  --nhwc-input \
  --output ./artifacts/resnet18_nhwc.tflite
```

### Export a fine-tuned checkpoint to ONNX

```bash
uv run timmx export onnx resnet18 \
  --checkpoint ./checkpoints/model.pth \
  --input-size 3 224 224 \
  --dynamic-batch \
  --output ./artifacts/resnet18_finetuned.onnx
```

### Export a model with `torch.export` (`.pt2`)

```bash
uv run timmx export torch-export resnet18 \
  --pretrained \
  --dynamic-batch \
  --batch-size 2 \
  --output ./artifacts/resnet18.pt2
```

When using `--dynamic-batch` with `torch-export`, set `--batch-size` to at least `2` so
PyTorch can capture a symbolic batch dimension.

## Roadmap

- [x] ONNX
- [x] Core ML
- [x] LiteRT / TFLite
- [x] torch.export
- [ ] ExecuTorch (XNNPACK + more delegates TBD)
- [ ] OpenVINO
- [ ] TensorRT
- [ ] TensorFlow (SavedModel / .pb)
- [ ] TensorFlow.js
- [ ] TFLite Edge TPU
- [ ] NCNN
- [ ] RKNN
- [ ] MNN
- [ ] PaddlePaddle
- [ ] TorchScript (legacy)

## Development Commands

```bash
uv sync --group dev
uvx ruff format .
uvx ruff check .
uv run pytest
uv build
```

## Extending with New Formats

Backends are isolated under `src/timmx/export/`. Add a new backend by implementing the
`ExportBackend` contract and registering it in `create_builtin_registry()`.

Detailed contributor guidance is in `AGENTS.md`.
