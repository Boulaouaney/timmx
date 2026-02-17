# timmx

`timmx` is an extensible CLI and Python package for exporting `timm` models to deployment
formats.

Current format support:
- ONNX (`timmx export onnx ...`)

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

### Export a fine-tuned checkpoint to ONNX

```bash
uv run timmx export onnx resnet18 \
  --checkpoint ./checkpoints/model.pth \
  --input-size 3 224 224 \
  --dynamic-batch \
  --output ./artifacts/resnet18_finetuned.onnx
```

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

