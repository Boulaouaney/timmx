# Common dev commands; `just` alone lists them.

extras := "--extra onnx --extra openvino --extra coreml --extra ncnn --extra coreai --extra executorch --extra litert"

default:
    @just --list

# Install every extra plus the dev group
sync:
    uv sync {{ extras }} --group dev

# Format Python and TOML in place
fmt:
    uvx ruff format .
    uvx ruff check --fix .
    uvx tombi format pyproject.toml

# Check formatting and lint without modifying anything
lint:
    uvx ruff format --check .
    uvx ruff check .
    uvx tombi format --check pyproject.toml
    uvx tombi lint pyproject.toml

# Run the suite; litert runs in its own process (see the caveat in AGENTS.md)
test:
    uv run pytest --ignore=tests/test_litert_backend.py
    uv run pytest tests/test_litert_backend.py

# Run one file or node id, e.g. `just test-one tests/test_cli.py::test_version_flag`
test-one target:
    uv run pytest {{ target }}

build:
    uv build

# Everything CI runs
check: lint test build
