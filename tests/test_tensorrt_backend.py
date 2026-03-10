import copy
from pathlib import Path

import pytest
import torch

from timmx.errors import ConfigurationError
from timmx.export.common import PreparedExport, wrap_with_preprocessing
from timmx.export.tensorrt_backend import TensorRTBackend


class _ConvModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class _ClassifierModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = torch.nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def _build_kwargs(
    output_path: Path,
    *,
    mode: str = "fp32",
    device: str = "cuda",
    workspace: int = 4,
    dynamic_batch: bool = False,
    batch_min: int = 1,
    batch_max: int = 16,
    batch_size: int = 1,
    opset: int = 18,
    calibration_data: Path | None = None,
    calibration_steps: int | None = None,
    calibration_samples: int | None = None,
    calibration_cache: Path | None = None,
    random_calibration: bool = False,
    keep_onnx: bool = False,
    verbose: bool = False,
    normalize: bool = False,
    softmax: bool = False,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
) -> dict:
    return {
        "model_name": "resnet18",
        "output": output_path,
        "checkpoint": None,
        "pretrained": False,
        "num_classes": None,
        "in_chans": None,
        "batch_size": batch_size,
        "input_size": (3, 16, 16),
        "device": device,
        "mode": mode,
        "workspace": workspace,
        "opset": opset,
        "dynamic_batch": dynamic_batch,
        "batch_min": batch_min,
        "batch_max": batch_max,
        "calibration_data": calibration_data,
        "calibration_steps": calibration_steps,
        "calibration_samples": calibration_samples,
        "calibration_cache": calibration_cache,
        "random_calibration": random_calibration,
        "keep_onnx": keep_onnx,
        "verbose": verbose,
        "normalize": normalize,
        "softmax": softmax,
        "mean": mean,
        "std": std,
    }


def _patch_model_helpers(monkeypatch: pytest.MonkeyPatch, model: torch.nn.Module) -> None:
    monkeypatch.setattr(
        "timmx.export.common.create_timm_model",
        lambda *_args, **_kwargs: model,
    )
    monkeypatch.setattr(
        "timmx.export.common.resolve_input_size",
        lambda _model, _requested: (3, 16, 16),
    )


class _FakeLogger:
    WARNING = 1
    VERBOSE = 2

    def __init__(self, level: int) -> None:
        self.level = level


class _FakeConfig:
    def __init__(self) -> None:
        self.flags: list[object] = []
        self.int8_calibrator: object | None = None
        self.profiles: list[object] = []
        self.workspace_limit: tuple[object, int] | None = None

    def set_memory_pool_limit(self, pool_type: object, size: int) -> None:
        self.workspace_limit = (pool_type, size)

    def set_flag(self, flag: object) -> None:
        self.flags.append(flag)

    def add_optimization_profile(self, profile: object) -> None:
        self.profiles.append(profile)


class _FakeProfile:
    def __init__(self) -> None:
        self.shapes: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = {}

    def set_shape(
        self,
        name: str,
        min_shape: tuple[int, ...],
        opt_shape: tuple[int, ...],
        max_shape: tuple[int, ...],
    ) -> None:
        self.shapes[name] = (min_shape, opt_shape, max_shape)


class _FakeBuilder:
    def __init__(self, logger: _FakeLogger) -> None:
        self.logger = logger

    def create_network(self, _flags: int) -> object:
        return object()

    def create_builder_config(self) -> _FakeConfig:
        return _FakeConfig()

    def create_optimization_profile(self) -> _FakeProfile:
        return _FakeProfile()

    def build_serialized_network(self, _network: object, _config: _FakeConfig) -> bytes:
        return b"engine"


class _FakeParser:
    def __init__(self, _network: object, _logger: _FakeLogger) -> None:
        self.num_errors = 0

    def parse(self, _payload: bytes) -> bool:
        return True

    def get_error(self, _index: int) -> str:
        return ""


class _FakeTRT:
    Logger = _FakeLogger
    Builder = _FakeBuilder
    OnnxParser = _FakeParser

    class NetworkDefinitionCreationFlag:
        EXPLICIT_BATCH = 0

    class MemoryPoolType:
        WORKSPACE = "workspace"

    class BuilderFlag:
        FP16 = "fp16"
        INT8 = "int8"

    class IInt8MinMaxCalibrator:
        pass


def _patch_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    output_path: Path,
    *,
    model: torch.nn.Module,
    batch_size: int = 1,
    capture_prepare: dict[str, object] | None = None,
    capture_export: dict[str, object] | None = None,
) -> None:
    example_input = torch.rand(batch_size, 3, 16, 16)

    def fake_prepare_export(**kwargs) -> PreparedExport:
        if capture_prepare is not None:
            capture_prepare.update(kwargs)

        export_model = copy.deepcopy(model).eval()
        if kwargs["normalize"] or kwargs["softmax"]:
            export_model = wrap_with_preprocessing(
                export_model,
                normalize=kwargs["normalize"],
                softmax=kwargs["softmax"],
                mean=kwargs["mean"],
                std=kwargs["std"],
            ).eval()

        return PreparedExport(
            model=export_model,
            example_input=example_input,
            resolved_input_size=(3, 16, 16),
            output_path=output_path,
            torch_device=torch.device("cpu"),
        )

    def fake_onnx_export(model_to_export, args, f, **kwargs) -> None:
        if capture_export is not None:
            capture_export["model"] = model_to_export
            capture_export["args"] = args
            capture_export["kwargs"] = kwargs
        Path(f).write_bytes(b"onnx")

    monkeypatch.setattr("timmx.export.tensorrt_backend._import_tensorrt", lambda: _FakeTRT())
    monkeypatch.setattr("timmx.export.tensorrt_backend._require_onnxscript", lambda: None)
    monkeypatch.setattr("timmx.export.tensorrt_backend.prepare_export", fake_prepare_export)
    monkeypatch.setattr(torch.onnx, "export", fake_onnx_export)


@pytest.fixture()
def _fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend CUDA is available so validation tests can reach TRT-specific checks."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)


# ---------------------------------------------------------------------------
# Validation tests (run on any platform, no GPU needed)
# ---------------------------------------------------------------------------


def test_tensorrt_rejects_cpu_device(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.engine", device="cpu")
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--device cuda"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_calibration_args_for_fp32(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.pt"
    torch.save(torch.randn(4, 3, 16, 16), calibration_path)
    kwargs = _build_kwargs(
        tmp_path / "model.engine",
        mode="fp32",
        calibration_data=calibration_path,
    )
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="only valid with --mode int8"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_calibration_args_for_fp16(tmp_path: Path) -> None:
    kwargs = _build_kwargs(
        tmp_path / "model.engine",
        mode="fp16",
        calibration_steps=5,
    )
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="only valid with --mode int8"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_invalid_workspace(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.engine", workspace=0)
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--workspace"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_invalid_opset(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.engine", opset=5)
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--opset"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_dynamic_batch_with_batch_size_1(tmp_path: Path) -> None:
    kwargs = _build_kwargs(
        tmp_path / "model.engine",
        dynamic_batch=True,
        batch_size=1,
    )
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--batch-size must be >= 2"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_batch_max_lt_batch_size(tmp_path: Path) -> None:
    kwargs = _build_kwargs(
        tmp_path / "model.engine",
        dynamic_batch=True,
        batch_size=8,
        batch_max=4,
    )
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--batch-max"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_cuda")
def test_tensorrt_rejects_batch_min_gt_batch_size(tmp_path: Path) -> None:
    kwargs = _build_kwargs(
        tmp_path / "model.engine",
        dynamic_batch=True,
        batch_size=2,
        batch_min=4,
    )
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--batch-min"):
        command(**kwargs)


def test_tensorrt_rejects_mean_std_without_wrapper_flags_outside_int8(tmp_path: Path) -> None:
    backend = TensorRTBackend()
    command = backend.create_command()
    with pytest.raises(
        ConfigurationError,
        match="--mean/--std require --normalize unless used for --mode int8 calibration",
    ):
        command(
            **_build_kwargs(
                tmp_path / "model.engine",
                mode="fp32",
                mean=(0.5, 0.25, 0.75),
                std=(0.125, 0.5, 0.25),
            )
        )


def test_tensorrt_int8_allows_mean_std_for_calibration_without_wrapper_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "model.engine"
    prepare_kwargs: dict[str, object] = {}
    calibrator_kwargs: dict[str, object] = {}
    _patch_fake_runtime(
        monkeypatch,
        output_path,
        model=_ConvModel().eval(),
        batch_size=2,
        capture_prepare=prepare_kwargs,
    )
    monkeypatch.setattr(
        "timmx.export.tensorrt_backend._create_calibrator",
        lambda **kwargs: calibrator_kwargs.update(kwargs) or object(),
    )

    TensorRTBackend().create_command()(
        **_build_kwargs(
            output_path,
            mode="int8",
            batch_size=2,
            random_calibration=True,
            mean=(0.5, 0.25, 0.75),
            std=(0.125, 0.5, 0.25),
        )
    )

    assert output_path.exists()
    assert prepare_kwargs["normalize"] is False
    assert prepare_kwargs["softmax"] is False
    assert prepare_kwargs["mean"] is None
    assert prepare_kwargs["std"] is None
    assert calibrator_kwargs["mean"] == (0.5, 0.25, 0.75)
    assert calibrator_kwargs["std"] == (0.125, 0.5, 0.25)
    assert calibrator_kwargs["normalize_images"] is True


def test_tensorrt_int8_wrapper_disables_image_normalization_for_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "model.engine"
    prepare_kwargs: dict[str, object] = {}
    calibrator_kwargs: dict[str, object] = {}
    _patch_fake_runtime(
        monkeypatch,
        output_path,
        model=_ConvModel().eval(),
        batch_size=2,
        capture_prepare=prepare_kwargs,
    )
    monkeypatch.setattr(
        "timmx.export.tensorrt_backend._create_calibrator",
        lambda **kwargs: calibrator_kwargs.update(kwargs) or object(),
    )

    TensorRTBackend().create_command()(
        **_build_kwargs(
            output_path,
            mode="int8",
            batch_size=2,
            random_calibration=True,
            normalize=True,
            softmax=True,
            mean=(0.5, 0.25, 0.75),
            std=(0.125, 0.5, 0.25),
        )
    )

    assert output_path.exists()
    assert prepare_kwargs["softmax"] is True
    assert prepare_kwargs["mean"] == (0.5, 0.25, 0.75)
    assert prepare_kwargs["std"] == (0.125, 0.5, 0.25)
    assert calibrator_kwargs["normalize_images"] is False


def test_tensorrt_int8_softmax_only_keeps_image_normalization_for_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "model.engine"
    calibrator_kwargs: dict[str, object] = {}
    _patch_fake_runtime(
        monkeypatch,
        output_path,
        model=_ConvModel().eval(),
        batch_size=2,
    )
    monkeypatch.setattr(
        "timmx.export.tensorrt_backend._create_calibrator",
        lambda **kwargs: calibrator_kwargs.update(kwargs) or object(),
    )

    TensorRTBackend().create_command()(
        **_build_kwargs(
            output_path,
            mode="int8",
            batch_size=2,
            random_calibration=True,
            softmax=True,
        )
    )

    assert output_path.exists()
    assert calibrator_kwargs["normalize_images"] is True


def test_tensorrt_fp32_wraps_preprocessing_and_softmax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "model.engine"
    prepare_kwargs: dict[str, object] = {}
    export_capture: dict[str, object] = {}
    reference_model = _ClassifierModel().eval()
    _patch_fake_runtime(
        monkeypatch,
        output_path,
        model=reference_model,
        capture_prepare=prepare_kwargs,
        capture_export=export_capture,
    )

    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    TensorRTBackend().create_command()(
        **_build_kwargs(
            output_path,
            mode="fp32",
            normalize=True,
            softmax=True,
            mean=mean,
            std=std,
        )
    )

    exported_model = export_capture["model"]
    example_input = export_capture["args"][0]
    expected = wrap_with_preprocessing(
        copy.deepcopy(reference_model).eval(),
        normalize=True,
        softmax=True,
        mean=mean,
        std=std,
    )(example_input)
    actual = exported_model(example_input)

    assert output_path.exists()
    assert prepare_kwargs["softmax"] is True
    assert prepare_kwargs["mean"] == mean
    assert prepare_kwargs["std"] == std
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


# ---------------------------------------------------------------------------
# GPU-dependent tests (skipped when CUDA or tensorrt is not available)
# ---------------------------------------------------------------------------

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TensorRT export requires NVIDIA GPU with CUDA",
)

try:
    import tensorrt as _trt  # noqa: F401

    _has_tensorrt = True
except ImportError:
    _has_tensorrt = False

requires_tensorrt = pytest.mark.skipif(
    not _has_tensorrt,
    reason="tensorrt package not installed",
)

try:
    import onnxscript as _onnxscript  # noqa: F401

    _has_onnxscript = True
except ImportError:
    _has_onnxscript = False

requires_onnxscript = pytest.mark.skipif(
    not _has_onnxscript,
    reason="onnxscript package not installed",
)


@requires_cuda
@requires_tensorrt
@requires_onnxscript
def test_export_tensorrt_fp32(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.engine"
    kwargs = _build_kwargs(output_path, mode="fp32")
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = TensorRTBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@requires_cuda
@requires_tensorrt
@requires_onnxscript
def test_export_tensorrt_fp16(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.engine"
    kwargs = _build_kwargs(output_path, mode="fp16")
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = TensorRTBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@requires_cuda
@requires_tensorrt
@requires_onnxscript
def test_export_tensorrt_dynamic_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.engine"
    kwargs = _build_kwargs(
        output_path,
        mode="fp32",
        dynamic_batch=True,
        batch_size=2,
        batch_min=1,
        batch_max=8,
    )
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = TensorRTBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@requires_cuda
@requires_tensorrt
@requires_onnxscript
def test_export_tensorrt_int8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.engine"
    kwargs = _build_kwargs(output_path, mode="int8", random_calibration=True)
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = TensorRTBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


@requires_cuda
@requires_tensorrt
@requires_onnxscript
def test_export_tensorrt_keep_onnx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.engine"
    kwargs = _build_kwargs(output_path, keep_onnx=True)
    _patch_model_helpers(monkeypatch, _ConvModel().eval())

    backend = TensorRTBackend()
    command = backend.create_command()
    command(**kwargs)

    assert output_path.exists()
    onnx_path = output_path.with_suffix(".onnx")
    assert onnx_path.exists()
