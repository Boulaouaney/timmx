from pathlib import Path

import pytest
import torch

from timmx.errors import ConfigurationError
from timmx.export.common import create_timm_model, wrap_with_preprocessing
from timmx.export.executorch_backend import ExecuTorchBackend


def _build_kwargs(
    output_path: Path,
    *,
    delegate: str = "xnnpack",
    mode: str = "fp32",
    compute_precision: str | None = None,
    batch_size: int = 1,
    dynamic_batch: bool = False,
    calibration_data: Path | None = None,
    calibration_steps: int | None = None,
    calibration_samples: int | None = None,
    random_calibration: bool = False,
    per_channel: bool = True,
    normalize: bool = False,
    softmax: bool = False,
    mean: tuple[float, float, float] | None = None,
    std: tuple[float, float, float] | None = None,
    checkpoint: Path | None = None,
) -> dict:
    return {
        "model_name": "resnet18",
        "output": output_path,
        "checkpoint": checkpoint,
        "pretrained": False,
        "num_classes": None,
        "in_chans": None,
        "batch_size": batch_size,
        "input_size": (3, 32, 32),
        "device": "cpu",
        "delegate": delegate,
        "mode": mode,
        "compute_precision": compute_precision,
        "dynamic_batch": dynamic_batch,
        "calibration_data": calibration_data,
        "calibration_steps": calibration_steps,
        "calibration_samples": calibration_samples,
        "random_calibration": random_calibration,
        "per_channel": per_channel,
        "normalize": normalize,
        "softmax": softmax,
        "mean": mean,
        "std": std,
    }


# ---------------------------------------------------------------------------
# Validation tests (no executorch needed)
# ---------------------------------------------------------------------------


def test_rejects_compute_precision_with_xnnpack(tmp_path: Path) -> None:
    backend = ExecuTorchBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--compute-precision.*--delegate coreml"):
        command(
            **_build_kwargs(tmp_path / "m.pte", delegate="xnnpack", compute_precision="float16")
        )


def test_rejects_calibration_args_without_int8(tmp_path: Path) -> None:
    cal = tmp_path / "cal.pt"
    torch.save(torch.randn(4, 3, 32, 32), cal)
    backend = ExecuTorchBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="only valid with --mode int8"):
        command(**_build_kwargs(tmp_path / "m.pte", mode="fp32", calibration_data=cal))


def test_rejects_dynamic_batch_with_batch_size_1(tmp_path: Path) -> None:
    backend = ExecuTorchBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--batch-size >= 2"):
        command(**_build_kwargs(tmp_path / "m.pte", dynamic_batch=True, batch_size=1))


def test_rejects_mean_std_without_wrapper_flags_outside_int8(tmp_path: Path) -> None:
    backend = ExecuTorchBackend()
    command = backend.create_command()
    with pytest.raises(
        ConfigurationError,
        match="--mean/--std require --normalize unless used for --mode int8 calibration",
    ):
        command(
            **_build_kwargs(
                tmp_path / "m.pte",
                mode="fp32",
                mean=(0.5, 0.25, 0.75),
                std=(0.125, 0.5, 0.25),
            )
        )


def test_allows_mean_std_for_int8_calibration_without_wrapper_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _FakeProgram:
        def write_to_file(self, handle) -> None:
            handle.write(b"pte")

    def fake_export_quantized(**kwargs):
        captured.update(kwargs)
        return _FakeProgram()

    monkeypatch.setattr("timmx.export.executorch_backend._import_executorch", lambda: None)
    monkeypatch.setattr("timmx.export.executorch_backend._build_partitioner", lambda **_: [])
    monkeypatch.setattr("timmx.export.executorch_backend._export_quantized", fake_export_quantized)

    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    output = tmp_path / "model_int8_mean_std.pte"
    ExecuTorchBackend().create_command()(
        **_build_kwargs(
            output,
            mode="int8",
            random_calibration=True,
            mean=mean,
            std=std,
        )
    )

    assert output.exists()
    assert captured["mean"] == mean
    assert captured["std"] == std
    assert captured["normalize_calibration_images"] is True


def test_int8_wrapper_disables_image_normalization_for_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _FakeProgram:
        def write_to_file(self, handle) -> None:
            handle.write(b"pte")

    def fake_export_quantized(**kwargs):
        captured.update(kwargs)
        return _FakeProgram()

    monkeypatch.setattr("timmx.export.executorch_backend._import_executorch", lambda: None)
    monkeypatch.setattr("timmx.export.executorch_backend._build_partitioner", lambda **_: [])
    monkeypatch.setattr("timmx.export.executorch_backend._export_quantized", fake_export_quantized)

    output = tmp_path / "model_int8_wrapped.pte"
    ExecuTorchBackend().create_command()(
        **_build_kwargs(
            output,
            mode="int8",
            random_calibration=True,
            normalize=True,
            softmax=True,
            mean=(0.5, 0.25, 0.75),
            std=(0.125, 0.5, 0.25),
        )
    )

    assert output.exists()
    assert captured["normalize_calibration_images"] is False


def test_int8_softmax_only_keeps_image_normalization_for_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    class _FakeProgram:
        def write_to_file(self, handle) -> None:
            handle.write(b"pte")

    def fake_export_quantized(**kwargs):
        captured.update(kwargs)
        return _FakeProgram()

    monkeypatch.setattr("timmx.export.executorch_backend._import_executorch", lambda: None)
    monkeypatch.setattr("timmx.export.executorch_backend._build_partitioner", lambda **_: [])
    monkeypatch.setattr("timmx.export.executorch_backend._export_quantized", fake_export_quantized)

    output = tmp_path / "model_int8_softmax_only.pte"
    ExecuTorchBackend().create_command()(
        **_build_kwargs(
            output,
            mode="int8",
            random_calibration=True,
            softmax=True,
        )
    )

    assert output.exists()
    assert captured["normalize_calibration_images"] is True


# ---------------------------------------------------------------------------
# Runtime tests (skipped when executorch is not installed)
# ---------------------------------------------------------------------------

try:
    import executorch  # noqa: F401

    _has_executorch = True
except ImportError:
    _has_executorch = False

requires_executorch = pytest.mark.skipif(
    not _has_executorch,
    reason="executorch not installed",
)


@requires_executorch
def test_export_xnnpack_fp32(tmp_path: Path) -> None:
    output = tmp_path / "model.pte"
    backend = ExecuTorchBackend()
    command = backend.create_command()
    command(**_build_kwargs(output))
    assert output.exists()
    assert output.stat().st_size > 0


@requires_executorch
def test_export_xnnpack_fp32_wraps_preprocessing_and_softmax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import executorch.exir as exir

    captured: dict[str, object] = {}

    class _FakeExecuTorchProgram:
        def write_to_file(self, handle) -> None:
            handle.write(b"pte")

    class _FakeLoweredProgram:
        def __init__(self, exported_program) -> None:
            self._exported_program = exported_program

        def to_executorch(self) -> _FakeExecuTorchProgram:
            return _FakeExecuTorchProgram()

    def fake_lower(exported_program, partitioner=None, compile_config=None):
        captured["exported_program"] = exported_program
        captured["partitioner"] = partitioner
        captured["compile_config"] = compile_config
        return _FakeLoweredProgram(exported_program)

    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    reference_model = create_timm_model(
        "resnet18",
        pretrained=False,
        checkpoint=None,
        num_classes=None,
        in_chans=None,
    )
    checkpoint_path = tmp_path / "resnet18.pth"
    torch.save(reference_model.state_dict(), checkpoint_path)

    monkeypatch.setattr(exir, "to_edge_transform_and_lower", fake_lower)
    monkeypatch.setattr("timmx.export.executorch_backend._build_partitioner", lambda **_: [])

    output = tmp_path / "model_wrapped.pte"
    ExecuTorchBackend().create_command()(
        **_build_kwargs(
            output,
            checkpoint=checkpoint_path,
            normalize=True,
            softmax=True,
            mean=mean,
            std=std,
        )
    )

    exported_program = captured["exported_program"]
    wrapped_reference = wrap_with_preprocessing(
        reference_model,
        normalize=True,
        softmax=True,
        mean=mean,
        std=std,
    ).eval()
    example_input = torch.rand(1, 3, 32, 32)
    actual = exported_program.module()(example_input)
    expected = wrapped_reference(example_input)

    assert output.exists()
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)
    assert any("softmax" in str(node.target) for node in exported_program.graph.nodes)


@requires_executorch
def test_export_xnnpack_dynamic_batch(tmp_path: Path) -> None:
    output = tmp_path / "model_dynamic.pte"
    backend = ExecuTorchBackend()
    command = backend.create_command()
    command(**_build_kwargs(output, dynamic_batch=True, batch_size=2))
    assert output.exists()
    assert output.stat().st_size > 0


@requires_executorch
def test_pt2e_quantized_module_exported_in_eval_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import executorch.exir as exir

    captured: dict[str, object] = {}
    original_export = torch.export.export

    def tracking_export(model, *args, **kwargs):
        # The second torch.export.export call in _export_quantized passes
        # dynamic_shapes; internal PT2E calls do not.
        if "dynamic_shapes" in kwargs:
            captured["training"] = model.training
        return original_export(model, *args, **kwargs)

    class _FakeExecuTorchProgram:
        def write_to_file(self, handle) -> None:
            handle.write(b"pte")

    class _FakeLoweredProgram:
        def to_executorch(self) -> _FakeExecuTorchProgram:
            return _FakeExecuTorchProgram()

    def fake_lower(exported_program, **kwargs):
        return _FakeLoweredProgram()

    monkeypatch.setattr(torch.export, "export", tracking_export)
    monkeypatch.setattr(exir, "to_edge_transform_and_lower", fake_lower)
    monkeypatch.setattr("timmx.export.executorch_backend._build_partitioner", lambda **_: [])

    output = tmp_path / "model_int8_eval.pte"
    ExecuTorchBackend().create_command()(
        **_build_kwargs(output, mode="int8", random_calibration=True)
    )

    assert output.exists()
    assert captured["training"] is False


@requires_executorch
def test_export_xnnpack_int8(tmp_path: Path) -> None:
    output = tmp_path / "model_int8.pte"
    backend = ExecuTorchBackend()
    command = backend.create_command()
    command(**_build_kwargs(output, mode="int8", random_calibration=True))
    assert output.exists()
    assert output.stat().st_size > 0


@requires_executorch
def test_export_xnnpack_int8_with_calibration_data(tmp_path: Path) -> None:
    cal = tmp_path / "calibration.pt"
    torch.save(torch.randn(8, 3, 32, 32), cal)

    output = tmp_path / "model_int8_cal.pte"
    backend = ExecuTorchBackend()
    command = backend.create_command()
    command(
        **_build_kwargs(
            output, mode="int8", calibration_data=cal, calibration_steps=2, batch_size=2
        )
    )
    assert output.exists()
    assert output.stat().st_size > 0


# ---------------------------------------------------------------------------
# CoreML delegate tests (require macOS + executorch CoreML support)
# ---------------------------------------------------------------------------

try:
    from executorch.backends.apple.coreml.partition import CoreMLPartitioner  # noqa: F401

    _has_coreml_delegate = True
except (ImportError, ModuleNotFoundError):
    _has_coreml_delegate = False

requires_coreml_delegate = pytest.mark.skipif(
    not (_has_executorch and _has_coreml_delegate),
    reason="executorch CoreML delegate not available",
)


@requires_coreml_delegate
def test_export_coreml_default(tmp_path: Path) -> None:
    output = tmp_path / "model_coreml.pte"
    backend = ExecuTorchBackend()
    command = backend.create_command()
    command(**_build_kwargs(output, delegate="coreml"))
    assert output.exists()
    assert output.stat().st_size > 0


@requires_coreml_delegate
def test_export_coreml_fp32_precision(tmp_path: Path) -> None:
    output = tmp_path / "model_coreml_fp32.pte"
    backend = ExecuTorchBackend()
    command = backend.create_command()
    command(**_build_kwargs(output, delegate="coreml", compute_precision="float32"))
    assert output.exists()
    assert output.stat().st_size > 0


# CoreML int8 quantization needs CoreMLQuantizer
try:
    from executorch.backends.apple.coreml.quantizer import CoreMLQuantizer  # noqa: F401

    _has_coreml_quantizer = True
except (ImportError, ModuleNotFoundError):
    _has_coreml_quantizer = False

requires_coreml_quantizer = pytest.mark.skipif(
    not (_has_executorch and _has_coreml_delegate and _has_coreml_quantizer),
    reason="executorch CoreML quantizer not available",
)


@requires_coreml_quantizer
def test_export_coreml_int8(tmp_path: Path) -> None:
    output = tmp_path / "model_coreml_int8.pte"
    backend = ExecuTorchBackend()
    command = backend.create_command()
    command(**_build_kwargs(output, delegate="coreml", mode="int8", random_calibration=True))
    assert output.exists()
    assert output.stat().st_size > 0
