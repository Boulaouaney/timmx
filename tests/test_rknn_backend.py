import copy
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from timmx.errors import ConfigurationError
from timmx.export.common import PreparedExport, wrap_with_preprocessing
from timmx.export.rknn_backend import RknnBackend


class _ConvModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _build_kwargs(
    output_path: Path,
    *,
    source: str = "torchscript",
    mode: str = "fp16",
    target_platform: str = "rk3588",
    quant_algorithm: str = "normal",
    quant_method: str = "channel",
    calibration_data: Path | None = None,
    calibration_samples: int | None = None,
    random_calibration: bool = False,
    opset: int | None = None,
    keep_onnx: bool = False,
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
        "batch_size": 1,
        "input_size": (3, 16, 16),
        "device": "cpu",
        "source": source,
        "target_platform": target_platform,
        "mode": mode,
        "quant_algorithm": quant_algorithm,
        "quant_method": quant_method,
        "calibration_data": calibration_data,
        "calibration_samples": calibration_samples,
        "random_calibration": random_calibration,
        "opset": opset,
        "keep_onnx": keep_onnx,
        "normalize": normalize,
        "softmax": softmax,
        "mean": mean,
        "std": std,
    }


# ---------------------------------------------------------------------------
# Fake RKNN runtime (same pattern as _FakeTRT in test_tensorrt_backend.py)
# ---------------------------------------------------------------------------


class _FakeRKNN:
    def __init__(self) -> None:
        self.config_kwargs: dict[str, object] = {}
        self.load_onnx_kwargs: dict[str, object] = {}
        self.load_pytorch_kwargs: dict[str, object] = {}
        self.build_kwargs: dict[str, object] = {}
        self.export_path: str | None = None
        self.released = False

    def config(self, **kwargs: object) -> int:
        self.config_kwargs = kwargs
        return 0

    def load_onnx(self, *, model: str, input_size_list: list[list[int]]) -> int:
        self.load_onnx_kwargs = {"model": model, "input_size_list": input_size_list}
        return 0

    def load_pytorch(self, *, model: str, input_size_list: list[list[int]]) -> int:
        self.load_pytorch_kwargs = {"model": model, "input_size_list": input_size_list}
        return 0

    def build(self, **kwargs: object) -> int:
        self.build_kwargs = kwargs
        return 0

    def export_rknn(self, path: str) -> int:
        self.export_path = path
        Path(path).write_bytes(b"rknn")
        return 0

    def release(self) -> None:
        self.released = True


def _patch_fake_runtime(
    monkeypatch: pytest.MonkeyPatch,
    output_path: Path,
    *,
    model: torch.nn.Module,
    capture_prepare: dict[str, object] | None = None,
    capture_export: dict[str, object] | None = None,
    fake_rknn: _FakeRKNN | None = None,
) -> _FakeRKNN:
    example_input = torch.rand(1, 3, 16, 16)
    rknn_instance = fake_rknn or _FakeRKNN()

    def fake_prepare_export(**kwargs: object) -> PreparedExport:
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

    def fake_onnx_export(model_to_export: object, args: object, f: str, **kwargs: object) -> None:
        if capture_export is not None:
            capture_export["model"] = model_to_export
            capture_export["args"] = args
            capture_export["kwargs"] = kwargs
        Path(f).write_bytes(b"onnx")

    def fake_jit_trace(model_to_trace: object, example_input: object) -> object:
        return MagicMock()

    def fake_jit_save(traced_model: object, path: str) -> None:
        Path(path).write_bytes(b"torchscript")

    monkeypatch.setattr("timmx.export.rknn_backend._import_rknn", lambda: lambda: rknn_instance)
    monkeypatch.setattr("timmx.export.rknn_backend.prepare_export", fake_prepare_export)
    monkeypatch.setattr(torch.onnx, "export", fake_onnx_export)
    monkeypatch.setattr(torch.jit, "trace", fake_jit_trace)
    monkeypatch.setattr(torch.jit, "save", fake_jit_save)
    monkeypatch.setattr("timmx.export.rknn_backend.sys", MagicMock(platform="linux"))
    monkeypatch.setattr("timmx.export.rknn_backend._check_onnx_version", lambda: None)

    return rknn_instance


# ---------------------------------------------------------------------------
# Validation tests (run on any platform, no RKNN needed)
# ---------------------------------------------------------------------------


@pytest.fixture()
def _fake_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("timmx.export.rknn_backend.sys", MagicMock(platform="linux"))


def test_rknn_rejects_non_linux_platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("timmx.export.rknn_backend.sys", MagicMock(platform="darwin"))
    kwargs = _build_kwargs(tmp_path / "model.rknn")
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="only supported on Linux"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_linux")
def test_rknn_rejects_random_calibration(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.rknn", random_calibration=True)
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--random-calibration"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_linux")
def test_rknn_rejects_opset_above_19(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.rknn", source="onnx", opset=20)
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--opset"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_linux")
def test_rknn_rejects_opset_below_7(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.rknn", source="onnx", opset=5)
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--opset"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_linux")
def test_rknn_rejects_opset_with_torchscript_source(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.rknn", source="torchscript", opset=19)
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--opset is only valid with --source onnx"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_linux")
def test_rknn_rejects_keep_onnx_with_torchscript_source(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.rknn", source="torchscript", keep_onnx=True)
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="--keep-onnx is only valid with --source onnx"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_linux")
def test_rknn_rejects_calibration_args_for_fp32(tmp_path: Path) -> None:
    cal_dir = tmp_path / "images"
    cal_dir.mkdir()
    kwargs = _build_kwargs(tmp_path / "model.rknn", mode="fp32", calibration_data=cal_dir)
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="only valid with --mode int8"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_linux")
def test_rknn_rejects_calibration_args_for_fp16(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.rknn", mode="fp16", calibration_samples=10)
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="only valid with --mode int8"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_linux")
def test_rknn_rejects_quant_algorithm_for_non_int8(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.rknn", mode="fp32", quant_algorithm="mmse")
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="only valid with --mode int8"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_linux")
def test_rknn_rejects_quant_method_for_non_int8(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.rknn", mode="fp16", quant_method="layer")
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="only valid with --mode int8"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_linux")
def test_rknn_int8_requires_calibration_data(tmp_path: Path) -> None:
    kwargs = _build_kwargs(tmp_path / "model.rknn", mode="int8")
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="calibration data"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_linux")
def test_rknn_rejects_tensor_file_calibration(tmp_path: Path) -> None:
    cal_file = tmp_path / "calibration.pt"
    torch.save(torch.randn(4, 3, 16, 16), cal_file)
    kwargs = _build_kwargs(tmp_path / "model.rknn", mode="int8", calibration_data=cal_file)
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(ConfigurationError, match="image directory"):
        command(**kwargs)


@pytest.mark.usefixtures("_fake_linux")
def test_rknn_rejects_mean_std_without_normalize_outside_int8(tmp_path: Path) -> None:
    backend = RknnBackend()
    command = backend.create_command()
    with pytest.raises(
        ConfigurationError,
        match="--mean/--std require --normalize",
    ):
        command(
            **_build_kwargs(
                tmp_path / "model.rknn",
                mode="fp32",
                mean=(0.5, 0.25, 0.75),
                std=(0.125, 0.5, 0.25),
            )
        )


# ---------------------------------------------------------------------------
# Fake-runtime tests — TorchScript source (default)
# ---------------------------------------------------------------------------


def test_rknn_torchscript_fp32_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.rknn"
    fake = _patch_fake_runtime(monkeypatch, output_path, model=_ConvModel().eval())

    RknnBackend().create_command()(**_build_kwargs(output_path, mode="fp32"))

    assert output_path.exists()
    assert fake.build_kwargs["do_quantization"] is False
    assert fake.load_pytorch_kwargs  # load_pytorch was called
    assert not fake.load_onnx_kwargs  # load_onnx was NOT called
    assert fake.released


def test_rknn_torchscript_fp16_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.rknn"
    fake = _patch_fake_runtime(monkeypatch, output_path, model=_ConvModel().eval())

    RknnBackend().create_command()(**_build_kwargs(output_path, mode="fp16"))

    assert output_path.exists()
    assert fake.build_kwargs["do_quantization"] is False
    assert fake.config_kwargs["target_platform"] == "rk3588"
    assert fake.load_pytorch_kwargs
    assert fake.released


def test_rknn_torchscript_int8_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.rknn"
    cal_dir = tmp_path / "images"
    cal_dir.mkdir()
    for i in range(5):
        (cal_dir / f"img_{i}.jpg").write_bytes(b"fake-jpg")

    fake = _patch_fake_runtime(monkeypatch, output_path, model=_ConvModel().eval())

    RknnBackend().create_command()(
        **_build_kwargs(output_path, mode="int8", calibration_data=cal_dir)
    )

    assert output_path.exists()
    assert fake.build_kwargs["do_quantization"] is True
    assert "dataset" in fake.build_kwargs
    assert fake.config_kwargs.get("quantized_algorithm") == "normal"
    assert fake.config_kwargs.get("quantized_method") == "channel"
    assert fake.load_pytorch_kwargs
    assert fake.released


def test_rknn_torchscript_normalization_passed_to_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "model.rknn"
    fake = _patch_fake_runtime(monkeypatch, output_path, model=_ConvModel().eval())

    RknnBackend().create_command()(**_build_kwargs(output_path, mode="fp32"))

    # When normalize=False, RKNN config should get model's mean/std scaled to [0,255]
    mean_values = fake.config_kwargs.get("mean_values")
    std_values = fake.config_kwargs.get("std_values")
    assert mean_values is not None
    assert std_values is not None
    assert all(v > 1.0 for v in mean_values[0])
    assert all(v > 1.0 for v in std_values[0])


def test_rknn_torchscript_normalize_flag_uses_identity_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "model.rknn"
    fake = _patch_fake_runtime(monkeypatch, output_path, model=_ConvModel().eval())

    RknnBackend().create_command()(**_build_kwargs(output_path, mode="fp32", normalize=True))

    mean_values = fake.config_kwargs.get("mean_values")
    std_values = fake.config_kwargs.get("std_values")
    assert mean_values == [[0.0, 0.0, 0.0]]
    assert std_values == [[255.0, 255.0, 255.0]]


def test_rknn_torchscript_release_called_on_build_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "model.rknn"
    fake = _FakeRKNN()
    fake.build = lambda **kwargs: -1  # type: ignore[assignment]
    _patch_fake_runtime(monkeypatch, output_path, model=_ConvModel().eval(), fake_rknn=fake)

    with pytest.raises(Exception):
        RknnBackend().create_command()(**_build_kwargs(output_path, mode="fp32"))

    assert fake.released


def test_rknn_torchscript_preprocessing_and_softmax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "model.rknn"
    prepare_kwargs: dict[str, object] = {}
    reference_model = _ConvModel().eval()
    _patch_fake_runtime(
        monkeypatch,
        output_path,
        model=reference_model,
        capture_prepare=prepare_kwargs,
    )

    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    RknnBackend().create_command()(
        **_build_kwargs(
            output_path,
            mode="fp32",
            normalize=True,
            softmax=True,
            mean=mean,
            std=std,
        )
    )

    assert output_path.exists()
    assert prepare_kwargs["softmax"] is True
    assert prepare_kwargs["mean"] == mean
    assert prepare_kwargs["std"] == std


def test_rknn_torchscript_int8_allows_mean_std_for_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "model.rknn"
    cal_dir = tmp_path / "images"
    cal_dir.mkdir()
    for i in range(3):
        (cal_dir / f"img_{i}.jpg").write_bytes(b"fake-jpg")

    prepare_kwargs: dict[str, object] = {}
    fake = _patch_fake_runtime(
        monkeypatch, output_path, model=_ConvModel().eval(), capture_prepare=prepare_kwargs
    )

    custom_mean = (0.5, 0.25, 0.75)
    custom_std = (0.125, 0.5, 0.25)
    RknnBackend().create_command()(
        **_build_kwargs(
            output_path,
            mode="int8",
            calibration_data=cal_dir,
            mean=custom_mean,
            std=custom_std,
        )
    )

    assert output_path.exists()
    assert prepare_kwargs["mean"] is None
    assert prepare_kwargs["std"] is None
    rknn_mean = fake.config_kwargs["mean_values"][0]
    assert abs(rknn_mean[0] - 0.5 * 255) < 0.01
    assert abs(rknn_mean[1] - 0.25 * 255) < 0.01


def test_rknn_torchscript_custom_target_platform(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "model.rknn"
    fake = _patch_fake_runtime(monkeypatch, output_path, model=_ConvModel().eval())

    RknnBackend().create_command()(
        **_build_kwargs(output_path, mode="fp32", target_platform="rk3566")
    )

    assert fake.config_kwargs["target_platform"] == "rk3566"


# ---------------------------------------------------------------------------
# Fake-runtime tests — ONNX source
# ---------------------------------------------------------------------------


def test_rknn_onnx_source_fp32_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.rknn"
    fake = _patch_fake_runtime(monkeypatch, output_path, model=_ConvModel().eval())

    RknnBackend().create_command()(**_build_kwargs(output_path, source="onnx", mode="fp32"))

    assert output_path.exists()
    assert fake.build_kwargs["do_quantization"] is False
    assert fake.load_onnx_kwargs  # load_onnx was called
    assert not fake.load_pytorch_kwargs  # load_pytorch was NOT called
    assert fake.released


def test_rknn_onnx_source_uses_dynamo_false(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "model.rknn"
    export_capture: dict[str, object] = {}
    _patch_fake_runtime(
        monkeypatch, output_path, model=_ConvModel().eval(), capture_export=export_capture
    )

    RknnBackend().create_command()(**_build_kwargs(output_path, source="onnx", mode="fp32"))

    assert output_path.exists()
    assert export_capture["kwargs"].get("dynamo") is False


def test_rknn_onnx_source_keep_onnx(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.rknn"
    _patch_fake_runtime(monkeypatch, output_path, model=_ConvModel().eval())

    RknnBackend().create_command()(**_build_kwargs(output_path, source="onnx", keep_onnx=True))

    assert output_path.exists()
    onnx_path = output_path.with_suffix(".onnx")
    assert onnx_path.exists()


def test_rknn_onnx_source_custom_opset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.rknn"
    export_capture: dict[str, object] = {}
    _patch_fake_runtime(
        monkeypatch, output_path, model=_ConvModel().eval(), capture_export=export_capture
    )

    RknnBackend().create_command()(
        **_build_kwargs(output_path, source="onnx", mode="fp32", opset=13)
    )

    assert export_capture["kwargs"].get("opset_version") == 13


def test_rknn_onnx_source_int8_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "model.rknn"
    cal_dir = tmp_path / "images"
    cal_dir.mkdir()
    for i in range(5):
        (cal_dir / f"img_{i}.jpg").write_bytes(b"fake-jpg")

    fake = _patch_fake_runtime(monkeypatch, output_path, model=_ConvModel().eval())

    RknnBackend().create_command()(
        **_build_kwargs(output_path, source="onnx", mode="int8", calibration_data=cal_dir)
    )

    assert output_path.exists()
    assert fake.build_kwargs["do_quantization"] is True
    assert "dataset" in fake.build_kwargs
    assert fake.load_onnx_kwargs  # load_onnx was called
    assert not fake.load_pytorch_kwargs  # load_pytorch was NOT called
    assert fake.released


def test_rknn_onnx_source_release_called_on_build_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "model.rknn"
    fake = _FakeRKNN()
    fake.build = lambda **kwargs: -1  # type: ignore[assignment]
    _patch_fake_runtime(monkeypatch, output_path, model=_ConvModel().eval(), fake_rknn=fake)

    with pytest.raises(Exception):
        RknnBackend().create_command()(**_build_kwargs(output_path, source="onnx", mode="fp32"))

    assert fake.released


# ---------------------------------------------------------------------------
# onnx version check tests
# ---------------------------------------------------------------------------


def test_rknn_rejects_onnx_version_too_new(monkeypatch: pytest.MonkeyPatch) -> None:
    from timmx.export import rknn_backend

    monkeypatch.setattr(rknn_backend, "MAX_ONNX_VERSION", (1, 18, 99))

    import onnx.version

    monkeypatch.setattr(onnx.version, "version", "1.19.0")

    with pytest.raises(ConfigurationError, match="onnx < 1.19"):
        rknn_backend._check_onnx_version()


def test_rknn_accepts_onnx_version_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnx.version

    from timmx.export import rknn_backend

    monkeypatch.setattr(onnx.version, "version", "1.16.2")

    # Should not raise
    rknn_backend._check_onnx_version()
