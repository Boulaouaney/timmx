import asyncio
import platform
from pathlib import Path

import numpy as np
import pytest
import torch

from timmx.errors import ConfigurationError
from timmx.export import coreai_backend
from timmx.export.common import create_timm_model, wrap_with_preprocessing
from timmx.export.coreai_backend import CoreAIBackend

pytest.importorskip("coreai_torch")

# The Core AI runtime only executes on Apple platforms; conversion works everywhere.
requires_runtime = pytest.mark.skipif(
    platform.system() != "Darwin", reason="Core AI inference requires macOS"
)


def _build_kwargs(output_path: Path, **overrides: object) -> dict:
    kwargs: dict[str, object] = {
        "model_name": "resnet18",
        "output": output_path,
        "checkpoint": None,
        "pretrained": False,
        "num_classes": None,
        "in_chans": None,
        "batch_size": 1,
        "input_size": (3, 32, 32),
        "dynamic_batch": False,
        "device": "cpu",
        "half": False,
        "verify": True,
        "normalize": False,
        "softmax": False,
        "mean": None,
        "std": None,
    }
    kwargs.update(overrides)
    return kwargs


def _run_asset(output_path: Path, x: torch.Tensor, *, neural_engine: bool = False) -> np.ndarray:
    from coreai.runtime import AIModel, ComputeUnitKind, NDArray, SpecializationOptions

    options = None
    if neural_engine:
        options = SpecializationOptions.from_preferred_compute_unit_kind(
            ComputeUnitKind.neural_engine()
        )

    async def run() -> np.ndarray:
        model = await AIModel.load(output_path, options)
        function = model.load_function("main")
        assert function.desc.input_names == ["input"]
        assert function.desc.output_names == ["output"]
        outputs = await function({"input": NDArray(x)})
        return outputs["output"].numpy()

    return asyncio.run(run())


def test_export_coreai_writes_asset_and_verifies(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18.aimodel"
    CoreAIBackend().create_command()(**_build_kwargs(output_path))

    assert output_path.is_dir()
    assert {p.name for p in output_path.iterdir()} >= {"main.mlirb", "metadata.json"}


def test_rejects_non_aimodel_output(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match=".aimodel"):
        CoreAIBackend().create_command()(**_build_kwargs(tmp_path / "resnet18.mlpackage"))


def test_rejects_dynamic_batch_with_batch_size_1(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="--batch-size >= 2"):
        CoreAIBackend().create_command()(
            **_build_kwargs(tmp_path / "resnet18.aimodel", dynamic_batch=True)
        )


@requires_runtime
def test_export_coreai_names_io_and_matches_pytorch(tmp_path: Path) -> None:
    seed = 123
    x = torch.rand(1, 3, 32, 32)

    torch.manual_seed(seed)
    reference = create_timm_model(
        "resnet18", pretrained=False, checkpoint=None, num_classes=None, in_chans=None
    ).eval()

    output_path = tmp_path / "resnet18.aimodel"
    torch.manual_seed(seed)
    CoreAIBackend().create_command()(**_build_kwargs(output_path))

    np.testing.assert_allclose(
        _run_asset(output_path, x), reference(x).detach().numpy(), atol=1e-4, rtol=1e-4
    )


def test_half_captures_float16_model_with_fused_attention(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    capture_program = coreai_backend.capture_program

    def spy(model: torch.nn.Module, example_input: torch.Tensor, **kwargs: object) -> object:
        captured.update(model=model, example_input=example_input)
        return capture_program(model, example_input, **kwargs)

    monkeypatch.setattr(coreai_backend, "capture_program", spy)
    CoreAIBackend().create_command()(
        **_build_kwargs(
            tmp_path / "vit.aimodel",
            model_name="vit_tiny_patch16_224",
            input_size=None,
            half=True,
            verify=False,
        )
    )

    model = captured["model"]
    assert captured["example_input"].dtype == torch.float16
    assert next(model.parameters()).dtype == torch.float16
    attention = [m for m in model.modules() if hasattr(m, "fused_attn")]
    assert attention and all(m.fused_attn for m in attention)


@requires_runtime
def test_half_export_has_float16_io_and_matches_pytorch(tmp_path: Path) -> None:
    seed = 789
    x = torch.rand(1, 3, 32, 32)

    torch.manual_seed(seed)
    reference = create_timm_model(
        "resnet18", pretrained=False, checkpoint=None, num_classes=None, in_chans=None
    ).eval()

    output_path = tmp_path / "resnet18_half.aimodel"
    torch.manual_seed(seed)
    CoreAIBackend().create_command()(**_build_kwargs(output_path, half=True))

    actual = _run_asset(output_path, x.half(), neural_engine=True)
    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, reference(x).detach().numpy(), atol=1e-2, rtol=1e-2)


@requires_runtime
def test_dynamic_batch_accepts_other_batch_sizes(tmp_path: Path) -> None:
    output_path = tmp_path / "resnet18_dynamic.aimodel"
    CoreAIBackend().create_command()(**_build_kwargs(output_path, batch_size=2, dynamic_batch=True))

    for batch_size in (1, 3):
        actual = _run_asset(output_path, torch.rand(batch_size, 3, 32, 32))
        assert actual.shape == (batch_size, 1000)


@requires_runtime
def test_export_coreai_normalize_softmax_matches_wrapped_pytorch(tmp_path: Path) -> None:
    seed = 456
    mean = (0.5, 0.25, 0.75)
    std = (0.125, 0.5, 0.25)
    x = torch.rand(2, 3, 32, 32)

    torch.manual_seed(seed)
    reference = create_timm_model(
        "resnet18", pretrained=False, checkpoint=None, num_classes=None, in_chans=None
    ).eval()
    wrapped = wrap_with_preprocessing(
        reference, normalize=True, softmax=True, mean=mean, std=std
    ).eval()

    output_path = tmp_path / "resnet18_wrapped.aimodel"
    torch.manual_seed(seed)
    CoreAIBackend().create_command()(
        **_build_kwargs(output_path, batch_size=2, normalize=True, softmax=True, mean=mean, std=std)
    )

    actual = _run_asset(output_path, x)
    np.testing.assert_allclose(actual, wrapped(x).detach().numpy(), atol=1e-4, rtol=1e-4)
    np.testing.assert_allclose(actual.sum(axis=-1), np.ones(2), atol=1e-4, rtol=0)
