import asyncio
import platform
from pathlib import Path

import numpy as np
import pytest
import torch

from timmx.errors import ConfigurationError
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
        "verify": True,
        "normalize": False,
        "softmax": False,
        "mean": None,
        "std": None,
    }
    kwargs.update(overrides)
    return kwargs


def _run_asset(output_path: Path, x: torch.Tensor) -> np.ndarray:
    from coreai.runtime import AIModel, NDArray

    async def run() -> np.ndarray:
        model = await AIModel.load(output_path)
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
