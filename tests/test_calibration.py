from pathlib import Path

import pytest
import timm
import torch
from PIL import Image

from timmx.errors import ConfigurationError
from timmx.export.calibration import resolve_calibration_batches


@pytest.fixture()
def resnet18_model() -> torch.nn.Module:
    return timm.create_model("resnet18", pretrained=False).eval()


def test_random_calibration_defaults_to_single_batch() -> None:
    batches = resolve_calibration_batches(
        calibration_data=None,
        calibration_steps=None,
        batch_size=2,
        input_size=(3, 16, 16),
        device=torch.device("cpu"),
        random_calibration=True,
    )

    assert len(batches) == 1
    assert tuple(batches[0].shape) == (2, 3, 16, 16)


def test_random_calibration_requires_explicit_flag() -> None:
    with pytest.raises(ConfigurationError, match="calibration data"):
        resolve_calibration_batches(
            calibration_data=None,
            calibration_steps=None,
            batch_size=2,
            input_size=(3, 16, 16),
            device=torch.device("cpu"),
        )


def test_file_calibration_respects_requested_steps(tmp_path: Path) -> None:
    calibration_path = tmp_path / "calibration.pt"
    calibration_tensor = torch.randn(12, 3, 16, 16)
    torch.save(calibration_tensor, calibration_path)

    batches = resolve_calibration_batches(
        calibration_data=calibration_path,
        calibration_steps=3,
        batch_size=2,
        input_size=(3, 16, 16),
        device=torch.device("cpu"),
    )

    assert len(batches) == 3
    assert torch.equal(batches[0], calibration_tensor[0:2].to(dtype=torch.float32))
    assert torch.equal(batches[2], calibration_tensor[4:6].to(dtype=torch.float32))


def test_file_calibration_rejects_input_shape_mismatch(tmp_path: Path) -> None:
    calibration_path = tmp_path / "bad_calibration.pt"
    torch.save(torch.randn(4, 3, 8, 8), calibration_path)

    with pytest.raises(ConfigurationError):
        resolve_calibration_batches(
            calibration_data=calibration_path,
            calibration_steps=None,
            batch_size=2,
            input_size=(3, 16, 16),
            device=torch.device("cpu"),
        )


def test_image_dir_calibration(tmp_path: Path, resnet18_model: torch.nn.Module) -> None:
    """Loading calibration images from a directory applies timm transforms."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(8):
        img = Image.new("RGB", (64, 64), color=(i * 30, i * 20, i * 10))
        img.save(img_dir / f"img_{i:02d}.jpg")

    batches = resolve_calibration_batches(
        calibration_data=img_dir,
        calibration_steps=None,
        batch_size=4,
        input_size=(3, 224, 224),
        device=torch.device("cpu"),
        model=resnet18_model,
    )

    assert len(batches) == 2
    assert tuple(batches[0].shape) == (4, 3, 224, 224)
    assert batches[0].dtype == torch.float32


def test_image_dir_calibration_samples_limit(
    tmp_path: Path, resnet18_model: torch.nn.Module
) -> None:
    """--calibration-samples limits how many images are loaded."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(20):
        img = Image.new("RGB", (64, 64), color=(i * 10, 0, 0))
        img.save(img_dir / f"img_{i:02d}.png")

    batches = resolve_calibration_batches(
        calibration_data=img_dir,
        calibration_steps=None,
        batch_size=2,
        input_size=(3, 224, 224),
        device=torch.device("cpu"),
        model=resnet18_model,
        calibration_samples=6,
    )

    # 6 samples / batch_size 2 = 3 full batches
    assert len(batches) == 3


def test_image_dir_recursive(tmp_path: Path, resnet18_model: torch.nn.Module) -> None:
    """Images in subdirectories (ImageFolder layout) are found."""
    img_dir = tmp_path / "dataset"
    for cls in ("cat", "dog"):
        cls_dir = img_dir / cls
        cls_dir.mkdir(parents=True)
        for i in range(3):
            img = Image.new("RGB", (32, 32))
            img.save(cls_dir / f"{i}.jpg")

    batches = resolve_calibration_batches(
        calibration_data=img_dir,
        calibration_steps=None,
        batch_size=2,
        input_size=(3, 224, 224),
        device=torch.device("cpu"),
        model=resnet18_model,
    )

    # 6 images / batch_size 2 = 3 batches
    assert len(batches) == 3


def test_image_dir_empty_raises(tmp_path: Path, resnet18_model: torch.nn.Module) -> None:
    """Empty directory raises ConfigurationError."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    with pytest.raises(ConfigurationError, match="No image files found"):
        resolve_calibration_batches(
            calibration_data=empty_dir,
            calibration_steps=None,
            batch_size=1,
            input_size=(3, 224, 224),
            device=torch.device("cpu"),
            model=resnet18_model,
        )


def test_nonexistent_path_raises() -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        resolve_calibration_batches(
            calibration_data=Path("/nonexistent/path"),
            calibration_steps=None,
            batch_size=1,
            input_size=(3, 224, 224),
            device=torch.device("cpu"),
        )
