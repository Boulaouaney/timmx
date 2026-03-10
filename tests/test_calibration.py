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


@pytest.fixture()
def resnet18_grayscale_model() -> torch.nn.Module:
    return timm.create_model("resnet18", pretrained=False, exportable=True, in_chans=1).eval()


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


@pytest.mark.parametrize("bad_value", [0, -1, -5])
def test_calibration_samples_rejects_invalid_values(bad_value: int) -> None:
    with pytest.raises(ConfigurationError, match="--calibration-samples must be >= 1"):
        resolve_calibration_batches(
            calibration_data=Path("."),
            calibration_steps=None,
            batch_size=1,
            input_size=(3, 224, 224),
            device=torch.device("cpu"),
            calibration_samples=bad_value,
        )


def test_image_dir_calibration_custom_mean_std(
    tmp_path: Path, resnet18_model: torch.nn.Module
) -> None:
    """Custom mean/std override timm config for calibration images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(4):
        img = Image.new("RGB", (64, 64), color=(i * 30, i * 20, i * 10))
        img.save(img_dir / f"img_{i:02d}.jpg")

    custom_mean = (0.5, 0.5, 0.5)
    custom_std = (0.5, 0.5, 0.5)

    batches_custom = resolve_calibration_batches(
        calibration_data=img_dir,
        calibration_steps=None,
        batch_size=2,
        input_size=(3, 224, 224),
        device=torch.device("cpu"),
        model=resnet18_model,
        mean=custom_mean,
        std=custom_std,
    )

    batches_default = resolve_calibration_batches(
        calibration_data=img_dir,
        calibration_steps=None,
        batch_size=2,
        input_size=(3, 224, 224),
        device=torch.device("cpu"),
        model=resnet18_model,
    )

    # Custom normalization should produce different values than default
    assert not torch.allclose(batches_custom[0], batches_default[0], atol=1e-3)


def test_image_dir_calibration_can_skip_normalization(
    tmp_path: Path, resnet18_model: torch.nn.Module
) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for index in range(2):
        Image.new("RGB", (224, 224), color=(128, 64, 192)).save(img_dir / f"img_{index:02d}.png")

    batches = resolve_calibration_batches(
        calibration_data=img_dir,
        calibration_steps=None,
        batch_size=2,
        input_size=(3, 224, 224),
        device=torch.device("cpu"),
        model=resnet18_model,
        normalize_images=False,
    )

    channel_means = batches[0].mean(dim=(0, 2, 3))
    expected = torch.tensor([128 / 255, 64 / 255, 192 / 255], dtype=torch.float32)
    assert torch.allclose(channel_means, expected, atol=1e-3)


def test_image_dir_calibration_grayscale_model(
    tmp_path: Path, resnet18_grayscale_model: torch.nn.Module
) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for i in range(4):
        img = Image.new("RGB", (64, 64), color=(i * 40, i * 30, i * 20))
        img.save(img_dir / f"img_{i:02d}.jpg")

    batches = resolve_calibration_batches(
        calibration_data=img_dir,
        calibration_steps=None,
        batch_size=2,
        input_size=(1, 32, 32),
        device=torch.device("cpu"),
        model=resnet18_grayscale_model,
    )

    assert len(batches) == 2
    assert tuple(batches[0].shape) == (2, 1, 32, 32)
    assert batches[0].dtype == torch.float32


def test_nonexistent_path_raises() -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        resolve_calibration_batches(
            calibration_data=Path("/nonexistent/path"),
            calibration_steps=None,
            batch_size=1,
            input_size=(3, 224, 224),
            device=torch.device("cpu"),
        )
