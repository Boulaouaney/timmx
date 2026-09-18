import os

# Wide help output so option names are not truncated in CLI help tests on narrow runners.
# Typer reads this at import time, so it must be set before any test module imports typer.
os.environ.setdefault("TERMINAL_WIDTH", "200")

import pytest  # noqa: E402
import torch  # noqa: E402

_MEAN = (0.5, 0.25, 0.75)
_STD = (0.125, 0.5, 0.25)

# (export kwargs, expected resolve_calibration_batches kwargs) for the --mode int8 rules shared by
# executorch, litert and tensorrt: --mean/--std without --normalize only steer calibration image
# preprocessing; with --normalize the wrapper normalizes, so calibration images stay raw; --softmax
# alone leaves image normalization on.
CALIBRATION_NORMALIZATION_CASES = [
    pytest.param(
        ({"mean": _MEAN, "std": _STD}, {"mean": _MEAN, "std": _STD, "normalize_images": True}),
        id="mean-std-without-wrapper",
    ),
    pytest.param(
        (
            {"normalize": True, "softmax": True, "mean": _MEAN, "std": _STD},
            {"normalize_images": False},
        ),
        id="wrapper-disables-image-normalization",
    ),
    pytest.param(
        ({"softmax": True}, {"normalize_images": True}),
        id="softmax-only-keeps-image-normalization",
    ),
]


@pytest.fixture(params=CALIBRATION_NORMALIZATION_CASES)
def calibration_case(request: pytest.FixtureRequest) -> tuple[dict[str, object], dict[str, object]]:
    """One (export kwargs, expected resolve_calibration_batches kwargs) case from the list above."""
    return request.param


@pytest.fixture
def calibration_capture(monkeypatch: pytest.MonkeyPatch):
    """Replace a backend module's resolve_calibration_batches and expose the kwargs it received."""

    def install(module: str, batch: torch.Tensor) -> dict[str, object]:
        captured: dict[str, object] = {}

        def fake(**kwargs: object) -> list[torch.Tensor]:
            captured.update(kwargs)
            return [batch]

        monkeypatch.setattr(f"{module}.resolve_calibration_batches", fake)
        return captured

    return install
