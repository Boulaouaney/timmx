from __future__ import annotations

import timm
import torch
from timm.data import resolve_data_config

from timmx.export.common import PrePostWrapper, wrap_with_preprocessing


def _make_simple_model() -> torch.nn.Module:
    """Create a small timm model for testing."""
    model = timm.create_model("resnet18", pretrained=False, exportable=True)
    model.eval()
    return model


def test_prepostwrapper_normalizes_input() -> None:
    """PrePostWrapper output matches manual normalization."""
    model = _make_simple_model()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    wrapper = PrePostWrapper(model, mean=mean, std=std, softmax=False)
    wrapper.eval()

    x = torch.rand(1, 3, 32, 32)

    # Manual normalization
    mean_t = torch.tensor(mean).reshape(1, -1, 1, 1)
    std_t = torch.tensor(std).reshape(1, -1, 1, 1)
    x_norm = (x - mean_t) / std_t

    with torch.no_grad():
        expected = model(x_norm)
        actual = wrapper(x)

    assert torch.allclose(expected, actual, atol=1e-5), "Wrapper output should match manual norm"


def test_prepostwrapper_softmax_sums_to_one() -> None:
    """With softmax=True, output probabilities sum to ~1.0."""
    model = _make_simple_model()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    wrapper = PrePostWrapper(model, mean=mean, std=std, softmax=True)
    wrapper.eval()

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        out = wrapper(x)

    assert torch.allclose(out.sum(dim=-1), torch.tensor([1.0]), atol=1e-5), (
        "Softmax output should sum to 1.0"
    )


def test_prepostwrapper_no_softmax_differs() -> None:
    """Without softmax, output is raw logits (not summing to 1)."""
    model = _make_simple_model()
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    wrapper = PrePostWrapper(model, mean=mean, std=std, softmax=False)
    wrapper.eval()

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        out = wrapper(x)

    # Raw logits almost certainly don't sum to 1
    assert not torch.allclose(out.sum(dim=-1), torch.tensor([1.0]), atol=1e-3)


def test_wrap_with_preprocessing_uses_timm_config() -> None:
    """wrap_with_preprocessing picks up mean/std from the timm model config."""
    model = _make_simple_model()
    config = resolve_data_config(model=model)
    expected_mean = config["mean"]
    expected_std = config["std"]

    wrapped = wrap_with_preprocessing(model, softmax=False)
    assert isinstance(wrapped, PrePostWrapper)

    # Check buffers match timm config
    mean_buf = wrapped.mean.squeeze().tolist()
    std_buf = wrapped.std.squeeze().tolist()
    for a, b in zip(mean_buf, expected_mean):
        assert abs(a - b) < 1e-6
    for a, b in zip(std_buf, expected_std):
        assert abs(a - b) < 1e-6


def test_wrap_with_preprocessing_softmax() -> None:
    """wrap_with_preprocessing with softmax=True enables softmax."""
    model = _make_simple_model()
    wrapped = wrap_with_preprocessing(model, softmax=True)
    assert isinstance(wrapped, PrePostWrapper)
    assert wrapped.softmax is True
