from __future__ import annotations

import pytest
import timm
import torch
from timm.data import resolve_data_config

from timmx.errors import ConfigurationError
from timmx.export.common import (
    PrePostWrapper,
    create_timm_model,
    prepare_export,
    resolve_input_size,
    wrap_with_preprocessing,
)


def _make_simple_model(*, in_chans: int = 3) -> torch.nn.Module:
    """Create a small timm model for testing."""
    model = timm.create_model("resnet18", pretrained=False, exportable=True, in_chans=in_chans)
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


def test_prepostwrapper_softmax_without_normalization_matches_raw_logits_softmax() -> None:
    model = _make_simple_model()
    wrapper = PrePostWrapper(model, normalize=False, softmax=True)
    wrapper.eval()

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        expected = torch.softmax(model(x), dim=-1)
        actual = wrapper(x)

    assert torch.allclose(expected, actual, atol=1e-5)


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


def test_wrap_with_preprocessing_softmax_only_skips_normalization_when_requested() -> None:
    model = _make_simple_model()
    wrapped = wrap_with_preprocessing(model, normalize=False, softmax=True)
    assert isinstance(wrapped, PrePostWrapper)
    assert wrapped.normalize is False
    assert torch.equal(wrapped.mean, torch.zeros(1))
    assert torch.equal(wrapped.std, torch.ones(1))


def test_wrap_with_preprocessing_custom_mean_std() -> None:
    """Custom mean/std override timm config."""
    model = _make_simple_model()
    custom_mean = (0.5, 0.5, 0.5)
    custom_std = (0.5, 0.5, 0.5)
    wrapped = wrap_with_preprocessing(model, mean=custom_mean, std=custom_std)
    assert isinstance(wrapped, PrePostWrapper)

    mean_buf = wrapped.mean.squeeze().tolist()
    std_buf = wrapped.std.squeeze().tolist()
    for a, b in zip(mean_buf, custom_mean):
        assert abs(a - b) < 1e-6
    for a, b in zip(std_buf, custom_std):
        assert abs(a - b) < 1e-6


def test_prepostwrapper_custom_mean_std_output() -> None:
    """Wrapper with custom mean/std produces different output than default."""
    model = _make_simple_model()
    default_mean = (0.485, 0.456, 0.406)
    default_std = (0.229, 0.224, 0.225)
    custom_mean = (0.5, 0.5, 0.5)
    custom_std = (0.5, 0.5, 0.5)

    default_wrapper = PrePostWrapper(model, mean=default_mean, std=default_std)
    custom_wrapper = PrePostWrapper(model, mean=custom_mean, std=custom_std)
    default_wrapper.eval()
    custom_wrapper.eval()

    x = torch.rand(1, 3, 32, 32)
    with torch.no_grad():
        default_out = default_wrapper(x)
        custom_out = custom_wrapper(x)

    # Different normalization should produce different outputs
    assert not torch.allclose(default_out, custom_out, atol=1e-3)


def test_wrap_with_preprocessing_zero_mean_std() -> None:
    """Zero-valued mean/std should be used, not fall back to timm defaults."""
    model = _make_simple_model()
    zero_mean = (0.0, 0.0, 0.0)
    zero_std = (1.0, 1.0, 1.0)
    wrapped = wrap_with_preprocessing(model, mean=zero_mean, std=zero_std)
    assert isinstance(wrapped, PrePostWrapper)

    mean_buf = wrapped.mean.squeeze().tolist()
    std_buf = wrapped.std.squeeze().tolist()
    for a, b in zip(mean_buf, zero_mean):
        assert abs(a - b) < 1e-6, f"Zero mean not preserved: got {a}, expected {b}"
    for a, b in zip(std_buf, zero_std):
        assert abs(a - b) < 1e-6, f"Std not preserved: got {a}, expected {b}"


def test_wrap_with_preprocessing_grayscale_averages_rgb_stats() -> None:
    model = _make_simple_model(in_chans=1)
    custom_mean = (0.2, 0.4, 0.6)
    custom_std = (0.3, 0.6, 0.9)

    wrapped = wrap_with_preprocessing(model, mean=custom_mean, std=custom_std)

    assert tuple(wrapped.mean.shape) == (1, 1, 1, 1)
    assert tuple(wrapped.std.shape) == (1, 1, 1, 1)
    assert abs(float(wrapped.mean.item()) - (sum(custom_mean) / 3)) < 1e-6
    assert abs(float(wrapped.std.item()) - (sum(custom_std) / 3)) < 1e-6

    x = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        out = wrapped(x)
    assert out.shape == (1, 1000)


def test_wrap_with_preprocessing_preserves_wrapped_model_mode() -> None:
    train_model = _make_simple_model()
    train_model.train()
    wrapped_train = wrap_with_preprocessing(train_model)
    assert wrapped_train.training is True
    assert wrapped_train.model.training is True

    eval_model = _make_simple_model()
    eval_model.eval()
    wrapped_eval = wrap_with_preprocessing(eval_model)
    assert wrapped_eval.training is False
    assert wrapped_eval.model.training is False


def test_prepare_export_wrapped_model_stays_in_eval_mode(tmp_path) -> None:
    prep = prepare_export(
        model_name="resnet18",
        output=tmp_path / "out.pt",
        checkpoint=None,
        pretrained=False,
        num_classes=None,
        in_chans=None,
        batch_size=1,
        input_size=(3, 32, 32),
        device="cpu",
        normalize=True,
    )

    assert prep.model.training is False
    assert prep.model.model.training is False


def test_prepare_export_rejects_mean_std_without_wrapper_flags(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="--mean/--std require --normalize"):
        prepare_export(
            model_name="resnet18",
            output=tmp_path / "out.pt",
            checkpoint=None,
            pretrained=False,
            num_classes=None,
            in_chans=None,
            batch_size=1,
            input_size=(3, 32, 32),
            device="cpu",
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )


def test_prepare_export_softmax_only_matches_raw_model_softmax(tmp_path) -> None:
    seed = 321
    x = torch.rand(1, 3, 32, 32)

    torch.manual_seed(seed)
    reference_model = create_timm_model(
        "resnet18",
        pretrained=False,
        checkpoint=None,
        num_classes=None,
        in_chans=None,
    ).eval()

    torch.manual_seed(seed)
    prep = prepare_export(
        model_name="resnet18",
        output=tmp_path / "out.pt",
        checkpoint=None,
        pretrained=False,
        num_classes=None,
        in_chans=None,
        batch_size=1,
        input_size=(3, 32, 32),
        device="cpu",
        softmax=True,
    )

    with torch.no_grad():
        expected = torch.softmax(reference_model(x), dim=-1)
        actual = prep.model(x)

    assert prep.model.normalize is False
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_resolve_input_size_uses_model_in_chans_over_timm_config() -> None:
    model = _make_simple_model(in_chans=1)

    assert resolve_input_size(model, None) == (1, 224, 224)


def test_prepare_export_grayscale_wrapper_runs_forward_pass(tmp_path) -> None:
    prep = prepare_export(
        model_name="resnet18",
        output=tmp_path / "out.pt",
        checkpoint=None,
        pretrained=False,
        num_classes=None,
        in_chans=1,
        batch_size=1,
        input_size=(1, 32, 32),
        device="cpu",
        normalize=True,
    )

    assert tuple(prep.example_input.shape) == (1, 1, 32, 32)
    assert tuple(prep.model.mean.shape) == (1, 1, 1, 1)
    with torch.no_grad():
        out = prep.model(prep.example_input)
    assert out.shape == (1, 1000)


@pytest.mark.parametrize("bad_in_chans", [0, 2, 4])
def test_prepare_export_rejects_unsupported_in_chans(tmp_path, bad_in_chans: int) -> None:
    with pytest.raises(ConfigurationError, match="--in-chans input channels must be 1 or 3"):
        prepare_export(
            model_name="resnet18",
            output=tmp_path / "out.pt",
            checkpoint=None,
            pretrained=False,
            num_classes=None,
            in_chans=bad_in_chans,
            batch_size=1,
            input_size=(3, 32, 32),
            device="cpu",
        )


def test_prepare_export_rejects_input_size_channel_mismatch(tmp_path) -> None:
    with pytest.raises(ConfigurationError, match="--input-size channel count must match"):
        prepare_export(
            model_name="resnet18",
            output=tmp_path / "out.pt",
            checkpoint=None,
            pretrained=False,
            num_classes=None,
            in_chans=1,
            batch_size=1,
            input_size=(3, 32, 32),
            device="cpu",
            normalize=True,
        )
