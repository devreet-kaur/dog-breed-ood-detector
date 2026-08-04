"""Tests for entropy-based OOD utilities."""

import math

import pytest
import torch

from src.ood_entropy import classify_ood, compute_entropy


def test_uniform_logits_have_maximum_entropy() -> None:
    logits = torch.zeros((2, 4))

    entropy = compute_entropy(logits)

    expected = torch.full((2,), math.log(4))
    assert torch.allclose(entropy, expected, atol=1e-6)


def test_confident_logits_have_low_entropy() -> None:
    logits = torch.tensor([[20.0, -10.0, -10.0]])

    entropy = compute_entropy(logits)

    assert entropy.shape == (1,)
    assert entropy.item() < 1e-4


def test_higher_temperature_increases_entropy() -> None:
    logits = torch.tensor([[5.0, 1.0, -2.0]])

    low_temperature = compute_entropy(logits, temperature=0.5)
    high_temperature = compute_entropy(logits, temperature=2.0)

    assert high_temperature.item() > low_temperature.item()


def test_entropy_returns_one_score_per_sample() -> None:
    logits = torch.randn(8, 120)

    entropy = compute_entropy(logits)

    assert entropy.shape == (8,)
    assert torch.isfinite(entropy).all()


@pytest.mark.parametrize("temperature", [0.0, -1.0])
def test_invalid_temperature_raises_error(temperature: float) -> None:
    logits = torch.randn(2, 3)

    with pytest.raises(ValueError, match="greater than zero"):
        compute_entropy(logits, temperature=temperature)


def test_invalid_logits_shape_raises_error() -> None:
    logits = torch.randn(3)

    with pytest.raises(ValueError, match="shape"):
        compute_entropy(logits)


def test_classify_ood_uses_strict_threshold() -> None:
    scores = torch.tensor([0.2, 0.5, 0.8])

    predictions = classify_ood(scores, threshold=0.5)

    expected = torch.tensor([False, False, True])
    assert torch.equal(predictions, expected)