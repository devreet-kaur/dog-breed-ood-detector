"""Tests for entropy-based OOD utilities."""

import json
import math

import pytest
import torch

from src.ood_entropy import (
    calibrate_threshold,
    classify_ood,
    compute_entropy,
    save_threshold,
)


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
    

def test_calibrate_threshold_uses_target_quantile() -> None:
    scores = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])

    threshold = calibrate_threshold(scores, target_tpr=0.8)

    expected = torch.quantile(scores, 0.8).item()
    assert threshold == pytest.approx(expected)


def test_calibrated_threshold_accepts_expected_fraction_of_id_samples() -> None:
    scores = torch.linspace(0.0, 1.0, steps=101)

    threshold = calibrate_threshold(scores, target_tpr=0.95)
    accepted_fraction = (scores <= threshold).float().mean().item()

    assert accepted_fraction == pytest.approx(0.95, abs=0.02)


@pytest.mark.parametrize("target_tpr", [0.0, 1.0, -0.1, 1.1])
def test_invalid_target_tpr_raises_error(target_tpr: float) -> None:
    scores = torch.tensor([0.1, 0.2])

    with pytest.raises(ValueError, match="between 0 and 1"):
        calibrate_threshold(scores, target_tpr=target_tpr)


def test_empty_entropy_scores_raise_error() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        calibrate_threshold(torch.tensor([]))


def test_nonfinite_entropy_scores_raise_error() -> None:
    scores = torch.tensor([0.1, float("nan")])

    with pytest.raises(ValueError, match="finite"):
        calibrate_threshold(scores)


def test_save_threshold_writes_expected_json(tmp_path) -> None:
    output_path = tmp_path / "entropy_threshold.json"

    save_threshold(
        threshold=1.234,
        output_path=output_path,
        target_tpr=0.95,
        num_validation_samples=3084,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))

    assert saved == {
        "threshold": 1.234,
        "target_tpr": 0.95,
        "num_validation_samples": 3084,
    }    