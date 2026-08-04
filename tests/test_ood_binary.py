"""Tests for the binary OOD CNN."""

import pytest
import torch

from src.ood_binary import BinaryCNN


def test_binary_cnn_output_shape() -> None:
    model = BinaryCNN(dropout=0.3)
    inputs = torch.randn(4, 3, 224, 224)

    outputs = model(inputs)

    assert outputs.shape == (4, 2)


def test_binary_cnn_supports_single_image() -> None:
    model = BinaryCNN()
    inputs = torch.randn(1, 3, 224, 224)

    outputs = model(inputs)

    assert outputs.shape == (1, 2)


def test_binary_cnn_outputs_finite_logits() -> None:
    model = BinaryCNN()
    inputs = torch.randn(2, 3, 224, 224)

    outputs = model(inputs)

    assert torch.isfinite(outputs).all()


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.2])
def test_binary_cnn_rejects_invalid_dropout(dropout: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        BinaryCNN(dropout=dropout)


def test_binary_cnn_accepts_zero_dropout() -> None:
    model = BinaryCNN(dropout=0.0)

    assert isinstance(model, BinaryCNN)