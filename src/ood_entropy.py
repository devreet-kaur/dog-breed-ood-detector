"""Entropy-based out-of-distribution detection utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_entropy(
    logits: torch.Tensor,
    temperature: float = 1.0,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Compute predictive entropy from unnormalized model logits.

    Higher entropy indicates greater uncertainty and can provide evidence
    that an input is out of distribution.

    Args:
        logits: Tensor of shape ``(batch_size, num_classes)``.
        temperature: Positive softmax temperature.
        epsilon: Numerical-stability constant used before taking logarithms.

    Returns:
        One entropy score per sample, with shape ``(batch_size,)``.

    Raises:
        ValueError: If the input shape or temperature is invalid.
    """
    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape (batch_size, num_classes), "
            f"but received {tuple(logits.shape)}"
        )

    if logits.shape[1] < 2:
        raise ValueError("logits must contain at least two classes")

    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")

    probabilities = F.softmax(logits / temperature, dim=1)
    log_probabilities = torch.log(probabilities.clamp_min(epsilon))

    return -(probabilities * log_probabilities).sum(dim=1)


def classify_ood(
    entropy_scores: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    """Classify samples as OOD when entropy exceeds the threshold.

    Returns:
        Boolean tensor where ``True`` represents an OOD prediction.
    """
    if entropy_scores.ndim != 1:
        raise ValueError("entropy_scores must be a one-dimensional tensor")

    return entropy_scores > threshold