"""Entropy-based out-of-distribution detection utilities."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
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


def calibrate_threshold(
    id_entropy_scores: torch.Tensor,
    target_tpr: float = 0.95,
) -> float:
    """Calibrate an entropy threshold using ID validation scores.

    The threshold is selected so that approximately ``target_tpr`` of
    in-distribution validation samples are accepted as ID.

    Since lower entropy indicates greater confidence, the threshold is the
    ``target_tpr`` quantile of the ID entropy distribution.

    Args:
        id_entropy_scores: One-dimensional tensor of ID entropy scores.
        target_tpr: Desired true-positive rate for accepting ID samples.

    Returns:
        Calibrated entropy threshold.

    Raises:
        ValueError: If inputs are invalid.
    """
    if id_entropy_scores.ndim != 1:
        raise ValueError("id_entropy_scores must be a one-dimensional tensor")

    if id_entropy_scores.numel() == 0:
        raise ValueError("id_entropy_scores must not be empty")

    if not torch.isfinite(id_entropy_scores).all():
        raise ValueError("id_entropy_scores must contain only finite values")

    if not 0.0 < target_tpr < 1.0:
        raise ValueError("target_tpr must be between 0 and 1")

    threshold = torch.quantile(
        id_entropy_scores.float(),
        q=target_tpr,
    )

    return float(threshold.item())


def save_threshold(
    threshold: float,
    output_path: str | Path,
    target_tpr: float,
    num_validation_samples: int,
) -> None:
    """Save threshold calibration metadata as JSON."""
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    if num_validation_samples <= 0:
        raise ValueError("num_validation_samples must be greater than zero")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "threshold": float(threshold),
        "target_tpr": float(target_tpr),
        "num_validation_samples": int(num_validation_samples),
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )