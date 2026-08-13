"""Entropy-based out-of-distribution detection utilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import datasets

try:
    from src.evaluate import get_transform, load_model
except ModuleNotFoundError:
    from evaluate import get_transform, load_model


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

@torch.inference_mode()
def collect_entropy_scores(
    model: torch.nn.Module,
    dataloader: Iterable,
    device: torch.device,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Collect entropy scores for every sample in a dataloader.

    The dataloader may return either:

    - ``(images, labels)``
    - ``images`` only

    Labels are ignored because entropy calculation requires only model logits.

    Args:
        model: Classification model returning logits of shape
            ``(batch_size, num_classes)``.
        dataloader: Iterable yielding image batches or image-label pairs.
        device: Device used for inference.
        temperature: Positive softmax temperature.

    Returns:
        One-dimensional CPU tensor containing one entropy score per image.

    Raises:
        ValueError: If the dataloader contains no samples.
        TypeError: If a batch has an unsupported structure.
    """
    model = model.to(device)
    model.eval()

    collected_scores: list[torch.Tensor] = []

    for batch in dataloader:
        if isinstance(batch, (tuple, list)):
            if not batch:
                raise TypeError("dataloader returned an empty batch")

            images = batch[0]
        elif torch.is_tensor(batch):
            images = batch
        else:
            raise TypeError(
                "dataloader batches must be tensors or tuples/lists "
                "whose first item is an image tensor"
            )

        if not torch.is_tensor(images):
            raise TypeError("the image batch must be a torch.Tensor")

        images = images.to(device)
        logits = model(images)

        if not torch.is_tensor(logits):
            raise TypeError("model output must be a torch.Tensor")

        batch_scores = compute_entropy(
            logits=logits,
            temperature=temperature,
        )

        collected_scores.append(batch_scores.detach().cpu())

    if not collected_scores:
        raise ValueError("dataloader did not provide any samples")

    return torch.cat(collected_scores, dim=0)


def validate_ood_scores(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert ID and OOD score tensors to NumPy arrays.

    Higher scores must indicate a greater likelihood of being OOD.

    Args:
        id_scores: One-dimensional entropy scores for in-distribution samples.
        ood_scores: One-dimensional entropy scores for OOD samples.

    Returns:
        Tuple containing validated ID and OOD NumPy arrays.

    Raises:
        ValueError: If either score tensor is invalid.
    """
    for name, scores in (
        ("id_scores", id_scores),
        ("ood_scores", ood_scores),
    ):
        if not torch.is_tensor(scores):
            raise TypeError(f"{name} must be a torch.Tensor")

        if scores.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")

        if scores.numel() == 0:
            raise ValueError(f"{name} must not be empty")

        if not torch.isfinite(scores).all():
            raise ValueError(f"{name} must contain only finite values")

    return (
        id_scores.detach().cpu().numpy().astype(np.float64),
        ood_scores.detach().cpu().numpy().astype(np.float64),
    )


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
    

def compute_fpr_at_tpr(
    labels: np.ndarray,
    scores: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """Compute false-positive rate at the requested true-positive rate.

    OOD samples are treated as the positive class.

    Args:
        labels: Binary labels where 1 means OOD and 0 means ID.
        scores: OOD scores where higher values indicate greater OOD likelihood.
        target_tpr: Desired OOD true-positive rate.

    Returns:
        False-positive rate at the first threshold reaching ``target_tpr``.

    Raises:
        ValueError: If ``target_tpr`` is outside the open interval (0, 1).
    """
    if not 0.0 < target_tpr < 1.0:
        raise ValueError("target_tpr must be between 0 and 1")

    false_positive_rates, true_positive_rates, _ = roc_curve(
        labels,
        scores,
        pos_label=1,
    )

    eligible_indices = np.flatnonzero(true_positive_rates >= target_tpr)

    if eligible_indices.size == 0:
        return 1.0

    return float(false_positive_rates[eligible_indices[0]])


def compute_ood_metrics(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    target_tpr: float = 0.95,
) -> dict[str, float | int]:
    """Compute standard OOD detection metrics.

    Entropy is used as the OOD score, so larger values indicate greater
    uncertainty and a higher likelihood that a sample is OOD.

    Metrics:
        - AUROC: OOD samples are the positive class.
        - AUPR-OUT: OOD samples are the positive class.
        - AUPR-IN: ID samples are the positive class, using negated scores.
        - FPR@TPR: Fraction of ID samples incorrectly classified as OOD at
          the requested OOD true-positive rate.

    Args:
        id_scores: Entropy scores for in-distribution samples.
        ood_scores: Entropy scores for OOD samples.
        target_tpr: Target OOD true-positive rate for FPR calculation.

    Returns:
        Dictionary containing metrics and sample counts.
    """
    id_array, ood_array = validate_ood_scores(id_scores, ood_scores)

    labels_out = np.concatenate(
        [
            np.zeros(id_array.shape[0], dtype=np.int64),
            np.ones(ood_array.shape[0], dtype=np.int64),
        ]
    )
    scores_out = np.concatenate([id_array, ood_array])

    auroc = roc_auc_score(labels_out, scores_out)
    aupr_out = average_precision_score(labels_out, scores_out)

    labels_in = 1 - labels_out
    scores_in = -scores_out
    aupr_in = average_precision_score(labels_in, scores_in)

    fpr_at_target_tpr = compute_fpr_at_tpr(
        labels=labels_out,
        scores=scores_out,
        target_tpr=target_tpr,
    )

    return {
        "auroc": float(auroc),
        "fpr95": float(fpr_at_target_tpr),
        "aupr_in": float(aupr_in),
        "aupr_out": float(aupr_out),
        "target_tpr": float(target_tpr),
        "num_id_samples": int(id_array.shape[0]),
        "num_ood_samples": int(ood_array.shape[0]),
    }
    
    
def save_metrics(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save OOD metrics to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


def save_entropy_scores(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    number_of_classes: int,
    output_path: str | Path,
) -> None:
    """Save held-out labels and normalized entropy-based OOD scores."""
    if number_of_classes <= 1:
        raise ValueError("number_of_classes must be greater than one")

    if id_scores.ndim != 1 or ood_scores.ndim != 1:
        raise ValueError("entropy scores must be one-dimensional")

    if id_scores.numel() == 0 or ood_scores.numel() == 0:
        raise ValueError("entropy score tensors must not be empty")

    maximum_entropy = float(np.log(number_of_classes))

    combined_scores = torch.cat(
        [
            id_scores / maximum_entropy,
            ood_scores / maximum_entropy,
        ]
    ).clamp(0.0, 1.0)

    labels = torch.cat(
        [
            torch.zeros(id_scores.numel(), dtype=torch.int64),
            torch.ones(ood_scores.numel(), dtype=torch.int64),
        ]
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path,
        labels=labels.numpy(),
        scores=combined_scores.detach().cpu().numpy(),
    )
    

def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load and validate the project YAML configuration."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise TypeError("Configuration file must contain a YAML mapping")

    if "ood_entropy" not in config:
        raise KeyError("Configuration is missing the 'ood_entropy' section")

    required_keys = {"temperature", "target_tpr"}
    missing_keys = required_keys - set(config["ood_entropy"])

    if missing_keys:
        missing = ", ".join(sorted(missing_keys))
        raise KeyError(f"ood_entropy configuration is missing: {missing}")

    return config


def plot_entropy_distribution(
    id_scores: torch.Tensor,
    ood_scores: torch.Tensor,
    threshold: float,
    output_path: str | Path,
    bins: int = 40,
) -> None:
    """Plot ID and OOD entropy distributions with the calibrated threshold."""
    id_array, ood_array = validate_ood_scores(id_scores, ood_scores)

    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    if bins <= 0:
        raise ValueError("bins must be greater than zero")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plt.hist(
        id_array,
        bins=bins,
        alpha=0.65,
        label="In-distribution",
        density=True,
        edgecolor="black",
        linewidth=0.4,
    )

    plt.hist(
        ood_array,
        bins=bins,
        alpha=0.65,
        label="Out-of-distribution",
        density=True,
        edgecolor="black",
        linewidth=0.4,
    )

    plt.axvline(
        threshold,
        linestyle="--",
        linewidth=2,
        label=f"Threshold = {threshold:.4f}",
    )

    plt.title("Entropy Distribution: ID vs OOD")
    plt.xlabel("Predictive Entropy")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for entropy-based OOD detection."""
    parser = argparse.ArgumentParser(
        description="Entropy-based out-of-distribution detection"
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.yaml"),
        help="Path to the project YAML configuration.",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/resnet18_best.pt"),
        help="Path to the trained ResNet-18 checkpoint.",
    )

    parser.add_argument(
        "--threshold-output",
        type=Path,
        default=Path("reports/ood/entropy_threshold.json"),
        help="Path for calibrated threshold metadata.",
    )

    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("reports/ood/strategy_a_metrics.json"),
        help="Path for Strategy A metrics.",
    )

    parser.add_argument(
        "--plot-output",
        type=Path,
        default=Path("reports/ood/entropy_distribution.png"),
        help="Path for the entropy-distribution plot.",
    )
    
    parser.add_argument(
        "--id-validation-dir",
        type=Path,
        default=Path("data/processed/val"),
        help="Directory containing ID validation images.",
    )

    parser.add_argument(
        "--id-test-dir",
        type=Path,
        default=Path("data/processed/test"),
        help="Directory containing held-out ID test images.",
    )

    parser.add_argument(
        "--ood-test-dir",
        type=Path,
        default=Path("data/raw/ood/test"),
        help="Directory containing held-out OOD test images.",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override data.num_workers; use 0 on Windows.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration, model, and datasets without inference.",
    )
    
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("reports/ood/strategy_a_test_scores.npz"),
        help="Path for held-out Strategy A labels and normalized entropy scores.",
    )

    return parser.parse_args()


def build_entropy_dataloaders(
    id_validation_dir: str | Path,
    id_test_dir: str | Path,
    ood_test_dir: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build validation, ID-test, and OOD-test dataloaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if num_workers < 0:
        raise ValueError("num_workers must not be negative")

    transform = get_transform(image_size)

    id_validation_dataset = datasets.ImageFolder(
        str(id_validation_dir),
        transform=transform,
    )
    id_test_dataset = datasets.ImageFolder(
        str(id_test_dir),
        transform=transform,
    )
    ood_test_dataset = datasets.ImageFolder(
        str(ood_test_dir),
        transform=transform,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }

    return (
        DataLoader(id_validation_dataset, **loader_kwargs),
        DataLoader(id_test_dataset, **loader_kwargs),
        DataLoader(ood_test_dataset, **loader_kwargs),
    )


def run_entropy_pipeline(
    args: argparse.Namespace,
) -> dict[str, float | int] | None:
    """Run entropy-based OOD detection with the trained ResNet-18."""
    config = load_config(args.config)

    data_config = config["data"]
    entropy_config = config["ood_entropy"]
    head_config = config["train_head"]

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(data_config["num_workers"])
    )

    validation_loader, id_test_loader, ood_test_loader = (
        build_entropy_dataloaders(
            id_validation_dir=args.id_validation_dir,
            id_test_dir=args.id_test_dir,
            ood_test_dir=args.ood_test_dir,
            image_size=int(data_config["img_size"]),
            batch_size=int(head_config["batch_size"]),
            num_workers=num_workers,
        )
    )

    model = load_model(
        num_classes=int(data_config["num_classes"]),
        device=device,
    )

    print("Entropy-based OOD Detection — Strategy A")
    print("-" * 50)
    print(f"Device:                 {device}")
    print(f"ID validation samples:  {len(validation_loader.dataset):,}")
    print(f"ID test samples:        {len(id_test_loader.dataset):,}")
    print(f"OOD test samples:       {len(ood_test_loader.dataset):,}")

    if args.dry_run:
        print("Dry run completed successfully. No inference was performed.")
        return None

    temperature = float(entropy_config["temperature"])
    target_tpr = float(entropy_config["target_tpr"])

    validation_scores = collect_entropy_scores(
        model=model,
        dataloader=validation_loader,
        device=device,
        temperature=temperature,
    )

    threshold = calibrate_threshold(
        id_entropy_scores=validation_scores,
        target_tpr=target_tpr,
    )

    id_test_scores = collect_entropy_scores(
        model=model,
        dataloader=id_test_loader,
        device=device,
        temperature=temperature,
    )

    ood_test_scores = collect_entropy_scores(
        model=model,
        dataloader=ood_test_loader,
        device=device,
        temperature=temperature,
    )
    
    save_entropy_scores(
        id_scores=id_test_scores,
        ood_scores=ood_test_scores,
        number_of_classes=int(data_config["num_classes"]),
        output_path=args.scores_output,
    )

    metrics = compute_ood_metrics(
        id_scores=id_test_scores,
        ood_scores=ood_test_scores,
        target_tpr=target_tpr,
    )

    metrics["threshold"] = threshold
    metrics["temperature"] = temperature
    metrics["num_validation_samples"] = len(
        validation_loader.dataset
    )

    save_threshold(
        threshold=threshold,
        output_path=args.threshold_output,
        target_tpr=target_tpr,
        num_validation_samples=len(validation_loader.dataset),
    )

    save_metrics(
        metrics=metrics,
        output_path=args.metrics_output,
    )

    plot_entropy_distribution(
        id_scores=id_test_scores,
        ood_scores=ood_test_scores,
        threshold=threshold,
        output_path=args.plot_output,
    )

    print()
    print("Strategy A complete")
    print(f"Threshold:    {threshold:.4f}")
    print(f"AUROC:        {metrics['auroc']:.4f}")
    print(f"AUPR-IN:      {metrics['aupr_in']:.4f}")
    print(f"AUPR-OUT:     {metrics['aupr_out']:.4f}")
    print(f"FPR@95TPR:    {metrics['fpr95']:.4f}")

    return metrics


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    run_entropy_pipeline(args)


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()