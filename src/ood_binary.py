"""Binary CNN for dog-versus-OOD classification."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)
from torchvision import transforms

logger = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class EpochMetrics:
    """Loss and accuracy recorded for one epoch."""

    loss: float
    accuracy: float


class DogOODDataset(Dataset):
    """Binary image dataset for dog-versus-OOD classification.

    Labels:
        0: dog / in-distribution
        1: non-dog / out-of-distribution
    """

    def __init__(
        self,
        dog_files: list[Path],
        ood_files: list[Path],
        transform: Callable | None = None,
    ) -> None:
        if not dog_files:
            raise ValueError("dog_files must not be empty")

        if not ood_files:
            raise ValueError("ood_files must not be empty")

        self.samples: list[tuple[Path, int]] = [
            *((path, 0) for path in dog_files),
            *((path, 1) for path in ood_files),
        ]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

        if not torch.is_tensor(image):
            raise TypeError(
                "transform must convert the image to a torch.Tensor"
            )

        return image, label


class BinaryCNN(nn.Module):
    """Small CNN for binary dog-versus-OOD classification.

    Input:
        RGB images with shape ``(batch_size, 3, 224, 224)``.

    Output:
        Two logits per image:
        - class 0: dog / in-distribution
        - class 1: non-dog / out-of-distribution
    """

    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be between 0 and 1")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return binary classification logits."""
        features = self.features(inputs)
        return self.classifier(features)
    
    
def discover_image_files(root: str | Path) -> list[Path]:
    """Return all supported image files beneath a directory."""
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Expected a directory: {root}")

    image_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not image_files:
        raise ValueError(f"No supported images found in: {root}")

    return image_files


def split_files(
    files: list[Path],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """Split files deterministically into training and validation groups."""
    if not files:
        raise ValueError("files must not be empty")

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must be between 0 and 1"
        )

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        len(files),
        generator=generator,
    ).tolist()

    validation_size = max(
        1,
        round(len(files) * validation_fraction),
    )

    if validation_size >= len(files):
        raise ValueError(
            "validation split must leave at least one training sample"
        )

    validation_indices = set(permutation[:validation_size])

    training_files = [
        file
        for index, file in enumerate(files)
        if index not in validation_indices
    ]
    validation_files = [
        file
        for index, file in enumerate(files)
        if index in validation_indices
    ]

    return training_files, validation_files


def build_binary_transforms(
    image_size: int,
) -> tuple[transforms.Compose, transforms.Compose]:
    """Create training and evaluation transforms."""
    if image_size <= 0:
        raise ValueError("image_size must be greater than zero")

    normalization_mean = [0.485, 0.456, 0.406]
    normalization_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalization_mean,
                std=normalization_std,
            ),
        ]
    )

    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=normalization_mean,
                std=normalization_std,
            ),
        ]
    )

    return train_transform, evaluation_transform


def build_binary_dataloaders(
    dog_train_dir: str | Path,
    dog_val_dir: str | Path,
    ood_development_dir: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    ood_validation_fraction: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """Build binary training and validation dataloaders."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")

    if num_workers < 0:
        raise ValueError("num_workers must not be negative")

    dog_train_files = discover_image_files(dog_train_dir)
    dog_val_files = discover_image_files(dog_val_dir)
    ood_files = discover_image_files(ood_development_dir)

    ood_train_files, ood_val_files = split_files(
        files=ood_files,
        validation_fraction=ood_validation_fraction,
        seed=seed,
    )

    train_transform, evaluation_transform = build_binary_transforms(
        image_size=image_size
    )

    training_dataset = DogOODDataset(
        dog_files=dog_train_files,
        ood_files=ood_train_files,
        transform=train_transform,
    )

    validation_dataset = DogOODDataset(
        dog_files=dog_val_files,
        ood_files=ood_val_files,
        transform=evaluation_transform,
    )

    training_labels = [
        label
        for _, label in training_dataset.samples
    ]

    training_sampler = build_balanced_sampler(
        labels=training_labels,
        seed=seed,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=batch_size,
        sampler=training_sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return training_loader, validation_loader


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Train the binary classifier for one epoch."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("training dataloader did not provide any samples")

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )


def build_balanced_sampler(
    labels: list[int],
    seed: int,
) -> WeightedRandomSampler:
    """Create a sampler that balances the two binary classes."""
    if not labels:
        raise ValueError("labels must not be empty")

    unique_labels = set(labels)

    if unique_labels != {0, 1}:
        raise ValueError("labels must contain both binary classes 0 and 1")

    class_counts = {
        label: labels.count(label)
        for label in unique_labels
    }

    sample_weights = [
        1.0 / class_counts[label]
        for label in labels
    ]

    generator = torch.Generator().manual_seed(seed)

    minority_class_size = min(class_counts.values())
    balanced_epoch_size = 2 * minority_class_size

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=balanced_epoch_size,
        replacement=True,
        generator=generator,
    )
    
    
@torch.inference_mode()
def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    """Evaluate the binary classifier for one epoch."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise ValueError("validation dataloader did not provide any samples")

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
    )
    

@torch.inference_mode()
def collect_binary_predictions(
    model: nn.Module,
    dataloader: Iterable,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collect labels, OOD probabilities, and predicted classes.

    Class 1 represents OOD, so the returned probability tensor contains
    softmax probabilities for class 1.

    Args:
        model: Binary classification model returning two logits per sample.
        dataloader: Iterable yielding ``(images, labels)`` batches.
        device: Device used for inference.

    Returns:
        Tuple containing:
        - ground-truth labels
        - OOD probabilities
        - predicted class labels

        All returned tensors are one-dimensional CPU tensors.
    """
    model = model.to(device)
    model.eval()

    collected_labels: list[torch.Tensor] = []
    collected_probabilities: list[torch.Tensor] = []
    collected_predictions: list[torch.Tensor] = []

    for batch in dataloader:
        if not isinstance(batch, (tuple, list)) or len(batch) < 2:
            raise TypeError(
                "dataloader must return image-label pairs"
            )

        images, labels = batch[0], batch[1]

        if not torch.is_tensor(images) or not torch.is_tensor(labels):
            raise TypeError("images and labels must be torch tensors")

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        if not torch.is_tensor(logits):
            raise TypeError("model output must be a torch tensor")

        if logits.ndim != 2 or logits.shape[1] != 2:
            raise ValueError(
                "binary model logits must have shape "
                "(batch_size, 2)"
            )

        probabilities = torch.softmax(logits, dim=1)
        ood_probabilities = probabilities[:, 1]
        predictions = logits.argmax(dim=1)

        collected_labels.append(labels.detach().cpu())
        collected_probabilities.append(
            ood_probabilities.detach().cpu()
        )
        collected_predictions.append(predictions.detach().cpu())

    if not collected_labels:
        raise ValueError("dataloader did not provide any samples")

    return (
        torch.cat(collected_labels),
        torch.cat(collected_probabilities),
        torch.cat(collected_predictions),
    )

 
def compute_binary_fpr_at_tpr(
    labels: np.ndarray,
    ood_probabilities: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """Compute ID false-positive rate at the requested OOD TPR."""
    if not 0.0 < target_tpr < 1.0:
        raise ValueError("target_tpr must be between 0 and 1")

    false_positive_rates, true_positive_rates, _ = roc_curve(
        labels,
        ood_probabilities,
        pos_label=1,
    )

    eligible = np.flatnonzero(
        true_positive_rates >= target_tpr
    )

    if eligible.size == 0:
        return 1.0

    return float(false_positive_rates[eligible[0]])
 

def compute_binary_metrics(
    labels: torch.Tensor,
    ood_probabilities: torch.Tensor,
    predictions: torch.Tensor,
    target_tpr: float = 0.95,
) -> dict[str, float | int]:
    """Compute binary dog-versus-OOD evaluation metrics."""
    tensors = {
        "labels": labels,
        "ood_probabilities": ood_probabilities,
        "predictions": predictions,
    }

    for name, values in tensors.items():
        if not torch.is_tensor(values):
            raise TypeError(f"{name} must be a torch.Tensor")

        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")

        if values.numel() == 0:
            raise ValueError(f"{name} must not be empty")

        if not torch.isfinite(values.float()).all():
            raise ValueError(
                f"{name} must contain only finite values"
            )

    if not (
        labels.numel()
        == ood_probabilities.numel()
        == predictions.numel()
    ):
        raise ValueError(
            "labels, probabilities, and predictions "
            "must have equal length"
        )

    labels_array = labels.detach().cpu().numpy().astype(
        np.int64
    )
    probabilities_array = (
        ood_probabilities.detach()
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    predictions_array = (
        predictions.detach()
        .cpu()
        .numpy()
        .astype(np.int64)
    )

    if set(np.unique(labels_array)) != {0, 1}:
        raise ValueError(
            "labels must contain both binary classes 0 and 1"
        )

    if np.any(
        (probabilities_array < 0.0)
        | (probabilities_array > 1.0)
    ):
        raise ValueError(
            "ood_probabilities must be between 0 and 1"
        )

    accuracy = accuracy_score(
        labels_array,
        predictions_array,
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            labels_array,
            predictions_array,
            average="binary",
            pos_label=1,
            zero_division=0,
        )
    )

    auroc = roc_auc_score(
        labels_array,
        probabilities_array,
    )

    aupr_out = average_precision_score(
        labels_array,
        probabilities_array,
    )

    aupr_in = average_precision_score(
        1 - labels_array,
        1.0 - probabilities_array,
    )

    fpr95 = compute_binary_fpr_at_tpr(
        labels=labels_array,
        ood_probabilities=probabilities_array,
        target_tpr=target_tpr,
    )

    return {
        "accuracy": float(accuracy),
        "precision_ood": float(precision),
        "recall_ood": float(recall),
        "f1_ood": float(f1),
        "auroc": float(auroc),
        "aupr_in": float(aupr_in),
        "aupr_out": float(aupr_out),
        "fpr95": float(fpr95),
        "target_tpr": float(target_tpr),
        "num_samples": int(labels_array.shape[0]),
        "num_dog_samples": int(
            np.sum(labels_array == 0)
        ),
        "num_ood_samples": int(
            np.sum(labels_array == 1)
        ),
    }


def save_binary_metrics(
    metrics: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save binary OOD metrics as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    
def save_binary_checkpoint(
    model: nn.Module,
    output_path: str | Path,
) -> None:
    """Save the binary model state dictionary."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path)
    

@dataclass
class TrainingHistory:
    """Training and validation metrics collected across epochs."""

    train_loss: list[float]
    train_accuracy: list[float]
    val_loss: list[float]
    val_accuracy: list[float]
    best_epoch: int
    best_val_accuracy: float

    
def train_binary_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
) -> TrainingHistory:
    """Train the binary classifier and save the best checkpoint."""
    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")

    model = model.to(device)

    train_losses: list[float] = []
    train_accuracies: list[float] = []
    val_losses: list[float] = []
    val_accuracies: list[float] = []

    best_epoch = 0
    best_val_accuracy = float("-inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        validation_metrics = validate_one_epoch(
            model=model,
            dataloader=validation_loader,
            criterion=criterion,
            device=device,
        )

        train_losses.append(train_metrics.loss)
        train_accuracies.append(train_metrics.accuracy)
        val_losses.append(validation_metrics.loss)
        val_accuracies.append(validation_metrics.accuracy)

        logger.info(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_metrics.loss:.4f} | "
            f"train_acc={train_metrics.accuracy:.4f} | "
            f"val_loss={validation_metrics.loss:.4f} | "
            f"val_acc={validation_metrics.accuracy:.4f}"
        )

        if validation_metrics.accuracy > best_val_accuracy:
            best_val_accuracy = validation_metrics.accuracy
            best_epoch = epoch
            save_binary_checkpoint(
                model=model,
                output_path=checkpoint_path,
            )

    return TrainingHistory(
        train_loss=train_losses,
        train_accuracy=train_accuracies,
        val_loss=val_losses,
        val_accuracy=val_accuracies,
        best_epoch=best_epoch,
        best_val_accuracy=best_val_accuracy,
    )
    
    
def load_binary_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """Load and validate configuration for the binary OOD strategy."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise TypeError(
            "Configuration file must contain a YAML mapping"
        )

    required_sections = {
        "data",
        "ood_binary",
        "ood_entropy",
    }

    missing_sections = required_sections - set(config)

    if missing_sections:
        missing = ", ".join(sorted(missing_sections))
        raise KeyError(
            f"Configuration is missing required sections: {missing}"
        )

    required_data_keys = {
        "img_size",
        "num_workers",
        "seed",
    }
    required_binary_keys = {
        "epochs",
        "lr",
        "batch_size",
        "dropout",
    }
    required_entropy_keys = {
        "target_tpr",
    }

    missing_data = required_data_keys - set(config["data"])
    missing_binary = required_binary_keys - set(
        config["ood_binary"]
    )
    missing_entropy = required_entropy_keys - set(
        config["ood_entropy"]
    )

    if missing_data:
        missing = ", ".join(sorted(missing_data))
        raise KeyError(f"data configuration is missing: {missing}")

    if missing_binary:
        missing = ", ".join(sorted(missing_binary))
        raise KeyError(
            f"ood_binary configuration is missing: {missing}"
        )

    if missing_entropy:
        missing = ", ".join(sorted(missing_entropy))
        raise KeyError(
            f"ood_entropy configuration is missing: {missing}"
        )

    return config


def select_device() -> torch.device:
    """Select CUDA, Apple MPS, or CPU in that order."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def plot_training_history(
    history: TrainingHistory,
    output_path: str | Path,
) -> None:
    """Save binary-CNN loss and accuracy training curves."""
    number_of_epochs = len(history.train_loss)

    if number_of_epochs == 0:
        raise ValueError("training history must not be empty")

    history_lengths = {
        len(history.train_loss),
        len(history.train_accuracy),
        len(history.val_loss),
        len(history.val_accuracy),
    }

    if len(history_lengths) != 1:
        raise ValueError(
            "all training-history lists must have equal length"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = list(range(1, number_of_epochs + 1))

    figure, loss_axis = plt.subplots(figsize=(10, 6))

    loss_axis.plot(
        epochs,
        history.train_loss,
        label="Training Loss",
    )
    loss_axis.plot(
        epochs,
        history.val_loss,
        label="Validation Loss",
    )
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(
        linestyle="--",
        alpha=0.3,
    )

    accuracy_axis = loss_axis.twinx()

    accuracy_axis.plot(
        epochs,
        history.train_accuracy,
        linestyle="--",
        label="Training Accuracy",
    )
    accuracy_axis.plot(
        epochs,
        history.val_accuracy,
        linestyle="--",
        label="Validation Accuracy",
    )
    accuracy_axis.set_ylabel("Accuracy")

    loss_lines, loss_labels = (
        loss_axis.get_legend_handles_labels()
    )
    accuracy_lines, accuracy_labels = (
        accuracy_axis.get_legend_handles_labels()
    )

    loss_axis.legend(
        loss_lines + accuracy_lines,
        loss_labels + accuracy_labels,
        loc="center right",
    )

    plt.title("Binary OOD CNN Training History")
    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)
    
    
def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Strategy B."""
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the binary dog-versus-OOD CNN"
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("params.yaml"),
        help="Path to the project YAML configuration.",
    )

    parser.add_argument(
        "--dog-train-dir",
        type=Path,
        default=Path("data/processed/train"),
        help="Directory containing dog training images.",
    )

    parser.add_argument(
        "--dog-val-dir",
        type=Path,
        default=Path("data/processed/val"),
        help="Directory containing dog validation images.",
    )

    parser.add_argument(
        "--ood-development-dir",
        type=Path,
        default=Path("data/raw/ood/val"),
        help=(
            "OOD development directory used for binary "
            "training and validation."
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/binary_cnn.pt"),
        help="Path for the best binary CNN checkpoint.",
    )

    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path(
            "reports/ood/strategy_b_validation_metrics.json"
        ),
        help="Path for binary validation metrics.",
    )

    parser.add_argument(
        "--history-plot",
        type=Path,
        default=Path(
            "reports/ood/binary_training_history.png"
        ),
        help="Path for binary CNN training curves.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate configuration and dataloaders "
            "without training."
        ),
    )

    return parser.parse_args()


def run_binary_pipeline(
    args: argparse.Namespace,
) -> dict[str, float | int] | None:
    """Run binary-CNN setup, training, and validation."""
    config = load_binary_config(args.config)

    data_config = config["data"]
    binary_config = config["ood_binary"]
    target_tpr = float(
        config["ood_entropy"]["target_tpr"]
    )

    device = select_device()

    logger.info("Binary OOD Detection — Strategy B")
    logger.info("-" * 50)
    logger.info(f"Device:          {device}")
    logger.info(f"Image size:      {data_config['img_size']}")
    logger.info(f"Batch size:      {binary_config['batch_size']}")
    logger.info(f"Epochs:          {binary_config['epochs']}")
    logger.info(f"Learning rate:   {binary_config['lr']}")
    logger.info(f"Dropout:         {binary_config['dropout']}")

    train_loader, validation_loader = (
        build_binary_dataloaders(
            dog_train_dir=args.dog_train_dir,
            dog_val_dir=args.dog_val_dir,
            ood_development_dir=args.ood_development_dir,
            image_size=int(data_config["img_size"]),
            batch_size=int(binary_config["batch_size"]),
            num_workers=int(data_config["num_workers"]),
            seed=int(data_config["seed"]),
        )
    )

    logger.info(
        f"Training samples:   "
        f"{len(train_loader.dataset):,}"
    )
    logger.info(
        f"Validation samples: "
        f"{len(validation_loader.dataset):,}"
    )

    if args.dry_run:
        logger.info(
            "Dry run completed successfully. "
            "No model was trained."
        )
        return None

    model = BinaryCNN(
        dropout=float(binary_config["dropout"])
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(binary_config["lr"]),
    )

    criterion = nn.CrossEntropyLoss()

    history = train_binary_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=int(binary_config["epochs"]),
        checkpoint_path=args.checkpoint,
    )

    plot_training_history(
        history=history,
        output_path=args.history_plot,
    )

    model.load_state_dict(
        torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=True,
        )
    )

    labels, ood_probabilities, predictions = (
        collect_binary_predictions(
            model=model,
            dataloader=validation_loader,
            device=device,
        )
    )

    metrics = compute_binary_metrics(
        labels=labels,
        ood_probabilities=ood_probabilities,
        predictions=predictions,
        target_tpr=target_tpr,
    )

    metrics["best_epoch"] = history.best_epoch
    metrics["best_val_accuracy"] = (
        history.best_val_accuracy
    )

    save_binary_metrics(
        metrics=metrics,
        output_path=args.metrics_output,
    )

    logger.info()
    logger.info("Training complete")
    logger.info(f"Best epoch:       {history.best_epoch}")
    logger.info(
        f"Best validation:  "
        f"{history.best_val_accuracy:.4f}"
    )
    logger.info(f"Validation AUROC: {metrics['auroc']:.4f}")
    logger.info(f"FPR@95TPR:        {metrics['fpr95']:.4f}")
    logger.info(f"Checkpoint:       {args.checkpoint}")

    return metrics


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    run_binary_pipeline(args)


if __name__ == "__main__":
    main()