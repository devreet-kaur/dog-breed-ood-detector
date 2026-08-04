"""Binary CNN for dog-versus-OOD classification."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

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

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
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

        print(
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