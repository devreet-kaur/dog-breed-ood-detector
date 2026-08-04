"""Tests for the binary OOD CNN."""

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.ood_binary import (
    BinaryCNN,
    DogOODDataset,
    build_binary_dataloaders,
    build_binary_transforms,
    discover_image_files,
    split_files,
)


def create_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(120, 80, 40)).save(path)

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
    
    
def test_discover_image_files_finds_nested_images(tmp_path) -> None:
    create_test_image(tmp_path / "breed_a" / "dog1.jpg")
    create_test_image(tmp_path / "breed_b" / "dog2.png")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    files = discover_image_files(tmp_path)

    assert len(files) == 2
    assert all(path.suffix.lower() in {".jpg", ".png"} for path in files)


def test_discover_image_files_rejects_missing_directory(
    tmp_path,
) -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        discover_image_files(tmp_path / "missing")


def test_split_files_is_deterministic() -> None:
    files = [Path(f"image_{index}.jpg") for index in range(10)]

    first_train, first_val = split_files(
        files,
        validation_fraction=0.2,
        seed=42,
    )
    second_train, second_val = split_files(
        files,
        validation_fraction=0.2,
        seed=42,
    )

    assert first_train == second_train
    assert first_val == second_val
    assert len(first_train) == 8
    assert len(first_val) == 2


def test_split_files_has_no_overlap() -> None:
    files = [Path(f"image_{index}.jpg") for index in range(10)]

    training, validation = split_files(
        files,
        validation_fraction=0.3,
        seed=42,
    )

    assert set(training).isdisjoint(validation)
    assert set(training + validation) == set(files)


def test_dog_ood_dataset_returns_tensor_and_label(tmp_path) -> None:
    dog_path = tmp_path / "dogs" / "dog.jpg"
    ood_path = tmp_path / "ood" / "cat.jpg"

    create_test_image(dog_path)
    create_test_image(ood_path)

    _, evaluation_transform = build_binary_transforms(image_size=64)

    dataset = DogOODDataset(
        dog_files=[dog_path],
        ood_files=[ood_path],
        transform=evaluation_transform,
    )

    dog_image, dog_label = dataset[0]
    ood_image, ood_label = dataset[1]

    assert dog_image.shape == (3, 64, 64)
    assert ood_image.shape == (3, 64, 64)
    assert dog_label == 0
    assert ood_label == 1


def test_binary_dataloaders_return_binary_batches(tmp_path) -> None:
    dog_train = tmp_path / "dog_train"
    dog_val = tmp_path / "dog_val"
    ood_development = tmp_path / "ood_val"

    for index in range(4):
        create_test_image(
            dog_train / "breed_a" / f"dog_train_{index}.jpg"
        )

    for index in range(2):
        create_test_image(
            dog_val / "breed_a" / f"dog_val_{index}.jpg"
        )

    for index in range(5):
        create_test_image(
            ood_development / "cats" / f"ood_{index}.jpg"
        )

    train_loader, validation_loader = build_binary_dataloaders(
        dog_train_dir=dog_train,
        dog_val_dir=dog_val,
        ood_development_dir=ood_development,
        image_size=32,
        batch_size=2,
        num_workers=0,
        seed=42,
        ood_validation_fraction=0.2,
    )

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(validation_loader))

    assert train_images.shape[1:] == (3, 32, 32)
    assert val_images.shape[1:] == (3, 32, 32)
    assert set(train_labels.tolist()).issubset({0, 1})
    assert set(val_labels.tolist()).issubset({0, 1})