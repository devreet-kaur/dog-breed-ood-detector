"""Tests for the binary OOD CNN."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.ood_binary import (
    BinaryCNN,
    DogOODDataset,
    EpochMetrics,
    TrainingHistory,
    build_balanced_sampler,
    build_binary_dataloaders,
    build_binary_test_dataloader,
    build_binary_transforms,
    collect_binary_predictions,
    compute_binary_fpr_at_tpr,
    compute_binary_metrics,
    discover_image_files,
    load_binary_config,
    plot_training_history,
    save_binary_checkpoint,
    save_binary_metrics,
    save_binary_scores,
    select_device,
    split_files,
    train_binary_model,
    train_one_epoch,
    validate_one_epoch,
)


def create_test_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(120, 80, 40)).save(path)


class TinyBinaryClassifier(nn.Module):
    """Small classifier for training-loop tests."""

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(4, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(inputs)


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
    
    
def test_balanced_sampler_assigns_higher_weight_to_minority_class() -> None:
    labels = [0, 0, 0, 0, 1]

    sampler = build_balanced_sampler(labels=labels, seed=42)

    weights = list(sampler.weights)

    assert weights[4] > weights[0]


def test_balanced_sampler_assigns_equal_total_weight_to_classes() -> None:
    labels = [0] * 20 + [1] * 2

    sampler = build_balanced_sampler(labels=labels, seed=42)
    weights = list(sampler.weights)

    dog_weight = sum(
        weight
        for weight, label in zip(weights, labels, strict=True)
        if label == 0
    )
    ood_weight = sum(
        weight
        for weight, label in zip(weights, labels, strict=True)
        if label == 1
    )

    assert dog_weight == pytest.approx(ood_weight)


def test_balanced_sampler_rejects_single_class() -> None:
    with pytest.raises(ValueError, match="both binary classes"):
        build_balanced_sampler(labels=[0, 0, 0], seed=42)


def test_balanced_sampler_rejects_empty_labels() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        build_balanced_sampler(labels=[], seed=42)
        
        
def test_train_one_epoch_returns_metrics() -> None:
    features = torch.randn(8, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    dataloader = DataLoader(
        TensorDataset(features, labels),
        batch_size=4,
        shuffle=False,
    )

    model = TinyBinaryClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    metrics = train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert isinstance(metrics, EpochMetrics)
    assert metrics.loss >= 0.0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_train_one_epoch_updates_parameters() -> None:
    features = torch.randn(8, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])

    dataloader = DataLoader(
        TensorDataset(features, labels),
        batch_size=4,
    )

    model = TinyBinaryClassifier()
    initial_weights = model.classifier.weight.detach().clone()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    train_one_epoch(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert not torch.equal(
        initial_weights,
        model.classifier.weight.detach(),
    )


def test_validate_one_epoch_does_not_update_parameters() -> None:
    features = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])

    dataloader = DataLoader(
        TensorDataset(features, labels),
        batch_size=3,
    )

    model = TinyBinaryClassifier()
    initial_weights = model.classifier.weight.detach().clone()

    metrics = validate_one_epoch(
        model=model,
        dataloader=dataloader,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    assert torch.equal(
        initial_weights,
        model.classifier.weight.detach(),
    )
    assert metrics.loss >= 0.0
    assert 0.0 <= metrics.accuracy <= 1.0


def test_validate_one_epoch_sets_evaluation_mode() -> None:
    features = torch.randn(4, 4)
    labels = torch.tensor([0, 1, 0, 1])

    model = TinyBinaryClassifier()
    model.train()

    validate_one_epoch(
        model=model,
        dataloader=DataLoader(
            TensorDataset(features, labels),
            batch_size=2,
        ),
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    assert model.training is False


def test_save_binary_checkpoint_creates_file(tmp_path) -> None:
    output_path = tmp_path / "models" / "binary_cnn.pt"
    model = TinyBinaryClassifier()

    save_binary_checkpoint(model, output_path)

    assert output_path.exists()

    saved_state = torch.load(
        output_path,
        map_location="cpu",
        weights_only=True,
    )

    assert saved_state.keys() == model.state_dict().keys()
    
    
def test_train_binary_model_returns_complete_history(tmp_path) -> None:
    features = torch.randn(12, 4)
    labels = torch.tensor([0, 1] * 6)

    train_loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=4,
        shuffle=False,
    )
    validation_loader = DataLoader(
        TensorDataset(features, labels),
        batch_size=4,
        shuffle=False,
    )

    model = TinyBinaryClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    history = train_binary_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=3,
        checkpoint_path=tmp_path / "binary_cnn.pt",
    )

    assert isinstance(history, TrainingHistory)
    assert len(history.train_loss) == 3
    assert len(history.train_accuracy) == 3
    assert len(history.val_loss) == 3
    assert len(history.val_accuracy) == 3
    assert 1 <= history.best_epoch <= 3
    assert 0.0 <= history.best_val_accuracy <= 1.0


def test_train_binary_model_saves_best_checkpoint(tmp_path) -> None:
    features = torch.randn(8, 4)
    labels = torch.tensor([0, 1] * 4)

    dataloader = DataLoader(
        TensorDataset(features, labels),
        batch_size=4,
    )

    checkpoint_path = tmp_path / "models" / "binary_cnn.pt"

    model = TinyBinaryClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    train_binary_model(
        model=model,
        train_loader=dataloader,
        validation_loader=dataloader,
        optimizer=optimizer,
        criterion=nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
        epochs=2,
        checkpoint_path=checkpoint_path,
    )

    assert checkpoint_path.exists()


@pytest.mark.parametrize("epochs", [0, -1])
def test_train_binary_model_rejects_invalid_epochs(
    epochs: int,
    tmp_path,
) -> None:
    features = torch.randn(4, 4)
    labels = torch.tensor([0, 1, 0, 1])

    dataloader = DataLoader(
        TensorDataset(features, labels),
        batch_size=2,
    )

    model = TinyBinaryClassifier()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="greater than zero"):
        train_binary_model(
            model=model,
            train_loader=dataloader,
            validation_loader=dataloader,
            optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"),
            epochs=epochs,
            checkpoint_path=tmp_path / "binary_cnn.pt",
        )
        
        
def test_collect_binary_predictions_returns_expected_shapes() -> None:
    features = torch.randn(6, 4)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])

    dataloader = DataLoader(
        TensorDataset(features, labels),
        batch_size=2,
        shuffle=False,
    )

    model = TinyBinaryClassifier()

    collected_labels, probabilities, predictions = (
        collect_binary_predictions(
            model=model,
            dataloader=dataloader,
            device=torch.device("cpu"),
        )
    )

    assert collected_labels.shape == (6,)
    assert probabilities.shape == (6,)
    assert predictions.shape == (6,)
    assert torch.all(
        (probabilities >= 0.0)
        & (probabilities <= 1.0)
    )


def test_collect_binary_predictions_matches_argmax() -> None:
    features = torch.randn(4, 4)
    labels = torch.tensor([0, 1, 0, 1])

    model = TinyBinaryClassifier()
    dataloader = DataLoader(
        TensorDataset(features, labels),
        batch_size=4,
    )

    _, probabilities, predictions = (
        collect_binary_predictions(
            model=model,
            dataloader=dataloader,
            device=torch.device("cpu"),
        )
    )

    expected_logits = model(features)
    expected_probabilities = torch.softmax(
        expected_logits,
        dim=1,
    )[:, 1]
    expected_predictions = expected_logits.argmax(dim=1)

    assert torch.allclose(
        probabilities,
        expected_probabilities,
    )
    assert torch.equal(
        predictions,
        expected_predictions,
    )


def test_perfect_binary_predictions_have_perfect_metrics() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    probabilities = torch.tensor([0.1, 0.2, 0.8, 0.9])
    predictions = torch.tensor([0, 0, 1, 1])

    metrics = compute_binary_metrics(
        labels=labels,
        ood_probabilities=probabilities,
        predictions=predictions,
    )

    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["precision_ood"] == pytest.approx(1.0)
    assert metrics["recall_ood"] == pytest.approx(1.0)
    assert metrics["f1_ood"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["aupr_in"] == pytest.approx(1.0)
    assert metrics["aupr_out"] == pytest.approx(1.0)
    assert metrics["fpr95"] == pytest.approx(0.0)


def test_binary_metrics_count_classes() -> None:
    metrics = compute_binary_metrics(
        labels=torch.tensor([0, 0, 0, 1, 1]),
        ood_probabilities=torch.tensor(
            [0.1, 0.2, 0.3, 0.8, 0.9]
        ),
        predictions=torch.tensor([0, 0, 0, 1, 1]),
    )

    assert metrics["num_samples"] == 5
    assert metrics["num_dog_samples"] == 3
    assert metrics["num_ood_samples"] == 2


def test_compute_binary_fpr_for_perfect_separation() -> None:
    labels = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array(
        [0.1, 0.2, 0.3, 0.8, 0.9, 1.0]
    )

    result = compute_binary_fpr_at_tpr(
        labels=labels,
        ood_probabilities=probabilities,
        target_tpr=0.95,
    )

    assert result == pytest.approx(0.0)


def test_binary_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        compute_binary_metrics(
            labels=torch.tensor([0, 1]),
            ood_probabilities=torch.tensor([0.2]),
            predictions=torch.tensor([0, 1]),
        )


def test_binary_metrics_reject_invalid_probabilities() -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        compute_binary_metrics(
            labels=torch.tensor([0, 1]),
            ood_probabilities=torch.tensor([0.2, 1.2]),
            predictions=torch.tensor([0, 1]),
        )


def test_save_binary_metrics_writes_json(tmp_path) -> None:
    output_path = tmp_path / "reports" / "metrics.json"
    metrics = {
        "accuracy": 0.9,
        "auroc": 0.95,
        "fpr95": 0.15,
    }

    save_binary_metrics(metrics, output_path)

    saved = json.loads(
        output_path.read_text(encoding="utf-8")
    )

    assert saved == metrics
    
    
def test_load_binary_config_reads_required_values(
    tmp_path,
) -> None:
    config_path = tmp_path / "params.yaml"

    config_path.write_text(
        """
data:
  img_size: 224
  num_workers: 0
  seed: 42

ood_binary:
  epochs: 3
  lr: 0.001
  batch_size: 8
  dropout: 0.3

ood_entropy:
  target_tpr: 0.95
""".strip(),
        encoding="utf-8",
    )

    config = load_binary_config(config_path)

    assert config["data"]["img_size"] == 224
    assert config["ood_binary"]["epochs"] == 3
    assert config["ood_entropy"]["target_tpr"] == 0.95


def test_load_binary_config_rejects_missing_file(
    tmp_path,
) -> None:
    with pytest.raises(
        FileNotFoundError,
        match="not found",
    ):
        load_binary_config(tmp_path / "missing.yaml")


def test_load_binary_config_rejects_missing_section(
    tmp_path,
) -> None:
    config_path = tmp_path / "params.yaml"

    config_path.write_text(
        """
data:
  img_size: 224
  num_workers: 0
  seed: 42
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(
        KeyError,
        match="required sections",
    ):
        load_binary_config(config_path)


def test_plot_training_history_creates_file(
    tmp_path,
) -> None:
    history = TrainingHistory(
        train_loss=[0.9, 0.6, 0.4],
        train_accuracy=[0.5, 0.7, 0.8],
        val_loss=[0.8, 0.7, 0.5],
        val_accuracy=[0.55, 0.65, 0.75],
        best_epoch=3,
        best_val_accuracy=0.75,
    )

    output_path = tmp_path / "plots" / "history.png"

    plot_training_history(
        history=history,
        output_path=output_path,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_training_history_rejects_empty_history(
    tmp_path,
) -> None:
    history = TrainingHistory(
        train_loss=[],
        train_accuracy=[],
        val_loss=[],
        val_accuracy=[],
        best_epoch=0,
        best_val_accuracy=0.0,
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        plot_training_history(
            history=history,
            output_path=tmp_path / "history.png",
        )


def test_select_device_returns_torch_device() -> None:
    device = select_device()

    assert isinstance(device, torch.device)
    assert device.type in {"cpu", "cuda", "mps"}
    
    
def test_limited_validation_subset_is_balanced(tmp_path) -> None:
    dog_train = tmp_path / "dog_train"
    dog_val = tmp_path / "dog_val"
    ood_development = tmp_path / "ood_val"

    for index in range(8):
        create_test_image(
            dog_train / "breed_a" / f"train_{index}.jpg"
        )

    for index in range(10):
        create_test_image(
            dog_val / "breed_a" / f"val_{index}.jpg"
        )

    for index in range(10):
        create_test_image(
            ood_development / "cats" / f"ood_{index}.jpg"
        )

    _, validation_loader = build_binary_dataloaders(
        dog_train_dir=dog_train,
        dog_val_dir=dog_val,
        ood_development_dir=ood_development,
        image_size=32,
        batch_size=4,
        num_workers=0,
        seed=42,
        ood_validation_fraction=0.5,
        max_validation_samples=8,
    )

    labels = [
        label
        for _, label in validation_loader.dataset.samples
    ]

    assert labels.count(0) == 4
    assert labels.count(1) == 4
    

def test_binary_test_dataloader_contains_both_classes(
    tmp_path,
) -> None:
    dog_test = tmp_path / "dog_test"
    ood_test = tmp_path / "ood_test"

    for index in range(4):
        create_test_image(
            dog_test / "breed_a" / f"dog_{index}.jpg"
        )

    for index in range(3):
        create_test_image(
            ood_test / "cats" / f"ood_{index}.jpg"
        )

    loader = build_binary_test_dataloader(
        dog_test_dir=dog_test,
        ood_test_dir=ood_test,
        image_size=32,
        batch_size=2,
        num_workers=0,
    )

    labels = [
        label
        for _, label in loader.dataset.samples
    ]

    assert labels.count(0) == 4
    assert labels.count(1) == 3


def test_binary_test_dataloader_uses_all_samples(
    tmp_path,
) -> None:
    dog_test = tmp_path / "dog_test"
    ood_test = tmp_path / "ood_test"

    for index in range(5):
        create_test_image(
            dog_test / "breed_a" / f"dog_{index}.jpg"
        )

    for index in range(2):
        create_test_image(
            ood_test / "cars" / f"ood_{index}.jpg"
        )

    loader = build_binary_test_dataloader(
        dog_test_dir=dog_test,
        ood_test_dir=ood_test,
        image_size=32,
        batch_size=4,
        num_workers=0,
    )

    assert len(loader.dataset) == 7
    

def test_save_binary_scores_writes_npz(tmp_path) -> None:
    output_path = tmp_path / "scores" / "binary_scores.npz"

    labels = torch.tensor([0, 1, 0, 1])
    probabilities = torch.tensor([0.1, 0.9, 0.2, 0.8])

    save_binary_scores(
        labels=labels,
        ood_probabilities=probabilities,
        output_path=output_path,
    )

    saved = np.load(output_path)

    assert saved["labels"].tolist() == [0, 1, 0, 1]
    assert np.allclose(
        saved["scores"],
        [0.1, 0.9, 0.2, 0.8],
    )


def test_save_binary_scores_rejects_mismatched_lengths(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="equal length"):
        save_binary_scores(
            labels=torch.tensor([0, 1]),
            ood_probabilities=torch.tensor([0.2]),
            output_path=tmp_path / "scores.npz",
        )