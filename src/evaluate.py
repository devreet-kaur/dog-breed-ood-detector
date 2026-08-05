"""
src/evaluate.py
Test set evaluation for the ResNet-18 dog breed classifier.

Owner: Arushi (ML Lead) -- assisted by Devreet

Outputs:
    reports/metrics_test.json          -- top1_acc, top5_acc
    reports/plots/confusion_matrix.png -- 120x120 confusion matrix
    reports/plots/training_curves.png  -- val top1 across epochs from MLflow

Usage:
    python src/evaluate.py

All hyperparameters read from params.yaml. Never hardcode values here.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import datasets, models, transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PARAMS_PATH     = Path("params.yaml")
MODEL_PATH      = Path("models/resnet18_best.pt")
TEST_DIR        = Path("data/processed/test")
METRICS_PATH    = Path("reports/metrics_test.json")
PLOTS_DIR       = Path("reports/plots")


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(num_classes: int, device: torch.device) -> nn.Module:
    params  = load_params()
    dropout = params["train_head"]["dropout"]
    model   = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    log.info("Model loaded from %s", MODEL_PATH)
    return model


def get_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> tuple[float, float, np.ndarray]:
    top1_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)
    top5_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)

            top1_metric(logits, labels)
            top5_metric(logits, labels)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    top1 = top1_metric.compute().item()
    top5 = top5_metric.compute().item()

    conf_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    for true, pred in zip(all_labels, all_preds):
        conf_matrix[true][pred] += 1

    return top1, top5, conf_matrix


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    class_names: list[str],
    out_path: Path,
) -> None:
    _fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_title("Confusion Matrix -- ResNet-18 Dog Breed Classifier", fontsize=14)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=5)
    ax.set_yticklabels(class_names, fontsize=5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info("Confusion matrix saved to %s", out_path)


def plot_training_curves(out_path: Path) -> None:
    """Pull val_top1 per epoch from MLflow and plot training curves."""
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient(tracking_uri="sqlite:///mlflow.db")
        experiments = client.search_experiments(filter_string="name = 'dog-breed-classifier'")
        if not experiments:
            log.warning("No MLflow experiment found -- skipping training curves")
            return

        exp_id = experiments[0].experiment_id
        runs   = client.search_runs(
            experiment_ids=[exp_id],
            order_by=["start_time ASC"],
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("ResNet-18 Training Curves", fontsize=14)

        for run in runs:
            stage     = run.data.tags.get("stage", run.info.run_name or "run")
            val_top1  = client.get_metric_history(run.info.run_id, "val_top1")
            val_top5  = client.get_metric_history(run.info.run_id, "val_top5")
            tr_loss   = client.get_metric_history(run.info.run_id, "train_loss")

            epochs_v1 = [m.step for m in val_top1]
            vals_v1   = [m.value for m in val_top1]
            epochs_v5 = [m.step for m in val_top5]
            vals_v5   = [m.value for m in val_top5]
            epochs_l  = [m.step for m in tr_loss]
            vals_l    = [m.value for m in tr_loss]

            if vals_v1:
                axes[0].plot(epochs_v1, vals_v1, marker="o", label=f"{stage} val top1")
            if vals_v5:
                axes[0].plot(epochs_v5, vals_v5, marker="s", linestyle="--",
                             label=f"{stage} val top5")
            if vals_l:
                axes[1].plot(epochs_l, vals_l, marker="o", label=f"{stage} train loss")

        axes[0].set_title("Validation Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].set_title("Training Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        log.info("Training curves saved to %s", out_path)

    except Exception as exc:  # noqa: BLE001
        log.warning("Could not plot training curves: %s", exc)


def main() -> None:
    params      = load_params()
    device      = get_device()
    img_size    = params["data"]["img_size"]
    batch_size  = params["train_head"]["batch_size"]
    num_workers = params["data"]["num_workers"]

    log.info("Device: %s", device)

    transform   = get_transform(img_size)
    test_ds     = datasets.ImageFolder(str(TEST_DIR), transform=transform)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    num_classes  = len(test_ds.classes)
    class_names  = test_ds.classes
    log.info("Test set: %d images, %d classes", len(test_ds), num_classes)

    model = load_model(num_classes, device)

    log.info("Running evaluation on test set...")
    top1, top5, conf_matrix = evaluate(model, test_loader, device, num_classes)

    log.info("Top-1 accuracy: %.4f (%.2f%%)", top1, top1 * 100)
    log.info("Top-5 accuracy: %.4f (%.2f%%)", top5, top5 * 100)

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "top1_acc": round(top1, 6),
        "top5_acc": round(top5, 6),
        "top1_pct": round(top1 * 100, 2),
        "top5_pct": round(top5 * 100, 2),
        "num_test_images": len(test_ds),
        "num_classes": num_classes,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Metrics saved to %s", METRICS_PATH)

    plot_confusion_matrix(
        conf_matrix=conf_matrix,
        class_names=class_names,
        out_path=PLOTS_DIR / "confusion_matrix.png",
    )

    plot_training_curves(out_path=PLOTS_DIR / "training_curves.png")

    print("\n── Evaluation Results ────────────────────────")
    print(f"  Top-1 accuracy: {top1 * 100:.2f}%")
    print(f"  Top-5 accuracy: {top5 * 100:.2f}%")
    print(f"  Test images:    {len(test_ds)}")
    print(f"  Classes:        {num_classes}")
    print(f"  Metrics saved:  {METRICS_PATH}")


if __name__ == "__main__":
    main()
