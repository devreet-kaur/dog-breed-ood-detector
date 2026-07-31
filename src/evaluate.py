"""
src/evaluate.py
Test set evaluation for the trained ResNet-18 dog breed classifier.

Outputs:
    reports/metrics_test.json        — top1_acc, top5_acc, test_loss
    reports/plots/confusion_matrix.png
    reports/plots/training_curves.png  (pulled from MLflow)

Usage (called by DVC pipeline):
    python src/evaluate.py

All hyperparameters read from params.yaml.
"""

import os
import json
import yaml
import torch
import torch.nn as nn
import mlflow
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Params ───────────────────────────────────────────────────────────────────

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")
    return device


# ── Transforms ───────────────────────────────────────────────────────────────

def get_eval_transforms(img_size: int):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


# ── Data ─────────────────────────────────────────────────────────────────────

def get_test_loader(data_p: dict) -> tuple:
    img_size   = data_p["img_size"]
    val_split  = data_p["val_split"]
    test_split = data_p["test_split"]
    num_workers = data_p["num_workers"]

    train_dir = os.path.join("data", "processed", "train")
    assert os.path.isdir(train_dir), (
        f"Processed data not found at '{train_dir}'. "
        "Run Ryan's prepare.py and dvc pull first."
    )

    full_dataset = datasets.ImageFolder(
        train_dir, transform=get_eval_transforms(img_size)
    )
    n = len(full_dataset)
    n_val   = int(n * val_split)
    n_test  = int(n * test_split)
    n_train = n - n_val - n_test

    _, _, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(
        test_set, batch_size=32, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    log.info(f"Test set size: {n_test} images | {len(full_dataset.classes)} classes")
    return test_loader, full_dataset.classes


# ── Model ────────────────────────────────────────────────────────────────────

def load_model(num_classes: int, dropout: float, device: torch.device) -> nn.Module:
    ckpt = "models/resnet18_best.pt"
    assert os.path.isfile(ckpt), (
        f"Model checkpoint not found at '{ckpt}'. "
        "Run src/train.py --stage head and --stage finetune first."
    )
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    log.info(f"Loaded model from {ckpt}")
    return model


# ── Evaluation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(model, loader, criterion, device, num_classes):
    top1_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)
    top5_metric = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)

    all_preds  = []
    all_labels = []
    running_loss = 0.0

    for imgs, labels in tqdm(loader, desc="  evaluating"):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)

        top1_metric.update(logits, labels)
        top5_metric.update(logits, labels)

        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(loader.dataset)
    top1_acc  = top1_metric.compute().item()
    top5_acc  = top5_metric.compute().item()

    return test_loss, top1_acc, top5_acc, np.array(all_preds), np.array(all_labels)


# ── Confusion Matrix ─────────────────────────────────────────────────────────

def plot_confusion_matrix(preds, labels, classes, out_path: str):
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds)
    # Show only top-20 most confused classes for readability
    class_totals = cm.sum(axis=1)
    top20_idx    = np.argsort(class_totals)[-20:]
    cm_top20     = cm[np.ix_(top20_idx, top20_idx)]
    top20_names  = [classes[i] for i in top20_idx]

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(
        cm_top20, annot=True, fmt="d", cmap="Blues",
        xticklabels=top20_names, yticklabels=top20_names,
        ax=ax, linewidths=0.4
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Top 20 Most Frequent Classes", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Confusion matrix saved to {out_path}")


# ── Metrics JSON ─────────────────────────────────────────────────────────────

def save_metrics(test_loss: float, top1_acc: float, top5_acc: float, out_path: str):
    metrics = {
        "test_loss": round(test_loss, 6),
        "top1_acc":  round(top1_acc, 6),
        "top5_acc":  round(top5_acc, 6),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Metrics saved to {out_path}")
    return metrics


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    params      = load_params()
    device      = get_device()
    data_p      = params["data"]
    head_p      = params["train_head"]
    finetune_p  = params["train_finetune"]
    num_classes = data_p["num_classes"]

    os.makedirs("reports/plots", exist_ok=True)

    test_loader, classes = get_test_loader(data_p)
    model     = load_model(num_classes, head_p["dropout"], device)
    criterion = nn.CrossEntropyLoss(label_smoothing=finetune_p["label_smoothing"])

    test_loss, top1_acc, top5_acc, preds, labels = run_evaluation(
        model, test_loader, criterion, device, num_classes
    )

    log.info(f"Test loss : {test_loss:.4f}")
    log.info(f"Top-1 acc : {top1_acc:.4f}")
    log.info(f"Top-5 acc : {top5_acc:.4f}")

    # Save metrics JSON
    metrics = save_metrics(
        test_loss, top1_acc, top5_acc,
        out_path="reports/metrics_test.json"
    )

    # Confusion matrix
    plot_confusion_matrix(
        preds, labels, classes,
        out_path="reports/plots/confusion_matrix.png"
    )

    # Log to MLflow
    mlflow.set_experiment("dog-breed-classifier")
    with mlflow.start_run(run_name="resnet18-evaluate"):
        mlflow.log_params({
            "stage":       "evaluate",
            "model":       "resnet18",
            "num_classes": num_classes,
        })
        mlflow.log_metrics(metrics)
        mlflow.log_artifact("reports/metrics_test.json")
        mlflow.log_artifact("reports/plots/confusion_matrix.png")

    log.info("Evaluation complete.")


if __name__ == "__main__":
    main()