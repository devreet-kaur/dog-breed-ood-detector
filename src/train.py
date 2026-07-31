"""
src/train.py
Two-stage ResNet50 transfer learning trainer for dog breed classification.
Stage 1 (train_head)   — freeze backbone, train classifier head only
Stage 2 (train_finetune) — unfreeze all, cosine-annealing fine-tune

Usage (called by DVC pipeline):
    python src/train.py

All hyperparameters are read from params.yaml — never hardcode values here.
"""

import os
import yaml
import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchmetrics import Accuracy
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Load params ────────────────────────────────────────────────────────────

def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

# ─── Device ─────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"Using device: {device}")
    return device

# ─── Data ───────────────────────────────────────────────────────────────────

def get_transforms(img_size: int, augment: bool = True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def get_dataloaders(data_params: dict):
    img_size    = data_params["img_size"]
    val_split   = data_params["val_split"]
    test_split  = data_params["test_split"]
    num_workers = data_params["num_workers"]
    batch_size  = 32  # default; overridden per stage

    train_dir = os.path.join("data", "processed", "train")
    assert os.path.isdir(train_dir), (
        f"Processed data not found at '{train_dir}'. "
        "Run Ryan's prepare.py (feat/data-pipeline) first."
    )

    full_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(img_size, augment=True))
    n = len(full_dataset)
    n_val  = int(n * val_split)
    n_test = int(n * test_split)
    n_train = n - n_val - n_test

    train_set, val_set, test_set = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # val/test use no-augment transforms
    val_set.dataset  = datasets.ImageFolder(train_dir, transform=get_transforms(img_size, augment=False))
    test_set.dataset = datasets.ImageFolder(train_dir, transform=get_transforms(img_size, augment=False))

    log.info(f"Dataset split — train: {n_train}, val: {n_val}, test: {n_test}")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, full_dataset.classes

# ─── Model ──────────────────────────────────────────────────────────────────

def build_model(num_classes: int, dropout: float) -> nn.Module:
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Freeze all backbone layers
    for param in model.parameters():
        param.requires_grad = False
    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


def unfreeze_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
    log.info("All layers unfrozen for fine-tuning.")

# ─── Train / Eval loops ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, accuracy_metric):
    model.train()
    accuracy_metric.reset()
    running_loss = 0.0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        accuracy_metric.update(logits, labels)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_metric.compute().item()
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device, accuracy_metric):
    model.eval()
    accuracy_metric.reset()
    running_loss = 0.0

    for imgs, labels in tqdm(loader, desc="  eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        accuracy_metric.update(logits, labels)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc  = accuracy_metric.compute().item()
    return epoch_loss, epoch_acc

# ─── Stage runners ──────────────────────────────────────────────────────────

def run_stage(stage_name, model, train_loader, val_loader,
              optimizer, scheduler, criterion, device,
              num_classes, epochs, patience, params):

    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
    best_val_acc = 0.0
    epochs_no_improve = 0
    os.makedirs("models", exist_ok=True)
    best_path = f"models/best_{stage_name}.pt"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, accuracy_metric)
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, accuracy_metric)

        if scheduler:
            scheduler.step()

        log.info(
            f"[{stage_name}] Epoch {epoch}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        mlflow.log_metrics({
            f"{stage_name}_train_loss": train_loss,
            f"{stage_name}_train_acc":  train_acc,
            f"{stage_name}_val_loss":   val_loss,
            f"{stage_name}_val_acc":    val_acc,
        }, step=epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_path)
            log.info(f"  ✓ New best val_acc={best_val_acc:.4f} — saved to {best_path}")
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                log.info(f"  Early stopping at epoch {epoch}.")
                break

    # Reload best weights before returning
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model, best_val_acc

# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    params = load_params()
    device = get_device()

    data_p     = params["data"]
    head_p     = params["train_head"]
    finetune_p = params["train_finetune"]

    train_loader, val_loader, test_loader, classes = get_dataloaders(data_p)
    num_classes = data_p["num_classes"]
    assert len(classes) == num_classes, (
        f"Expected {num_classes} classes, found {len(classes)} in data/processed/train"
    )

    model = build_model(num_classes, dropout=head_p["dropout"]).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=finetune_p["label_smoothing"])

    mlflow.set_experiment("dog-breed-classifier")
    with mlflow.start_run(run_name="resnet50-two-stage"):
        mlflow.log_params({
            "model":        "resnet50",
            "num_classes":  num_classes,
            **{f"head_{k}": v     for k, v in head_p.items()},
            **{f"finetune_{k}": v for k, v in finetune_p.items()},
        })

        # ── Stage 1: train head only ──────────────────────────────────────
        log.info("=== Stage 1: Training classifier head ===")
        optimizer_head = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=head_p["lr"]
        )
        model, best_head_acc = run_stage(
            stage_name="head",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer_head,
            scheduler=None,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            epochs=head_p["epochs"],
            patience=None,
            params=params,
        )
        log.info(f"Stage 1 complete — best val_acc: {best_head_acc:.4f}")

        # ── Stage 2: fine-tune all layers ────────────────────────────────
        log.info("=== Stage 2: Fine-tuning all layers ===")
        unfreeze_model(model)
        optimizer_ft = torch.optim.AdamW(
            model.parameters(),
            lr=finetune_p["lr_initial"],
            weight_decay=finetune_p["weight_decay"]
        )
        scheduler_ft = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer_ft,
            T_max=finetune_p["epochs"],
            eta_min=finetune_p["lr_min"]
        )
        model, best_ft_acc = run_stage(
            stage_name="finetune",
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer_ft,
            scheduler=scheduler_ft,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            epochs=finetune_p["epochs"],
            patience=finetune_p["early_stopping_patience"],
            params=params,
        )
        log.info(f"Stage 2 complete — best val_acc: {best_ft_acc:.4f}")

        # ── Final test evaluation ────────────────────────────────────────
        log.info("=== Final test evaluation ===")
        accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device, accuracy_metric)
        log.info(f"Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
        mlflow.log_metrics({"test_loss": test_loss, "test_acc": test_acc})

        # ── Save final model ─────────────────────────────────────────────
        os.makedirs("models", exist_ok=True)
        final_path = "models/resnet50_final.pt"
        torch.save(model.state_dict(), final_path)
        mlflow.log_artifact(final_path)
        log.info(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()