"""
src/train.py
Two-stage ResNet-18 transfer learning for 120-class dog breed classification.

Stage head     — freeze backbone, train classifier head only
Stage finetune — unfreeze all layers, cosine annealing LR schedule

Usage:
    python src/train.py --stage head
    python src/train.py --stage finetune

All hyperparameters read from params.yaml. Never hardcode values here.
MLflow tracking URI set via env var: MLFLOW_TRACKING_URI=sqlite:///mlflow.db
"""

import argparse
import logging
import os

import mlflow
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import datasets, models, transforms
from tqdm import tqdm

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

def get_transforms(img_size: int, augment: bool):
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


# ── Data ─────────────────────────────────────────────────────────────────────

def get_dataloaders(data_p: dict, batch_size: int):
    """Load the three splits that src/prepare.py wrote to disk.

    Changed in PR #5: this used to point ImageFolder at data/processed/train/
    and call random_split on it. Reading the folders directly instead means
      - the split lives in DVC, so it is identical for everyone and across runs
      - it is stratified per breed (a flat random split left rare breeds thin)
      - val and test get the eval transform. The old code built one dataset with
        augment=True and split it, so val/test were randomly cropped, flipped
        and colour-jittered, which quietly inflated the loss they reported.
    """
    img_size    = data_p["img_size"]
    num_workers = data_p["num_workers"]
    out_dir     = data_p.get("out_dir", os.path.join("data", "processed"))

    split_dirs = {s: os.path.join(out_dir, s) for s in ("train", "val", "test")}
    missing = [d for d in split_dirs.values() if not os.path.isdir(d)]
    assert not missing, (
        f"Processed data not found at: {', '.join(missing)}. "
        "Run `dvc pull`, or build it with `python src/prepare.py --download` "
        "followed by `dvc repro prepare`."
    )

    train_set = datasets.ImageFolder(
        split_dirs["train"], transform=get_transforms(img_size, augment=True)
    )
    val_set = datasets.ImageFolder(
        split_dirs["val"], transform=get_transforms(img_size, augment=False)
    )
    test_set = datasets.ImageFolder(
        split_dirs["test"], transform=get_transforms(img_size, augment=False)
    )

    # A class present in train but not val/test would silently shift label ids.
    assert train_set.classes == val_set.classes == test_set.classes, (
        "Class lists differ between splits -- re-run `dvc repro prepare`."
    )

    log.info(
        f"Split — train: {len(train_set)}  val: {len(val_set)}  "
        f"test: {len(test_set)}  ({len(train_set.classes)} classes)"
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_set.classes


# ── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, dropout: float) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False          # freeze backbone
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


def load_head_checkpoint(num_classes: int, dropout: float, device: torch.device) -> nn.Module:
    ckpt = "models/resnet18_best.pt"
    assert os.path.isfile(ckpt), (
        f"Checkpoint not found at '{ckpt}'. Run --stage head first."
    )
    model = build_model(num_classes, dropout)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    log.info(f"Loaded checkpoint from {ckpt}")
    return model


# ── Metrics ──────────────────────────────────────────────────────────────────

def get_metrics(num_classes: int, device: torch.device):
    top1 = Accuracy(task="multiclass", num_classes=num_classes, top_k=1).to(device)
    top5 = Accuracy(task="multiclass", num_classes=num_classes, top_k=5).to(device)
    return top1, top5


# ── Train / Eval loops ───────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device, top1, top5):
    model.train()
    top1.reset()
    top5.reset()
    running_loss = 0.0

    for imgs, labels in tqdm(loader, desc="  train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        top1.update(logits, labels)
        top5.update(logits, labels)

    return (
        running_loss / len(loader.dataset),
        top1.compute().item(),
        top5.compute().item()
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, top1, top5):
    model.eval()
    top1.reset()
    top5.reset()
    running_loss = 0.0

    for imgs, labels in tqdm(loader, desc="  eval ", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        running_loss += loss.item() * imgs.size(0)
        top1.update(logits, labels)
        top5.update(logits, labels)

    return (
        running_loss / len(loader.dataset),
        top1.compute().item(),
        top5.compute().item()
    )


# ── Stage: head ──────────────────────────────────────────────────────────────

def stage_head(params: dict, device: torch.device):
    head_p     = params["train_head"]
    data_p     = params["data"]
    finetune_p = params["train_finetune"]
    num_classes = data_p["num_classes"]

    train_loader, val_loader, _, _ = get_dataloaders(data_p, head_p["batch_size"])
    model = build_model(num_classes, head_p["dropout"]).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=finetune_p["label_smoothing"])
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=head_p["lr"]
    )

    top1, top5 = get_metrics(num_classes, device)
    os.makedirs("models", exist_ok=True)
    best_val_top1 = 0.0

    mlflow.set_experiment("dog-breed-classifier")
    with mlflow.start_run(run_name="resnet18-head"):
        mlflow.log_params({
            "stage":       "head",
            "model":       "resnet18",
            "num_classes": num_classes,
            **{f"head_{k}": v for k, v in head_p.items()},
        })

        for epoch in range(1, head_p["epochs"] + 1):
            tr_loss, tr_top1, tr_top5 = train_one_epoch(
                model, train_loader, optimizer, criterion, device, top1, top5)
            val_loss, val_top1, val_top5 = evaluate(
                model, val_loader, criterion, device, top1, top5)

            log.info(
                f"[head] Epoch {epoch}/{head_p['epochs']} | "
                f"tr_loss={tr_loss:.4f} tr_top1={tr_top1:.4f} tr_top5={tr_top5:.4f} | "
                f"val_loss={val_loss:.4f} val_top1={val_top1:.4f} val_top5={val_top5:.4f}"
            )

            mlflow.log_metrics({
                "head_train_loss":  tr_loss,
                "head_train_top1":  tr_top1,
                "head_train_top5":  tr_top5,
                "head_val_loss":    val_loss,
                "head_val_top1":    val_top1,
                "head_val_top5":    val_top5,
            }, step=epoch)

            if val_top1 > best_val_top1:
                best_val_top1 = val_top1
                torch.save(model.state_dict(), "models/resnet18_best.pt")
                log.info(f"  ✓ New best val_top1={best_val_top1:.4f} — saved models/resnet18_best.pt")

    log.info(f"Stage head complete. Best val_top1={best_val_top1:.4f}")


# ── Stage: finetune ──────────────────────────────────────────────────────────

def stage_finetune(params: dict, device: torch.device):
    finetune_p  = params["train_finetune"]
    head_p      = params["train_head"]
    data_p      = params["data"]
    num_classes = data_p["num_classes"]

    train_loader, val_loader, _, _ = get_dataloaders(data_p, finetune_p["batch_size"])
    model = load_head_checkpoint(num_classes, head_p["dropout"], device).to(device)
    unfreeze_model(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=finetune_p["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=finetune_p["lr_initial"],
        weight_decay=finetune_p["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=finetune_p["epochs"],
        eta_min=finetune_p["lr_min"]
    )

    top1, top5 = get_metrics(num_classes, device)
    os.makedirs("models", exist_ok=True)
    best_val_top1   = 0.0
    epochs_no_improve = 0
    patience = finetune_p["early_stopping_patience"]

    mlflow.set_experiment("dog-breed-classifier")
    with mlflow.start_run(run_name="resnet18-finetune"):
        mlflow.log_params({
            "stage":       "finetune",
            "model":       "resnet18",
            "num_classes": num_classes,
            **{f"finetune_{k}": v for k, v in finetune_p.items()},
        })

        for epoch in range(1, finetune_p["epochs"] + 1):
            tr_loss, tr_top1, tr_top5 = train_one_epoch(
                model, train_loader, optimizer, criterion, device, top1, top5)
            val_loss, val_top1, val_top5 = evaluate(
                model, val_loader, criterion, device, top1, top5)
            scheduler.step()

            log.info(
                f"[finetune] Epoch {epoch}/{finetune_p['epochs']} | "
                f"tr_loss={tr_loss:.4f} tr_top1={tr_top1:.4f} tr_top5={tr_top5:.4f} | "
                f"val_loss={val_loss:.4f} val_top1={val_top1:.4f} val_top5={val_top5:.4f}"
            )

            mlflow.log_metrics({
                "finetune_train_loss":  tr_loss,
                "finetune_train_top1":  tr_top1,
                "finetune_train_top5":  tr_top5,
                "finetune_val_loss":    val_loss,
                "finetune_val_top1":    val_top1,
                "finetune_val_top5":    val_top5,
            }, step=epoch)

            if val_top1 > best_val_top1:
                best_val_top1     = val_top1
                epochs_no_improve = 0
                torch.save(model.state_dict(), "models/resnet18_best.pt")
                log.info(f"  ✓ New best val_top1={best_val_top1:.4f} — saved models/resnet18_best.pt")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    log.info(f"  Early stopping at epoch {epoch}.")
                    break

        mlflow.log_artifact("models/resnet18_best.pt")

    log.info(f"Stage finetune complete. Best val_top1={best_val_top1:.4f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train ResNet-18 dog breed classifier")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["head", "finetune"],
        help="head = train classifier only | finetune = unfreeze all and fine-tune"
    )
    args = parser.parse_args()

    params = load_params()
    device = get_device()

    if args.stage == "head":
        stage_head(params, device)
    elif args.stage == "finetune":
        stage_finetune(params, device)


if __name__ == "__main__":
    main()