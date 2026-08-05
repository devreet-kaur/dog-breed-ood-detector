"""
calibrate.py
Post-hoc temperature scaling calibration for the dog breed classifier.
Owner: Devreet

Temperature scaling learns a single scalar T on the validation set such that
softmax(logits / T) is better calibrated (ECE minimized). Better calibration
directly improves Strategy A OOD entropy thresholding accuracy.

Reference: Guo et al. (2017) "On Calibration of Modern Neural Networks"

Usage:
    python src/calibrate.py
    python src/calibrate.py --model models/resnet18_best.pt --out models/temperature.json

Outputs:
    models/temperature.json         -- learned temperature value
    reports/calibration/            -- reliability diagrams before and after
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_model(model_path: str, num_classes: int, device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model


def get_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run full val set through the model, collect raw logits and true labels."""
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


def expected_calibration_error(
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    ECE: expected calibration error.
    Bins predictions by confidence, measures gap between confidence and accuracy per bin.
    Lower is better. Perfect calibration = 0.
    """
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels)

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean().item()
        bin_acc  = correct[mask].float().mean().item()
        bin_size = mask.sum().item()
        ece += (bin_size / len(labels)) * abs(bin_acc - bin_conf)

    return ece


def learn_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    max_iter: int = 50,
    lr: float = 0.01,
) -> float:
    """
    Learn a single scalar temperature T by minimizing NLL on the val set.
    T > 1 softens the distribution (reduces overconfidence).
    T < 1 sharpens it (rare but possible).
    """
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    optimizer   = optim.LBFGS([temperature], lr=lr, max_iter=max_iter)
    nll_loss    = nn.CrossEntropyLoss()

    def eval_step():
        optimizer.zero_grad()
        scaled_logits = logits / temperature
        loss = nll_loss(scaled_logits, labels)
        loss.backward()
        return loss

    optimizer.step(eval_step)

    return temperature.item()


def reliability_diagram(
    probs: torch.Tensor,
    labels: torch.Tensor,
    title: str,
    save_path: Path,
    n_bins: int = 15,
) -> None:
    """Plot confidence vs accuracy reliability diagram."""
    confidences, predictions = probs.max(dim=1)
    correct = predictions.eq(labels).float()

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_mids       = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    bin_acc        = []
    bin_conf       = []
    bin_count      = []

    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences.numpy() > lo) & (confidences.numpy() <= hi)
        if mask.sum() == 0:
            bin_acc.append(0)
            bin_conf.append((lo + hi) / 2)
            bin_count.append(0)
        else:
            bin_acc.append(correct.numpy()[mask].mean())
            bin_conf.append(confidences.numpy()[mask].mean())
            bin_count.append(mask.sum())

    _, ax = plt.subplots(figsize=(6, 6))
    ax.bar(bin_mids, bin_acc, width=1 / n_bins, alpha=0.7,
           color="#378ADD", edgecolor="white", label="Model accuracy")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Temperature scaling calibration")
    parser.add_argument("--model",    type=str, default="models/resnet18_best.pt")
    parser.add_argument("--val_dir",  type=str, default="data/processed/val")
    parser.add_argument("--out",      type=str, default="models/temperature.json")
    parser.add_argument("--params",   type=str, default="params.yaml")
    parser.add_argument("--out_dir",  type=str, default="reports/calibration")
    args = parser.parse_args()

    params = load_params(args.params)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    img_size   = params["data"]["img_size"]
    batch_size = params["train_head"]["batch_size"]

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = ImageFolder(args.val_dir, transform=transform)
    val_loader  = DataLoader(val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=2)

    num_classes = len(val_dataset.classes)
    model       = load_model(args.model, num_classes, device)

    print("Collecting logits from val set...")
    logits, labels = get_logits_and_labels(model, val_loader, device)

    probs_before = torch.softmax(logits, dim=1)
    ece_before   = expected_calibration_error(probs_before, labels)
    print(f"ECE before calibration: {ece_before:.4f}")

    print("Learning temperature...")
    temperature = learn_temperature(logits, labels)
    print(f"Learned temperature: {temperature:.4f}")

    probs_after = torch.softmax(logits / temperature, dim=1)
    ece_after   = expected_calibration_error(probs_after, labels)
    print(f"ECE after calibration:  {ece_after:.4f}")
    print(f"ECE reduction: {(ece_before - ece_after):.4f} ({(ece_before - ece_after)/ece_before:.1%} improvement)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reliability_diagram(
        probs_before, labels,
        title=f"Before calibration (ECE={ece_before:.4f})",
        save_path=out_dir / "reliability_before.png",
    )
    reliability_diagram(
        probs_after, labels,
        title=f"After calibration T={temperature:.2f} (ECE={ece_after:.4f})",
        save_path=out_dir / "reliability_after.png",
    )

    result = {
        "temperature":  temperature,
        "ece_before":   round(ece_before, 6),
        "ece_after":    round(ece_after, 6),
        "ece_reduction": round(ece_before - ece_after, 6),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nTemperature saved: {args.out}")
    print(f"Reliability diagrams saved: {args.out_dir}/")


if __name__ == "__main__":
    main()
