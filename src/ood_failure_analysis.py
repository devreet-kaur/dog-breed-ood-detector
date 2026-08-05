"""
ood_failure_analysis.py
Deep failure analysis for OOD detection strategies A and B.
Owner: Devreet

Requires:
    - models/resnet18_best.pt          (from Arushi)
    - models/temperature.json          (from calibrate.py)
    - reports/ood/strategy_a_scores.json (from Soodeh)
    - reports/ood/strategy_b_scores.json (from Soodeh)

Outputs:
    reports/ood_analysis/
        entropy_distributions.png      -- per-category entropy histograms
        confidence_by_breed.png        -- top-20 breeds by mean confidence
        auroc_comparison.png           -- Strategy A vs B ROC curves
        failure_summary.json           -- which OOD categories fool each strategy most
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.datasets import ImageFolder

OOD_CATEGORIES = ["cats", "birds", "cars", "food", "furniture"]
PALETTE = {
    "cats":      "#378ADD",
    "birds":     "#1D9E75",
    "cars":      "#D85A30",
    "food":      "#BA7517",
    "furniture": "#7F77DD",
    "in_dist":   "#888780",
}


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path) as f:
        return yaml.safe_load(f)


def load_model(model_path: str, num_classes: int, device: torch.device,
               temperature: float = 1.0) -> tuple[nn.Module, float]:
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.fc.in_features, num_classes)
    )
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, temperature


def entropy(probs: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of a probability distribution. Higher = more uncertain."""
    log_probs = torch.log(probs + 1e-8)
    return -(probs * log_probs).sum(dim=1)



class FlatImageFolder(Dataset):
    """Dataset for a flat folder of images (no class subdirectories).
    download_ood.py writes OOD images directly as <category>/*.jpg,
    so torchvision's ImageFolder (which requires class subfolders) can't
    be used here. Labels are dummy (0) since OOD scoring doesn't need them.
    """
    def __init__(self, root, transform=None):
        self.paths = sorted(
            p for p in Path(root).iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


def collect_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> dict:
    all_entropy   = []
    all_max_prob  = []
    all_labels    = []
    all_top1      = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits / temperature, dim=1)

            all_entropy.append(entropy(probs).cpu())
            all_max_prob.append(probs.max(dim=1).values.cpu())
            all_top1.append(probs.argmax(dim=1).cpu())
            all_labels.append(labels)

    return {
        "entropy":  torch.cat(all_entropy).numpy(),
        "max_prob": torch.cat(all_max_prob).numpy(),
        "top1":     torch.cat(all_top1).numpy(),
        "labels":   torch.cat(all_labels).numpy(),
    }


def plot_entropy_distributions(
    in_dist_scores: dict,
    ood_scores_by_cat: dict[str, dict],
    threshold_95tpr: float,
    out_path: Path,
) -> None:
    _, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    axes[0].hist(in_dist_scores["entropy"], bins=50,
                 color=PALETTE["in_dist"], alpha=0.8, density=True)
    axes[0].axvline(threshold_95tpr, color="red", linestyle="--", linewidth=1.5,
                    label=f"Threshold={threshold_95tpr:.2f}")
    axes[0].set_title("In-distribution (Stanford Dogs)")
    axes[0].set_xlabel("Entropy")
    axes[0].legend()

    for i, cat in enumerate(OOD_CATEGORIES, start=1):
        scores = ood_scores_by_cat.get(cat, {})
        if not scores:
            continue
        axes[i].hist(scores["entropy"], bins=50,
                     color=PALETTE[cat], alpha=0.8, density=True)
        axes[i].axvline(threshold_95tpr, color="red", linestyle="--", linewidth=1.5)

        above = (scores["entropy"] > threshold_95tpr).mean()
        axes[i].set_title(f"OOD: {cat}\nDetected as OOD: {above:.1%}")
        axes[i].set_xlabel("Entropy")

    plt.suptitle("Entropy distributions by category (Strategy A)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_auroc_comparison(
    strategy_a_labels: np.ndarray,
    strategy_a_scores: np.ndarray,
    strategy_b_labels: np.ndarray,
    strategy_b_scores: np.ndarray,
    out_path: Path,
) -> None:
    fpr_a, tpr_a, _ = roc_curve(strategy_a_labels, strategy_a_scores)
    fpr_b, tpr_b, _ = roc_curve(strategy_b_labels, strategy_b_scores)
    auc_a = roc_auc_score(strategy_a_labels, strategy_a_scores)
    auc_b = roc_auc_score(strategy_b_labels, strategy_b_scores)

    _, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr_a, tpr_a, color="#378ADD", linewidth=2,
            label=f"Strategy A entropy (AUROC={auc_a:.3f})")
    ax.plot(fpr_b, tpr_b, color="#D85A30", linewidth=2,
            label=f"Strategy B binary CNN (AUROC={auc_b:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve: Strategy A vs Strategy B")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def plot_confidence_by_breed(
    scores: dict,
    class_names: list[str],
    out_path: Path,
    top_n: int = 20,
) -> None:
    mean_conf = {}
    for cls_idx, name in enumerate(class_names):
        mask = scores["labels"] == cls_idx
        if mask.sum() == 0:
            continue
        mean_conf[name] = scores["max_prob"][mask].mean()

    sorted_breeds = sorted(mean_conf.items(), key=lambda x: x[1])[:top_n]
    breeds, confs = zip(*sorted_breeds)

    _, ax = plt.subplots(figsize=(8, 8))
    ax.barh(breeds, confs, color="#7F77DD", alpha=0.8)
    ax.axvline(0.5, color="red", linestyle="--", linewidth=1, label="50% confidence")
    ax.set_xlabel("Mean top-1 confidence")
    ax.set_title(f"Bottom {top_n} breeds by confidence (hardest)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved: {out_path}")


def compute_threshold_at_tpr(
    in_entropy: np.ndarray,
    target_tpr: float = 0.95,
) -> float:
    """Find entropy threshold that catches target_tpr of in-distribution samples."""
    return float(np.percentile(in_entropy, (1 - target_tpr) * 100))


def main():
    params = load_params()
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    temp_path = Path("models/temperature.json")
    temperature = 1.0
    if temp_path.exists():
        with open(temp_path) as f:
            temperature = json.load(f)["temperature"]
        print(f"Using temperature: {temperature:.4f}")
    else:
        print("No temperature.json found, using T=1.0 (uncalibrated)")

    with open("data/processed/class_names.json") as f:
        class_names = json.load(f)

    img_size   = params.get("img_size", 224)
    batch_size = params.get("batch_size", 64)
    transform  = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model, temperature = load_model("models/resnet18_best.pt",
                                    len(class_names), device, temperature)

    print("Scoring in-distribution test set...")
    in_dist_ds     = ImageFolder("data/processed/test", transform=transform)
    in_dist_loader = DataLoader(in_dist_ds, batch_size=batch_size,
                                shuffle=False, num_workers=2)
    in_scores = collect_scores(model, in_dist_loader, device, temperature)

    ood_scores_by_cat = {}
    for cat in OOD_CATEGORIES:
        cat_dir = Path(f"data/raw/ood/test/{cat}")
        if not cat_dir.exists():
            print(f"Skipping {cat}: directory not found")
            continue
        print(f"Scoring OOD category: {cat}...")
        ds     = FlatImageFolder(cat_dir, transform=transform)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
        ood_scores_by_cat[cat] = collect_scores(model, loader, device, temperature)

    threshold_95tpr = compute_threshold_at_tpr(in_scores["entropy"], target_tpr=params.get("ood_entropy", {}).get("target_tpr", 0.95))
    print(f"Entropy threshold at 95% TPR: {threshold_95tpr:.4f}")

    out_dir = Path("reports/ood_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_entropy_distributions(
        in_dist_scores=in_scores,
        ood_scores_by_cat=ood_scores_by_cat,
        threshold_95tpr=threshold_95tpr,
        out_path=out_dir / "entropy_distributions.png",
    )

    plot_confidence_by_breed(
        scores=in_scores,
        class_names=class_names,
        out_path=out_dir / "confidence_by_breed.png",
    )

    failure_summary = {}
    for cat, scores in ood_scores_by_cat.items():
        detected = (scores["entropy"] > threshold_95tpr).mean()
        failure_summary[cat] = {
            "detection_rate":     round(float(detected), 4),
            "mean_entropy":       round(float(scores["entropy"].mean()), 4),
            "mean_max_prob":      round(float(scores["max_prob"].mean()), 4),
            "hardest_category":   cat if detected < 0.5 else None,
        }
        print(f"{cat}: detected {detected:.1%} as OOD")

    failure_summary["threshold_95tpr"] = round(threshold_95tpr, 4)
    failure_summary["temperature"]     = round(temperature, 4)

    with open(out_dir / "failure_summary.json", "w") as f:
        json.dump(failure_summary, f, indent=2)

    print(f"\nAll outputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
