"""
visualize_gradcam.py
Grad-CAM heatmap generation for the dog breed classifier.
Owner: Devreet

Usage:
    python src/visualize_gradcam.py --image path/to/image.jpg
    python src/visualize_gradcam.py --image path/to/image.jpg --model models/resnet18_best.pt
    python src/visualize_gradcam.py --batch data/processed/test/ --top_k 5

Outputs:
    reports/gradcam/<image_stem>_gradcam.png  -- original + heatmap overlay side by side
    reports/gradcam/failures/                 -- images where top-1 was wrong
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision import models


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


def get_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Denormalize and convert tensor to HxWx3 float32 [0,1] for Grad-CAM overlay."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = tensor.cpu().squeeze(0) * std + mean
    img = img.permute(1, 2, 0).numpy()
    return np.clip(img, 0, 1).astype(np.float32)


def run_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    """
    Run Grad-CAM on the last convolutional block of ResNet-18.
    Returns heatmap as HxW float32 [0,1].
    """
    target_layer = model.layer4[-1]
    targets = [ClassifierOutputTarget(target_class)]

    with GradCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

    return grayscale_cam[0]


def generate_and_save(
    image_path: Path,
    model: nn.Module,
    transform: T.Compose,
    class_names: list[str],
    out_dir: Path,
    device: torch.device,
    true_label: str | None = None,
) -> dict:
    """
    Generate Grad-CAM for a single image.
    Returns prediction metadata dict.
    """
    pil_img = Image.open(image_path).convert("RGB")
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1)
        top5   = torch.topk(probs, k=min(5, len(class_names)), dim=1)

    top1_idx   = top5.indices[0, 0].item()
    top1_prob  = top5.values[0, 0].item()
    top1_label = class_names[top1_idx]

    heatmap  = run_gradcam(model, input_tensor, top1_idx)
    rgb_img  = tensor_to_rgb(input_tensor)
    overlay  = show_cam_on_image(rgb_img, heatmap, use_rgb=True)

    original_uint8 = (rgb_img * 255).astype(np.uint8)
    overlay_uint8  = overlay.astype(np.uint8)

    gap = np.ones((224, 8, 3), dtype=np.uint8) * 240
    side_by_side = np.concatenate([original_uint8, gap, overlay_uint8], axis=1)

    correct = (true_label is None) or (true_label == top1_label)
    save_dir = out_dir if correct else out_dir / "failures"
    save_dir.mkdir(parents=True, exist_ok=True)

    out_path = save_dir / f"{image_path.stem}_gradcam.png"
    Image.fromarray(side_by_side).save(out_path)

    result = {
        "image":      str(image_path),
        "pred_label": top1_label,
        "pred_prob":  round(top1_prob, 4),
        "true_label": true_label,
        "correct":    correct,
        "saved_to":   str(out_path),
        "top5": [
            {
                "label": class_names[top5.indices[0, i].item()],
                "prob":  round(top5.values[0, i].item(), 4),
            }
            for i in range(top5.indices.shape[1])
        ],
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM visualization for dog breed classifier")
    parser.add_argument("--image",  type=str, help="Path to a single image")
    parser.add_argument("--batch",  type=str, help="Path to a directory of images")
    parser.add_argument("--model",  type=str, default="models/resnet18_best.pt")
    parser.add_argument("--params", type=str, default="params.yaml")
    parser.add_argument("--labels", type=str, default="data/processed/class_names.json")
    parser.add_argument("--out",    type=str, default="reports/gradcam")
    parser.add_argument("--top_k",  type=int, default=None,
                        help="Only process top_k images (for batch mode)")
    args = parser.parse_args()

    params = load_params(args.params)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    from torchvision.datasets import ImageFolder
    class_names = ImageFolder("data/processed/train").classes

    num_classes = len(class_names)
    model       = load_model(args.model, num_classes, device)
    transform   = get_transform(params.get("img_size", 224))
    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if args.image:
        result = generate_and_save(
            image_path=Path(args.image),
            model=model,
            transform=transform,
            class_names=class_names,
            out_dir=out_dir,
            device=device,
        )
        results.append(result)
        print(f"Pred: {result['pred_label']} ({result['pred_prob']:.2%})")
        print(f"Saved: {result['saved_to']}")

    elif args.batch:
        image_paths = sorted(Path(args.batch).rglob("*.jpg"))
        image_paths += sorted(Path(args.batch).rglob("*.jpeg"))
        image_paths += sorted(Path(args.batch).rglob("*.png"))

        if args.top_k:
            image_paths = image_paths[: args.top_k]

        print(f"Processing {len(image_paths)} images...")

        for i, img_path in enumerate(image_paths):
            true_label = img_path.parent.name
            result = generate_and_save(
                image_path=img_path,
                model=model,
                transform=transform,
                class_names=class_names,
                out_dir=out_dir,
                device=device,
                true_label=true_label,
            )
            results.append(result)
            status = "OK" if result["correct"] else "FAIL"
            print(f"[{i+1}/{len(image_paths)}] {status} {img_path.name} -> {result['pred_label']}")

    results_path = out_dir / "gradcam_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    if results:
        n_correct = sum(r["correct"] for r in results if r["true_label"] is not None)
        n_total   = sum(1 for r in results if r["true_label"] is not None)
        if n_total:
            print(f"\nAccuracy: {n_correct}/{n_total} ({n_correct/n_total:.1%})")
        print(f"Results saved: {results_path}")
        print(f"Failures saved: {out_dir}/failures/")


if __name__ == "__main__":
    main()
