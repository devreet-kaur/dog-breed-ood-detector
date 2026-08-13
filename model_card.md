# Model Card: Dog Breed Classifier with OOD Detection

**Project:** MAI202 Deep Learning, Seneca Polytechnic
**Repo:** devreet-kaur/dog-breed-ood-detector
**Owners:** Ryan Caezar Soria (data), Arushi Anand (model), Soodeh Vanaki (OOD), Devreet Kaur (deployment)
**Date:** August 2026

---

## Model Details

| | |
|---|---|
| Architecture | ResNet-18, pretrained ImageNet, fine-tuned |
| Task | 120-class dog breed classification + OOD gating |
| Framework | PyTorch |
| Input | RGB image, 224x224 |
| Output | Breed prediction (or null if OOD), confidence, top-5, entropy |
| Training hardware | Apple Silicon MPS |
| Checkpoint | `models/resnet18_best.pt` (DVC-tracked) |

### Training procedure
Two-stage fine-tuning:
1. **Head-only**: backbone frozen, 5 epochs, lr=1e-3, dropout=0.4
2. **Full fine-tune**: cosine annealing, lr_initial=1e-4, lr_min=1e-6, weight_decay=1e-4, label_smoothing=0.1, early stopping (patience=7)

All hyperparameters in `params.yaml`, no hardcoded values in training code.

---

## Training Data

**Stanford Dogs**: 120 breeds, 20,580 images, bounding-box crops applied before resize. Split 70/15/15 (train/val/test), stratified per breed.
- Train: 14,397 images
- Val: 3,084 images
- Test: 3,099 images

**OOD data**: 5 categories (cats, birds, cars, food, furniture), 250 images each, sourced from an ImageNet subset. 80/20 split: val (250, threshold calibration) / test (1,000, held-out evaluation).

---

## Performance

### Breed Classification (test set, 3,099 images, 120 classes)

| Metric | Value |
|---|---|
| Top-1 accuracy | 76.22% |
| Top-5 accuracy | 96.87% |

**Training progression:**
- Stage 1a (head only) best val top-1: 72.86%
- Stage 1b (finetune) best val top-1: 78.79%, early stop epoch 11/30

### Calibration
Post-hoc temperature scaling (Guo et al., 2017), LBFGS minimizing NLL on val set.

| | |
|---|---|
| Temperature (T) | 1.0167 |
| ECE before | 0.1215 |
| ECE after | 0.1287 |

Model was already well-calibrated at export, most likely because `label_smoothing=0.1` during finetuning suppresses overconfident logits the same way temperature scaling would. The small ECE increase reflects LBFGS optimizing NLL, not ECE directly. It is not a real degradation.

### OOD Detection

Evaluated on the identical held-out OOD test set (3,099 ID + 1,000 OOD images).

| Metric | Strategy A (Entropy) | Strategy B (Binary CNN) | Winner |
|---|---|---|---|
| AUROC | 0.9772 | 0.8426 | Strategy A |
| AUPR-IN | 0.9922 | 0.9354 | Strategy A |
| AUPR-OUT | 0.9446 | 0.7102 | Strategy A |
| FPR@95TPR | 0.1200 | 0.6531 | Strategy A |
| Threshold | 3.0241 | N/A | — |

**Strategy A vs B calibration** (ECE of the OOD-detection decision itself, not to be confused with the breed classifier's own confidence calibration reported above, a separate metric):

| | ECE |
|---|---|
| Strategy A | 0.1969 |
| Strategy B | 0.0518 |

**Overall winner: Strategy A (Predictive Entropy), 4/4 metrics.** Strategy B is better calibrated (lower ECE) but discriminates OOD from ID far worse, its FPR@95TPR of 0.65 means it wrongly lets through roughly two-thirds of non-dog images at the same recall point where Strategy A only lets through 12%. Likely explanation: Strategy A leverages the confidence signal of the already well-trained, ImageNet-pretrained ResNet-18 backbone, while Strategy B is a small CNN trained from scratch on a comparatively tiny binary dataset, and has far less capacity and pretraining to draw on.

**Per-category detection rate (held-out OOD test set, 95% TPR threshold, Strategy A):**

| Category | Detection Rate |
|---|---|
| Cats | 67.0% |
| Birds | 86.5% |
| Cars | 99.5% |
| Food | 87.0% |
| Furniture | 99.0% |

**Recommended strategy: Strategy A (entropy thresholding).** Simpler (no extra model to train or maintain), and clearly outperforms Strategy B on every OOD metric. Strategy B's better calibration alone doesn't outweigh its much weaker discrimination for this use case.

---

## Explainability

Grad-CAM (layer4[-1]) applied to test predictions. Heatmaps consistently concentrate on the dog itself (coat, torso, head), not background. For long-coated breeds (e.g. Afghan hound), attention focuses on coat texture rather than facial structure, which is consistent with observed failure cases involving other coat-similar breeds (e.g. Afghan hound → borzoi).

On a 50-image verification batch: 45/50 (90%) correct. This batch is not representative of overall test accuracy, since it covered only 2 breeds, selected alphabetically, not randomly sampled.

---

## Intended Use

Breed identification for dog photos, with a reject option (`is_ood: true`) when the input is not a dog. Not intended for:
- Medical, veterinary, or diagnostic use
- High-stakes decisions without human review
- Breeds outside the 120-class Stanford Dogs taxonomy (will be misclassified as the nearest visual match, not flagged as OOD, since OOD detection targets non-dog images, not unseen breeds)

---

## Limitations

- Cats are a persistent blind spot for Strategy A (67% detection vs. 86%+ for every other category), visually the closest OOD category to a dog.
- Only 5 OOD categories tested; real-world OOD inputs will be far more diverse than cats/birds/cars/food/furniture.
- Training data size is modest relative to the granularity of 120 breed classes; some breed pairs likely remain visually indistinguishable to the model (see confusion matrix, Arushi's section).
- Not evaluated on adversarial inputs.

---

## Ethical Considerations

- No personally identifiable information collected or processed; inputs are photographs of animals/objects only.
- Breed misclassification carries low real-world risk (no medical, legal, or safety consequences), but should not be relied upon for anything beyond casual identification.

---

## Deployment

- FastAPI: `GET /health`, `POST /predict`
- Strategy B takes priority when `models/binary_cnn.pt` is available; falls back to Strategy A, then to no OOD gating, gracefully
- Docker: `python:3.11-slim`, `libgomp1` installed for PyTorch CPU wheel support, port 5001 (macOS) / 8000 (Windows/Linux)
- CI: lint, test, docker build, smoke test on every push (this PR)
- Drift monitoring: EvidentlyAI on inference logs *(in progress, feat/ci)*

---

## References

- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. ICML.
- Hendrycks, D., & Gimpel, K. (2017). A baseline for detecting misclassified and out-of-distribution examples in neural networks. ICLR.