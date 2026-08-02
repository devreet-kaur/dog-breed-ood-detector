"""
tests/test_pipeline.py
Pipeline integration tests for the dog-breed-ood-detector project.

test_prepare  — validates Ryan's data pipeline outputs
test_evaluate — validates Arushi's evaluate.py outputs

Run:
    python -m pytest tests/test_pipeline.py -v
    python -m pytest tests/test_pipeline.py::test_evaluate -v
"""

import os
import json
import pytest


# ── test_prepare ─────────────────────────────────────────────────────────────

def test_prepare():
    """Verify Ryan's prepare.py produced the expected folder structure."""
    base = os.path.join("data", "processed")
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(base, split)
        assert os.path.isdir(split_dir), (
            f"Missing split directory: '{split_dir}'. "
            "Run prepare.py and dvc pull first."
        )
        breeds = os.listdir(split_dir)
        assert len(breeds) == 120, (
            f"Expected 120 breed folders in '{split_dir}', found {len(breeds)}."
        )
        for breed in breeds:
            breed_dir = os.path.join(split_dir, breed)
            images = [
                f for f in os.listdir(breed_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            assert len(images) > 0, (
                f"No images found in '{breed_dir}'."
            )


# ── test_evaluate ─────────────────────────────────────────────────────────────

def test_evaluate_metrics_file_exists():
    """reports/metrics_test.json must exist after evaluate.py runs."""
    path = os.path.join("reports", "metrics_test.json")
    assert os.path.isfile(path), (
        f"'{path}' not found. Run python src/evaluate.py first."
    )


def test_evaluate_metrics_keys():
    """metrics_test.json must contain top1_acc, top5_acc, and test_loss."""
    path = os.path.join("reports", "metrics_test.json")
    if not os.path.isfile(path):
        pytest.skip("metrics_test.json not yet generated — run evaluate.py first.")
    with open(path) as f:
        metrics = json.load(f)
    for key in ["top1_acc", "top5_acc", "test_loss"]:
        assert key in metrics, (
            f"Key '{key}' missing from metrics_test.json. Got: {list(metrics.keys())}"
        )


def test_evaluate_metrics_values():
    """Sanity check: top1_acc and top5_acc must be between 0 and 1."""
    path = os.path.join("reports", "metrics_test.json")
    if not os.path.isfile(path):
        pytest.skip("metrics_test.json not yet generated — run evaluate.py first.")
    with open(path) as f:
        metrics = json.load(f)
    assert 0.0 <= metrics["top1_acc"] <= 1.0, (
        f"top1_acc out of range: {metrics['top1_acc']}"
    )
    assert 0.0 <= metrics["top5_acc"] <= 1.0, (
        f"top5_acc out of range: {metrics['top5_acc']}"
    )
    assert metrics["top5_acc"] >= metrics["top1_acc"], (
        "top5_acc should be >= top1_acc."
    )


def test_evaluate_confusion_matrix_exists():
    """Confusion matrix plot must exist after evaluate.py runs."""
    path = os.path.join("reports", "plots", "confusion_matrix.png")
    assert os.path.isfile(path), (
        f"'{path}' not found. Run python src/evaluate.py first."
    )


def test_model_checkpoint_exists():
    """models/resnet18_best.pt must exist after training."""
    path = os.path.join("models", "resnet18_best.pt")
    assert os.path.isfile(path), (
        f"'{path}' not found. Run train.py --stage head and --stage finetune first."
    )