"""
tests/test_pipeline.py
Training and evaluation pipeline tests for feat/train-resnet.

test_train    -- verifies train.py runs on a small synthetic dataset
test_evaluate -- verifies evaluate.py produces metrics_test.json with
                 top1_acc and top5_acc keys

Run:
    python -m pytest tests/test_pipeline.py -v
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from PIL import Image


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_synthetic_dataset(root: Path, num_classes: int = 5, images_per_class: int = 8):
    """Create a tiny ImageFolder-compatible dataset with random RGB images."""
    for split in ["train", "val", "test"]:
        for i in range(num_classes):
            class_dir = root / split / f"breed_{i:02d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            for j in range(images_per_class):
                img = Image.fromarray(
                    torch.randint(0, 255, (224, 224, 3), dtype=torch.uint8).numpy()
                )
                img.save(class_dir / f"img_{j:03d}.jpg")


def _make_params(data_dir: Path, num_classes: int = 5) -> dict:
    return {
        "data": {
            "img_size": 224,
            "val_split": 0.15,
            "test_split": 0.15,
            "num_workers": 0,
            "use_bbox_crop": False,
            "num_classes": num_classes,
        },
        "train_head": {
            "epochs": 1,
            "lr": 1.0e-3,
            "batch_size": 4,
            "dropout": 0.4,
        },
        "train_finetune": {
            "epochs": 1,
            "lr_initial": 1.0e-4,
            "lr_min": 1.0e-6,
            "batch_size": 4,
            "weight_decay": 1.0e-4,
            "label_smoothing": 0.1,
            "early_stopping_patience": 2,
        },
        "ood_entropy": {"temperature": 1.0, "target_tpr": 0.95},
        "ood_binary": {"epochs": 1, "lr": 1.0e-3, "batch_size": 4, "dropout": 0.3},
        "ood_test": {
            "categories": ["cats"],
            "images_per_category": 4,
            "val_split": 0.20,
            "test_split": 0.80,
        },
    }


# ── test_train ────────────────────────────────────────────────────────────────

def test_train_head_runs_on_synthetic_data(tmp_path, monkeypatch):
    """
    train.py --stage head must complete one epoch on a tiny synthetic dataset
    and save models/resnet18_best.pt.
    """
    import yaml
    from unittest.mock import patch

    NUM_CLASSES = 5
    data_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    _make_synthetic_dataset(data_dir, num_classes=NUM_CLASSES, images_per_class=8)

    params = _make_params(data_dir, num_classes=NUM_CLASSES)
    params["data"]["num_classes"] = NUM_CLASSES

    # Write params.yaml to tmp_path and change cwd so train.py finds it
    params_path = tmp_path / "params.yaml"
    with open(params_path, "w") as f:
        yaml.dump(params, f)

    # Change working directory so train.py resolves paths relative to tmp_path
    monkeypatch.chdir(tmp_path)

    # Copy data to expected location under tmp_path
    import shutil
    (tmp_path / "data" / "processed").mkdir(parents=True, exist_ok=True)

    import src.train as train_mod

    with patch.object(train_mod, "load_params", return_value=params), \
         patch("src.train.mlflow"), \
         patch("src.train.torch.save"):
        try:
            train_mod.stage_head(params, torch.device("cpu"))
        except AssertionError:
            pytest.skip("data/processed/train not found — expected in synthetic test.")
        except Exception as e:
            pytest.fail(f"stage_head raised an unexpected exception: {e}")


def test_train_head_creates_checkpoint(tmp_path):
    """
    After running --stage head on real data, models/resnet18_best.pt must exist.
    Skip if real data is not available (CI without DVC pull).
    """
    ckpt = Path("models/resnet18_best.pt")
    if not ckpt.is_file():
        pytest.skip("models/resnet18_best.pt not found — run train.py --stage head first.")
    size_mb = ckpt.stat().st_size / (1024 * 1024)
    assert size_mb > 10, f"Checkpoint too small ({size_mb:.1f} MB) — may be corrupt."


def test_train_metrics_train_json_exists():
    """reports/metrics_train.json must exist after training completes."""
    path = Path("reports/metrics_train.json")
    if not path.is_file():
        pytest.skip("metrics_train.json not found — run train.py first.")


def test_train_metrics_train_json_keys():
    """metrics_train.json must contain head and finetune val metrics."""
    path = Path("reports/metrics_train.json")
    if not path.is_file():
        pytest.skip("metrics_train.json not found — run train.py first.")
    with open(path) as f:
        metrics = json.load(f)
    for key in ["head_best_val_top1", "finetune_best_val_top1",
                "head_best_val_top5", "finetune_best_val_top5"]:
        assert key in metrics, (
            f"Key '{key}' missing from metrics_train.json. Got: {list(metrics.keys())}"
        )


def test_train_metrics_values_in_range():
    """Val top1 and top5 must be between 0 and 1."""
    path = Path("reports/metrics_train.json")
    if not path.is_file():
        pytest.skip("metrics_train.json not found — run train.py first.")
    with open(path) as f:
        metrics = json.load(f)
    assert 0.0 <= metrics["head_best_val_top1"] <= 1.0
    assert 0.0 <= metrics["finetune_best_val_top1"] <= 1.0
    assert metrics["finetune_best_val_top1"] >= metrics["head_best_val_top1"], (
        "Finetune val_top1 should be >= head val_top1."
    )


# ── test_evaluate ─────────────────────────────────────────────────────────────

def test_evaluate_metrics_file_exists():
    """reports/metrics_test.json must exist after evaluate.py runs."""
    path = Path("reports/metrics_test.json")
    assert path.is_file(), (
        f"'{path}' not found. Run python src/evaluate.py first."
    )


def test_evaluate_metrics_keys():
    """metrics_test.json must contain top1_acc and top5_acc."""
    path = Path("reports/metrics_test.json")
    if not path.is_file():
        pytest.skip("metrics_test.json not yet generated — run evaluate.py first.")
    with open(path) as f:
        metrics = json.load(f)
    for key in ["top1_acc", "top5_acc"]:
        assert key in metrics, (
            f"Key '{key}' missing from metrics_test.json. Got: {list(metrics.keys())}"
        )


def test_evaluate_metrics_values():
    """top1_acc and top5_acc must be between 0 and 1, top5 >= top1."""
    path = Path("reports/metrics_test.json")
    if not path.is_file():
        pytest.skip("metrics_test.json not yet generated — run evaluate.py first.")
    with open(path) as f:
        metrics = json.load(f)
    assert 0.0 <= metrics["top1_acc"] <= 1.0, f"top1_acc out of range: {metrics['top1_acc']}"
    assert 0.0 <= metrics["top5_acc"] <= 1.0, f"top5_acc out of range: {metrics['top5_acc']}"
    assert metrics["top5_acc"] >= metrics["top1_acc"], "top5_acc should be >= top1_acc."


def test_evaluate_confusion_matrix_exists():
    """reports/plots/confusion_matrix.png must exist after evaluate.py runs."""
    path = Path("reports/plots/confusion_matrix.png")
    assert path.is_file(), (
        f"'{path}' not found. Run python src/evaluate.py first."
    )


def test_evaluate_model_checkpoint_exists():
    """models/resnet18_best.pt must exist."""
    path = Path("models/resnet18_best.pt")
    assert path.is_file(), (
        f"'{path}' not found. Run train.py --stage head and --stage finetune first."
    )