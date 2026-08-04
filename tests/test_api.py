"""
tests/test_api.py
Pytest tests for src/app.py FastAPI endpoints.

Owner: Devreet

Tests run without a real model -- they mock the model state so no GPU
or trained weights are needed. All 9 tests should pass in under 5 seconds.

Run:
    python -m pytest tests/test_api.py -v
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.app import _state, app

# ── Fixtures ──────────────────────────────────────────────────────────────────

CLASS_NAMES = [f"breed_{i:03d}" for i in range(120)]
IMG_SIZE    = 224


def make_fake_image(width: int = 224, height: int = 224) -> bytes:
    img = Image.new("RGB", (width, height), color=(120, 80, 60))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_dog_probs(hot_index: int = 0, num_classes: int = 120) -> torch.Tensor:
    probs = torch.zeros(num_classes)
    probs[hot_index] = 0.92
    probs[(hot_index + 1) % num_classes] = 0.05
    probs[(hot_index + 2) % num_classes] = 0.03
    return probs


def make_ood_probs(num_classes: int = 120) -> torch.Tensor:
    return torch.ones(num_classes) / num_classes  # max entropy


@pytest.fixture(autouse=True)
def mock_app_state(tmp_path):
    """Inject a fake model state so tests never need real .pt files."""
    fake_breed_model = MagicMock()
    fake_breed_model.return_value = torch.zeros(1, 120)

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _state.update({
        "device":            torch.device("cpu"),
        "breed_model":       fake_breed_model,
        "binary_model":      None,
        "class_names":       CLASS_NAMES,
        "num_classes":       len(CLASS_NAMES),
        "transform":         transform,
        "temperature":       1.0,
        "entropy_threshold": 2.5,
    })
    yield
    _state.clear()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


# ── Test 1: /health returns 200 ───────────────────────────────────────────────

def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200


# ── Test 2: /health returns correct schema ────────────────────────────────────

def test_health_schema(client: TestClient) -> None:
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["model"] == "resnet18"
    assert data["classes"] == 120
    assert "device" in data
    assert "temperature" in data
    assert "entropy_threshold" in data
    assert "binary_model_loaded" in data


# ── Test 3: /health shows binary model not loaded ─────────────────────────────

def test_health_binary_model_not_loaded(client: TestClient) -> None:
    _state["binary_model"] = None
    data = client.get("/health").json()
    assert data["binary_model_loaded"] is False


# ── Test 4: /predict returns 200 for a valid dog image ────────────────────────

def test_predict_dog_returns_200(client: TestClient) -> None:
    dog_logits = torch.log(make_dog_probs()).unsqueeze(0)
    _state["breed_model"].return_value = dog_logits
    response = client.post(
        "/predict",
        files={"file": ("dog.jpg", make_fake_image(), "image/jpeg")},
    )
    assert response.status_code == 200


# ── Test 5: /predict returns is_ood false for in-distribution image ───────────

def test_predict_in_distribution_not_ood(client: TestClient) -> None:
    dog_logits = torch.log(make_dog_probs(hot_index=5)).unsqueeze(0)
    _state["breed_model"].return_value = dog_logits
    _state["entropy_threshold"] = 2.5  # high threshold -- low entropy dog passes
    data = client.post(
        "/predict",
        files={"file": ("dog.jpg", make_fake_image(), "image/jpeg")},
    ).json()
    assert data["is_ood"] is False
    assert data["breed"] is not None
    assert data["confidence"] > 0.0


# ── Test 6: /predict returns is_ood true for OOD image ───────────────────────

def test_predict_ood_image_flagged(client: TestClient) -> None:
    ood_logits = torch.log(make_ood_probs() + 1e-8).unsqueeze(0)
    _state["breed_model"].return_value = ood_logits
    _state["entropy_threshold"] = 0.1  # very low threshold -- high entropy fails
    data = client.post(
        "/predict",
        files={"file": ("cat.jpg", make_fake_image(), "image/jpeg")},
    ).json()
    assert data["is_ood"] is True
    assert data["breed"] is None


# ── Test 7: /predict returns top5 list ────────────────────────────────────────

def test_predict_returns_top5(client: TestClient) -> None:
    dog_logits = torch.log(make_dog_probs()).unsqueeze(0)
    _state["breed_model"].return_value = dog_logits
    data = client.post(
        "/predict",
        files={"file": ("dog.jpg", make_fake_image(), "image/jpeg")},
    ).json()
    assert "top5" in data
    assert len(data["top5"]) == 5
    for item in data["top5"]:
        assert "breed" in item
        assert "confidence" in item


# ── Test 8: /predict rejects non-image file ───────────────────────────────────

def test_predict_rejects_non_image(client: TestClient) -> None:
    response = client.post(
        "/predict",
        files={"file": ("file.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 400


# ── Test 9: /predict returns 503 when model not loaded ───────────────────────

def test_predict_503_when_model_not_loaded(client: TestClient) -> None:
    _state["breed_model"] = None
    response = client.post(
        "/predict",
        files={"file": ("dog.jpg", make_fake_image(), "image/jpeg")},
    )
    assert response.status_code == 503
