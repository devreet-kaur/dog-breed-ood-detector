"""
src/app.py
FastAPI inference server for the dog breed classifier with OOD detection.

Owner: Devreet (DevOps + Report Lead)

Endpoints:
    GET  /health   -- liveness check, returns model and class info
    POST /predict  -- accepts an image, returns breed + OOD flag + confidence

Usage:
    macOS (port 5001 -- port 5000 blocked by AirPlay Receiver):
        uvicorn src.app:app --host 0.0.0.0 --port 5001 --reload

    Windows / Linux:
        uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload

All model paths and thresholds are read from params.yaml and
models/temperature.json. No hardcoded values.
"""

from __future__ import annotations

import io
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from torch import nn
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config paths ──────────────────────────────────────────────────────────────

PARAMS_PATH            = Path("params.yaml")
BREED_MODEL_PATH       = Path("models/resnet18_best.pt")
BINARY_MODEL_PATH      = Path("models/binary_cnn.pt")
CLASS_NAMES_PATH       = Path("data/processed/class_names.json")
TEMPERATURE_PATH       = Path("models/temperature.json")
ENTROPY_THRESHOLD_PATH = Path("reports/ood/entropy_threshold.json")


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Model loaders ─────────────────────────────────────────────────────────────

def load_breed_model(num_classes: int, device: torch.device) -> nn.Module:
    params = load_params()
    dropout = params["train_head"]["dropout"]
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(model.fc.in_features, num_classes),
    )
    state = torch.load(BREED_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    log.info("Breed classifier loaded from %s", BREED_MODEL_PATH)
    return model


class BinaryCNN(nn.Module):
    def __init__(self, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def load_binary_model(device: torch.device) -> nn.Module | None:
    if not BINARY_MODEL_PATH.exists():
        log.warning("Binary OOD model not found at %s -- Strategy B disabled", BINARY_MODEL_PATH)
        return None
    params = load_params()
    dropout = params["ood_binary"]["dropout"]
    model = BinaryCNN(dropout=dropout)
    state = torch.load(BINARY_MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    log.info("Binary OOD model loaded from %s", BINARY_MODEL_PATH)
    return model


# ── Transform ─────────────────────────────────────────────────────────────────

def get_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ── OOD helpers ───────────────────────────────────────────────────────────────

def compute_entropy(probs: torch.Tensor) -> float:
    log_probs = torch.log(probs + 1e-8)
    return float(-(probs * log_probs).sum().item())


def load_temperature() -> float:
    if TEMPERATURE_PATH.exists():
        with open(TEMPERATURE_PATH) as f:
            return float(json.load(f).get("temperature", 1.0))
    return 1.0


def load_entropy_threshold() -> float | None:
    if ENTROPY_THRESHOLD_PATH.exists():
        with open(ENTROPY_THRESHOLD_PATH) as f:
            data = json.load(f)
            val = data.get("threshold") or data.get("entropy_threshold")
            return float(val) if val is not None else None
    log.warning("Entropy threshold not found at %s -- entropy OOD check disabled", ENTROPY_THRESHOLD_PATH)
    return None


# ── App + state ───────────────────────────────────────────────────────────────

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    startup()
    yield


def startup() -> None:
    params  = load_params()
    device  = get_device()
    log.info("Device: %s", device)

    class_names: list[str] = []
    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)

    num_classes = len(class_names) or params["data"]["num_classes"]
    img_size    = params["data"]["img_size"]

    breed_model      = load_breed_model(num_classes, device) if BREED_MODEL_PATH.exists() else None
    binary_model     = load_binary_model(device)
    temperature      = load_temperature()
    entropy_threshold = load_entropy_threshold()

    _state.update({
        "device":           device,
        "breed_model":      breed_model,
        "binary_model":     binary_model,
        "class_names":      class_names,
        "num_classes":      num_classes,
        "transform":        get_transform(img_size),
        "temperature":      temperature,
        "entropy_threshold": entropy_threshold,
    })
    log.info("Startup complete -- %d classes, T=%.4f, threshold=%s",
             num_classes, temperature, entropy_threshold)


# ── Response schemas ───────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    model: str
    classes: int
    device: str
    temperature: float
    entropy_threshold: float | None
    binary_model_loaded: bool


class PredictResponse(BaseModel):
    breed: str | None
    confidence: float
    is_ood: bool
    ood_method: str
    entropy: float
    top5: list[dict]


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dog Breed + OOD Detector",
    description="ResNet-18 fine-tuned on Stanford Dogs (120 breeds) with OOD detection.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model="resnet18",
        classes=_state.get("num_classes", 0),
        device=str(_state.get("device", "unknown")),
        temperature=_state.get("temperature", 1.0),
        entropy_threshold=_state.get("entropy_threshold"),
        binary_model_loaded=_state.get("binary_model") is not None,
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:  # noqa: B008
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    breed_model = _state.get("breed_model")
    if breed_model is None:
        raise HTTPException(status_code=503, detail="Breed model not loaded yet.")

    try:
        contents = await file.read()
        pil_img  = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read image: {exc}") from exc

    device           = _state["device"]
    transform        = _state["transform"]
    temperature      = _state["temperature"]
    class_names      = _state["class_names"]
    entropy_threshold = _state["entropy_threshold"]
    binary_model     = _state.get("binary_model")

    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = breed_model(tensor)
        probs  = F.softmax(logits / temperature, dim=-1).squeeze(0)

    entropy = compute_entropy(probs)

    # Strategy B (binary CNN) takes priority when loaded
    if binary_model is not None:
        with torch.no_grad():
            bin_logits = binary_model(tensor)
            bin_probs  = F.softmax(bin_logits, dim=-1).squeeze(0)
        is_ood     = bool(bin_probs[0].item() < 0.5)   # index 0 = dog
        ood_method = "binary_cnn"
    elif entropy_threshold is not None:
        is_ood     = entropy > entropy_threshold
        ood_method = "entropy_threshold"
    else:
        is_ood     = False
        ood_method = "none"

    k       = min(5, len(class_names))
    top_v, top_i = torch.topk(probs, k=k)
    top5    = [
        {"breed": class_names[idx.item()] if class_names else str(idx.item()),
         "confidence": round(val.item(), 4)}
        for val, idx in zip(top_v, top_i)
    ]

    return PredictResponse(
        breed=None if is_ood else (class_names[top_i[0].item()] if class_names else None),
        confidence=round(top_v[0].item(), 4),
        is_ood=is_ood,
        ood_method=ood_method,
        entropy=round(entropy, 6),
        top5=top5,
    )