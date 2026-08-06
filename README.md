# Dog Breed Classifier with Out-of-Distribution Detection

Fine-grained dog breed classifier (120 breeds, Stanford Dogs) with an out-of-distribution (OOD) detection layer that rejects non-dog inputs. Compares two OOD strategies: softmax entropy thresholding vs. a supervised binary CNN gate. Built with PyTorch, DVC, MLflow, and FastAPI.

**MAI202 Deep Learning — Seneca Polytechnic, Summer 2026**
Team: Ryan Caezar Soria, Arushi Anand, Soodeh Vanaki, Devreet Kaur

---

## Results at a glance

| | |
|---|---|
| Breed classification | 76.22% top-1, 96.87% top-5 (120 classes) |
| OOD detection winner | Strategy A (entropy thresholding), AUROC 0.9772 vs Strategy B's 0.8426 |

Full results and analysis: [`model_card.md`](model_card.md).

---

## Setup

Requires Python 3.11, conda, and a DVC-compatible Google Drive account for pulling data/models.

```bash
conda create -n mlcourse python=3.11
conda activate mlcourse
pip install -r requirements.txt

# One-time DVC remote auth (ask a teammate for the client ID/secret)
dvc remote modify --local gdrive_remote gdrive_client_id "<client_id>"
dvc remote modify --local gdrive_remote gdrive_client_secret "<client_secret>"
dvc pull
```

---

## Running the pipeline

The full pipeline (data prep → train → evaluate → OOD detection → comparison) is DVC-managed:

```bash
dvc repro
```

Or run individual stages:
```bash
dvc repro prepare      # data split + bbox crops
dvc repro train        # ResNet-18 two-stage fine-tuning
dvc repro evaluate      # test set metrics
dvc repro calibrate     # temperature scaling
dvc repro ood_entropy   # Strategy A
dvc repro ood_binary    # Strategy B
dvc repro ood_binary_evaluate  # Strategy B held-out test evaluation
dvc repro ood_compare   # A vs B comparison
```

Track experiments with MLflow:
```bash
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```

---

## Running the API

**Locally:**
```bash
# macOS (port 5000 is reserved by AirPlay Receiver)
uvicorn src.app:app --host 0.0.0.0 --port 5001 --reload

# Windows / Linux
uvicorn src.app:app --host 0.0.0.0 --port 8000 --reload
```

Then visit `http://localhost:5001/docs` (or `:8000`) for interactive Swagger docs, or:
```bash
curl http://localhost:5001/health
curl -X POST http://localhost:5001/predict -F "file=@path/to/image.jpg"
```

**With Docker:**
```bash
docker build -t mai202-dog-ood .
docker compose up
```
By default the container is served on host port **8000**. macOS users should copy `.env.example` to `.env` and set `HOST_PORT=5001` before running `docker compose up`, since port 5000 is reserved by AirPlay Receiver on modern macOS:
```bash
cp .env.example .env
docker compose up
curl http://localhost:5001/health
```

---

## Testing

```bash
python -m pytest tests/ -v
ruff check src/ tests/ --no-cache
```

---

## Project structure

```
src/
  prepare.py              # data pipeline, bbox crops, stratified split
  train.py                # ResNet-18 two-stage fine-tuning
  evaluate.py              # test set metrics
  calibrate.py             # temperature scaling
  ood_entropy.py           # Strategy A
  ood_binary.py            # Strategy B
  ood_compare.py           # A vs B comparison + reliability curves
  visualize_gradcam.py     # Grad-CAM heatmaps
  ood_failure_analysis.py  # per-category OOD failure analysis
  app.py                   # FastAPI service
  monitor.py               # EvidentlyAI drift monitoring
  download_ood.py          # OOD dataset construction

tests/          # pytest suite (160+ tests across all modules)
reports/        # metrics, plots, OOD analysis outputs (DVC-tracked)
docs/figures/   # key figures for the report/presentation
model_card.md   # full results and model documentation
params.yaml     # all hyperparameters (single source of truth)
dvc.yaml        # pipeline stage definitions
```

---

## Team & roles

| Role | Owner | Focus |
|---|---|---|
| Data Engineer / Pipeline Lead | Ryan Caezar Soria | Data pipeline, DVC, fresh-clone testing |
| ML Lead | Arushi Anand | Model architecture, training |
| OOD Detection | Soodeh Vanaki | Strategy A/B, comparison pipeline |
| DevOps + Report Lead | Devreet Kaur | Calibration, Grad-CAM, API, Docker, CI |

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for branch/PR conventions.