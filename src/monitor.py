"""
src/monitor.py
EvidentlyAI data drift detection for the dog breed classifier.

Owner: Devreet

Compares inference logs against the training data distribution to detect
data drift. Generates an HTML report saved to reports/drift/.

Usage:
    python src/monitor.py --reference data/processed/train \
                          --current   inference_logs/current_batch \
                          --out       reports/drift/report.html

    # Or with a JSON log file from the API:
    python src/monitor.py --log inference_logs/predictions.jsonl

Outputs:
    reports/drift/report_<timestamp>.html  -- EvidentlyAI HTML drift report
    reports/drift/summary.json             -- machine-readable drift summary
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
DRIFT_DIR   = Path("reports/drift")


def load_params() -> dict:
    with open(PARAMS_PATH) as f:
        return yaml.safe_load(f)


def load_image_stats_from_folder(folder: Path, max_images: int = 1000) -> pd.DataFrame:
    """
    Load basic image statistics from a folder of images.
    Returns a DataFrame with columns: brightness, contrast, width, height.
    Used as features for drift detection when raw pixel data is too large.
    """
    from PIL import Image

    records = []
    image_paths = list(folder.rglob("*.jpg")) + list(folder.rglob("*.jpeg")) + list(folder.rglob("*.png"))
    image_paths = image_paths[:max_images]

    log.info("Loading stats from %d images in %s", len(image_paths), folder)

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            arr = np.array(img, dtype=np.float32)
            records.append({
                "brightness": float(arr.mean()),
                "contrast":   float(arr.std()),
                "width":      img.width,
                "height":     img.height,
                "aspect_ratio": img.width / img.height,
            })
        except Exception as exc:  # noqa: BLE001
            log.warning("Skipping %s: %s", path, exc)

    return pd.DataFrame(records)


def load_from_jsonl(log_path: Path) -> pd.DataFrame:
    """
    Load inference log from a JSONL file produced by the API.
    Each line is a JSON object with keys: confidence, entropy, is_ood, breed.
    """
    records = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line: %s", exc)

    df = pd.DataFrame(records)
    log.info("Loaded %d inference records from %s", len(df), log_path)
    return df


def run_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    out_path: Path,
) -> dict:
    """Run EvidentlyAI drift report and save HTML + summary JSON."""
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_path))
    log.info("Drift report saved to %s", out_path)

    result  = report.as_dict()
    metrics = result.get("metrics", [])

    drifted_cols = 0
    total_cols   = 0
    for m in metrics:
        result_data = m.get("result", {})
        if "number_of_drifted_columns" in result_data:
            drifted_cols = result_data["number_of_drifted_columns"]
            total_cols   = result_data.get("number_of_columns", 0)

    summary = {
        "timestamp":          datetime.now(tz=datetime.timezone.utc).isoformat(),
        "reference_size":     len(reference),
        "current_size":       len(current),
        "drifted_columns":    drifted_cols,
        "total_columns":      total_cols,
        "drift_detected":     drifted_cols > 0,
        "report_path":        str(out_path),
    }

    summary_path = out_path.parent / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved to %s", summary_path)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="EvidentlyAI drift detection")
    parser.add_argument("--reference", type=str, default="data/processed/train",
                        help="Reference data folder (training set)")
    parser.add_argument("--current",   type=str, default=None,
                        help="Current data folder (recent inference images)")
    parser.add_argument("--log",       type=str, default=None,
                        help="JSONL inference log file from the API")
    parser.add_argument("--out",       type=str, default=None,
                        help="Output HTML report path")
    parser.add_argument("--max",       type=int, default=500,
                        help="Max images to sample from each folder")
    args = parser.parse_args()

    timestamp = datetime.now(tz=datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path  = Path(args.out) if args.out else DRIFT_DIR / f"report_{timestamp}.html"

    if args.log:
        log.info("Loading current data from inference log: %s", args.log)
        current_df = load_from_jsonl(Path(args.log))
        log.info("Loading reference data from: %s", args.reference)
        ref_df = load_from_jsonl(Path(args.reference)) if Path(args.reference).suffix == ".jsonl" \
            else load_image_stats_from_folder(Path(args.reference), max_images=args.max)
        # Align columns
        common_cols = list(set(ref_df.columns) & set(current_df.columns))
        ref_df     = ref_df[common_cols]
        current_df = current_df[common_cols]
    elif args.current:
        log.info("Loading reference from: %s", args.reference)
        ref_df = load_image_stats_from_folder(Path(args.reference), max_images=args.max)
        log.info("Loading current from: %s", args.current)
        current_df = load_image_stats_from_folder(Path(args.current), max_images=args.max)
    else:
        parser.error("Provide either --current or --log")

    if ref_df.empty or current_df.empty:
        log.error("One of the datasets is empty. Cannot run drift report.")
        return

    summary = run_drift_report(ref_df, current_df, out_path)

    print("\n── Drift Report Summary ──────────────────────")
    print(f"  Timestamp:       {summary['timestamp']}")
    print(f"  Reference size:  {summary['reference_size']}")
    print(f"  Current size:    {summary['current_size']}")
    print(f"  Drifted columns: {summary['drifted_columns']} / {summary['total_columns']}")
    print(f"  Drift detected:  {summary['drift_detected']}")
    print(f"  Report:          {summary['report_path']}")


if __name__ == "__main__":
    main()
