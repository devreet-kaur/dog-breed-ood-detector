"""
src/download_ood.py
Build the out-of-distribution (OOD) test set: images that are NOT dogs.

Owner: Ryan (Data Engineer + Pipeline Lead)
Not a DVC pipeline stage -- the output is tracked with `dvc add data/raw/ood`.

Categories, counts and splits all come from params.yaml -> ood_test:
    categories: [cats, birds, cars, food, furniture]
    images_per_category: 250
    val_split: 0.2   -> data/raw/ood/val/   (threshold tuning + binary CNN)
    test_split: 0.8  -> data/raw/ood/test/  (held out until final evaluation)

Output layout (ImageFolder-compatible, one folder per category):
    data/raw/ood/val/<category>/*.jpg
    data/raw/ood/test/<category>/*.jpg

Two ways to source the raw images:

  1. STAGING (recommended, always works):
     Drop your collected images (from Kaggle / ImageNet subsets / anywhere)
     into data/raw/ood_staging/<category>/ then run:
         python src/download_ood.py --from-staging
     This resizes, trims to images_per_category, and does the 80/20 split.

  2. HUGGING FACE (automatic, best-effort):
         python src/download_ood.py
     Pulls each category from a public HF dataset (see SOURCES below). If HF is
     unreachable or a schema changed, it tells you to use --from-staging.
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

PARAMS_PATH = Path("params.yaml")
STAGING_DIR = Path("data/raw/ood_staging")
OOD_DIR = Path("data/raw/ood")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
SEED = 42
IMG_SIZE = 224  # OOD images are saved square to match the classifier input

# Category -> (HF dataset, config, split, image column, label column, label value)
# Used only in the automatic HF path. Edit freely if a dataset moves.
SOURCES = {
    "cats": dict(dataset="microsoft/cats_vs_dogs", split="train",
                 image_col="image", label_col="labels", keep=0),
    "birds": dict(dataset="cassiekang/cub200_dataset", split="train",
                  image_col="image", label_col=None, keep=None),
    "cars": dict(dataset="tanganke/stanford_cars", split="train",
                 image_col="image", label_col=None, keep=None),
    "food": dict(dataset="ethz/food101", split="train",
                 image_col="image", label_col=None, keep=None),
    "furniture": dict(dataset="Arkan0ID/furniture-dataset", split="train",
                      image_col="image", label_col=None, keep=None),
}


def load_params(path: Path = PARAMS_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_square(img: Image.Image, dest: Path, size: int = IMG_SIZE) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").resize((size, size), Image.BILINEAR).save(
        dest, "JPEG", quality=95
    )


# ── Split & write ─────────────────────────────────────────────────────────────
def split_and_write(
    images: list[Image.Image] | list[Path],
    category: str,
    val_split: float,
    seed: int = SEED,
) -> tuple[int, int]:
    """Shuffle a category's images and write val_split to val/, the rest to test/."""
    idx = list(range(len(images)))
    random.Random(seed).shuffle(idx)
    n_val = int(round(len(idx) * val_split))
    val_idx, test_idx = set(idx[:n_val]), set(idx[n_val:])

    for split_name, keep_idx in [("val", val_idx), ("test", test_idx)]:
        for rank, i in enumerate(sorted(keep_idx)):
            item = images[i]
            img = item if isinstance(item, Image.Image) else Image.open(item)
            save_square(img, OOD_DIR / split_name /
                        category / f"{category}_{rank:04d}.jpg")
    return len(val_idx), len(test_idx)


# ── Staging path (manual images already on disk) ──────────────────────────────
def build_from_staging(categories, images_per_category, val_split) -> None:
    if not STAGING_DIR.exists():
        raise SystemExit(
            f"[ood] {STAGING_DIR} not found. Create it with one subfolder per "
            f"category ({', '.join(categories)}) and drop your images inside."
        )
    for category in categories:
        src_dir = STAGING_DIR / category
        if not src_dir.exists():
            raise SystemExit(f"[ood] Missing staging folder: {src_dir}")
        paths = sorted(p for p in src_dir.iterdir()
                       if p.suffix.lower() in IMAGE_EXTS)
        if len(paths) < images_per_category:
            print(
                f"[ood] WARNING: {category} has {len(paths)} images, "
                f"fewer than the {images_per_category} requested."
            )
        paths = paths[:images_per_category]
        n_val, n_test = split_and_write(paths, category, val_split)
        print(f"[ood] {category:10s} -> val {n_val:3d} | test {n_test:3d}")


# ── Hugging Face path (automatic best-effort) ─────────────────────────────────
def build_from_huggingface(categories, images_per_category, val_split) -> None:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "[ood] `datasets` not installed. Use --from-staging instead.")

    for category in categories:
        cfg = SOURCES.get(category)
        if cfg is None:
            raise SystemExit(
                f"[ood] No HF source configured for '{category}'.")
        print(f"[ood] Streaming {category} from {cfg['dataset']} ...")
        try:
            ds = load_dataset(
                cfg["dataset"], split=cfg["split"], streaming=True)
            collected: list[Image.Image] = []
            for row in ds:
                if cfg["label_col"] and cfg["keep"] is not None:
                    if row[cfg["label_col"]] != cfg["keep"]:
                        continue
                img = row[cfg["image_col"]]
                if not isinstance(img, Image.Image):
                    continue
                collected.append(img.convert("RGB"))
                if len(collected) >= images_per_category:
                    break
        except Exception as e:  # network / schema / auth issues -> actionable message
            raise SystemExit(
                f"[ood] Could not pull '{category}' from HF ({e}). "
                f"Fall back to: python src/download_ood.py --from-staging"
            )
        if len(collected) < images_per_category:
            print(
                f"[ood] WARNING: only got {len(collected)} {category} images.")
        n_val, n_test = split_and_write(collected, category, val_split)
        print(f"[ood] {category:10s} -> val {n_val:3d} | test {n_test:3d}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the OOD test set.")
    parser.add_argument(
        "--from-staging",
        action="store_true",
        help="Use images already in data/raw/ood_staging/<category>/.",
    )
    args = parser.parse_args()

    ood = load_params()["ood_test"]
    categories = ood["categories"]
    images_per_category = ood["images_per_category"]
    val_split = ood["val_split"]

    # Fresh start so re-runs don't accumulate stale images.
    if OOD_DIR.exists():
        shutil.rmtree(OOD_DIR)

    if args.from_staging:
        build_from_staging(categories, images_per_category, val_split)
    else:
        build_from_huggingface(categories, images_per_category, val_split)

    print(
        f"\n[ood] Done. {len(categories)} categories, "
        f"{val_split:.0%} val / {1 - val_split:.0%} test -> {OOD_DIR}"
    )


if __name__ == "__main__":
    main()
