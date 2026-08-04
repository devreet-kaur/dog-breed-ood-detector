"""
src/prepare.py
Turn raw Stanford Dogs into the model-ready, bounding-box-cropped dataset.

Owner: Ryan (Data Engineer + Pipeline Lead)
This IS a DVC pipeline stage -- see the `prepare` stage in dvc.yaml.

All values read from params.yaml -> data:. Never hardcode numbers here.
    raw_dir        where the extracted tarballs live
    out_dir        parent of train/ val/ test/  (the DVC outputs)
    img_size       square output resolution
    use_bbox_crop  crop to the annotation box before resizing
    bbox_margin    fraction of box size added on each side (0.0 = tight)
    val_split      \\ train_split is the remainder, so the three
    test_split     /  ratios can never drift out of sync
    seed           change this and DVC re-runs the stage

Two commands, deliberately separate:

  1. ONE-TIME DOWNLOAD -- raw data is a pipeline *input*, not a stage output:
         python src/prepare.py --download
         dvc add data/raw/stanford_dogs
         dvc push
         git add data/raw/stanford_dogs.dvc data/raw/.gitignore
     Fetches images.tar (~757 MB) + annotation.tar (~21 MB) and extracts them.
     Existing files are skipped, so re-running is safe.

  2. THE PIPELINE STAGE -- what `dvc repro prepare` calls:
         python src/prepare.py
     Reads raw_dir, crops, resizes, splits, writes out_dir. A pure function of
     (raw data + params), so it reproduces identically on any machine.

Raw layout after --download:
    data/raw/stanford_dogs/Images/n02085620-Chihuahua/n02085620_7.jpg
    data/raw/stanford_dogs/Annotation/n02085620-Chihuahua/n02085620_7   <- XML, no extension

Output layout (ImageFolder-compatible, mirrors data/raw/ood/):
    data/processed/train/Chihuahua/n02085620_7.jpg
    data/processed/val/Chihuahua/...
    data/processed/test/Chihuahua/...
    reports/split_summary.json

The split is stratified per breed: every one of the 120 classes appears in all
three splits at the same ratio. A flat random split over 20,580 images would
leave the rarer breeds badly represented in val and test.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import tarfile
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

PARAMS_PATH = Path("params.yaml")
SUMMARY_PATH = Path("reports/split_summary.json")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val", "test")

# Directory names created by the two tarballs. Fixed by the dataset, not tunable.
IMAGES_SUBDIR = "Images"
ANNOTATIONS_SUBDIR = "Annotation"

EXPECTED_BREEDS = 120
EXPECTED_IMAGES = 20580


def load_params(path: Path = PARAMS_PATH) -> dict:
    if not path.exists():
        raise SystemExit(
            f"[prepare] {path} not found -- run this from the repository root."
        )
    with open(path) as f:
        return yaml.safe_load(f)["data"]


def split_ratios(params: dict) -> dict[str, float]:
    """Derive the three ratios from val_split/test_split in params.yaml."""
    val = float(params["val_split"])
    test = float(params["test_split"])
    train = 1.0 - val - test
    if train <= 0:
        raise SystemExit(
            f"[prepare] val_split ({val}) + test_split ({test}) must be < 1.0."
        )
    return {"train": train, "val": val, "test": test}


# ── Download ──────────────────────────────────────────────────────────────────
def _download(url: str, dest: Path) -> None:
    """Stream a URL to disk with a progress bar. Skips if dest already exists."""
    if dest.exists():
        print(f"[prepare] {dest.name} already downloaded, skipping.")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"[prepare] Downloading {url}")
    try:
        # URL is not user input -- it comes from data.sources in params.yaml.
        with urllib.request.urlopen(url) as response:
            total = int(response.headers.get("Content-Length", 0))
            with open(tmp, "wb") as f, tqdm(
                total=total or None, unit="B", unit_scale=True, unit_divisor=1024
            ) as bar:
                while chunk := response.read(1 << 20):
                    f.write(chunk)
                    bar.update(len(chunk))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"[prepare] Download failed for {url} ({e}).\n"
            f"[prepare] Stanford's server is occasionally down. Retry, or take the\n"
            f"[prepare] same two tarballs from the Kaggle mirror\n"
            f"[prepare]   https://www.kaggle.com/datasets/jessicali9530/stanford-dogs-dataset\n"
            f"[prepare] and drop them in {dest.parent}/ before re-running."
        )
    tmp.rename(dest)


def _extract(tar_path: Path, raw_dir: Path, expect_subdir: str) -> None:
    """Extract a tarball into raw_dir unless its top-level folder is already there."""
    target = raw_dir / expect_subdir
    if target.exists() and any(target.iterdir()):
        print(f"[prepare] {expect_subdir}/ already extracted, skipping.")
        return

    print(f"[prepare] Extracting {tar_path.name} -> {raw_dir}/")
    with tarfile.open(tar_path) as tar:
        members = [m for m in tar.getmembers() if not m.name.startswith(("/", ".."))]
        try:
            # Python 3.12+ warns without an explicit filter; 'data' is the safe one.
            tar.extractall(raw_dir, members=members, filter="data")
        except TypeError:  # Python < 3.12 has no `filter` argument
            # Safe: absolute and parent-traversal members were filtered out above.
            tar.extractall(raw_dir, members=members)

    if not target.exists():
        raise SystemExit(
            f"[prepare] Expected {target} after extracting {tar_path.name}, but it "
            f"is missing. The tarball layout may have changed."
        )


def download_raw(raw_dir: Path, sources: dict) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_tar = raw_dir / "images.tar"
    annotations_tar = raw_dir / "annotation.tar"

    _download(sources["images"], images_tar)
    _download(sources["annotations"], annotations_tar)
    _extract(images_tar, raw_dir, IMAGES_SUBDIR)
    _extract(annotations_tar, raw_dir, ANNOTATIONS_SUBDIR)

    n_breeds = len(list((raw_dir / IMAGES_SUBDIR).iterdir()))
    n_images = sum(1 for _ in (raw_dir / IMAGES_SUBDIR).rglob("*.jpg"))
    print(f"\n[prepare] Raw data ready: {n_breeds} breeds, {n_images} images.")
    if n_breeds != EXPECTED_BREEDS or n_images != EXPECTED_IMAGES:
        print(
            f"[prepare] WARNING: expected {EXPECTED_BREEDS} breeds / "
            f"{EXPECTED_IMAGES} images -- the download may be incomplete."
        )
    print(
        "[prepare] Next:\n"
        "[prepare]   dvc add data/raw/stanford_dogs\n"
        "[prepare]   dvc push\n"
        "[prepare]   git add data/raw/stanford_dogs.dvc data/raw/.gitignore"
    )


# ── Bounding boxes ────────────────────────────────────────────────────────────
def parse_boxes(annotation_path: Path) -> list[tuple[int, int, int, int]]:
    """Return every <bndbox> in a Stanford Dogs annotation file as (x1, y1, x2, y2)."""
    root = ET.parse(annotation_path).getroot()
    boxes = []
    for obj in root.findall("object"):
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            coords = tuple(
                int(float(box.findtext(k))) for k in ("xmin", "ymin", "xmax", "ymax")
            )
        except (TypeError, ValueError):
            continue  # one malformed entry should not kill a 20k-image run
        boxes.append(coords)
    return boxes


def largest_box(
    boxes: list[tuple[int, int, int, int]],
) -> tuple[int, int, int, int] | None:
    """Pick the box with the greatest area. Ties break on first occurrence.

    Stanford Dogs has multiple boxes whenever an image contains several dogs.
    Taking the largest gives exactly one crop per image, which keeps the split
    arithmetic clean and matches how the dataset is normally benchmarked.
    """
    valid = [b for b in boxes if b[2] > b[0] and b[3] > b[1]]
    if not valid:
        return None
    return max(valid, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def expand_and_clamp(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    margin: float = 0.0,
) -> tuple[int, int, int, int]:
    """Grow a box by `margin` of its own size on each side, clipped to the image."""
    x1, y1, x2, y2 = box
    dx = (x2 - x1) * margin
    dy = (y2 - y1) * margin
    x1 = max(0, round(x1 - dx))
    y1 = max(0, round(y1 - dy))
    x2 = min(width, round(x2 + dx))
    y2 = min(height, round(y2 + dy))
    # A few annotations sit partly or wholly outside the image they describe.
    if x2 <= x1 or y2 <= y1:
        return 0, 0, width, height
    return x1, y1, x2, y2


def crop_and_resize(
    img: Image.Image,
    box: tuple[int, int, int, int] | None,
    size: int,
    margin: float = 0.0,
) -> Image.Image:
    """Crop to `box` (whole image if None) and resize square to `size`."""
    img = img.convert("RGB")
    if box is not None:
        box = expand_and_clamp(box, img.width, img.height, margin)
        img = img.crop(box)
    return img.resize((size, size), Image.BILINEAR)


# ── Splitting ─────────────────────────────────────────────────────────────────
def split_indices(n: int, ratios: dict, seed: int) -> dict[str, list[int]]:
    """Shuffle range(n) and slice it into train/val/test by ratio.

    Rounds, then absorbs the remainder into test so the parts always sum to
    exactly n, and guarantees val and test each get at least one item once
    n >= 3 -- otherwise the smallest breeds would vanish from evaluation.
    """
    idx = list(range(n))
    random.Random(seed).shuffle(idx)

    n_train = round(n * ratios["train"])
    n_val = round(n * ratios["val"])
    if n >= 3:
        n_train = min(n_train, n - 2)
        n_val = max(1, min(n_val, n - n_train - 1))
    n_test = n - n_train - n_val

    assert n_train + n_val + n_test == n, "split sizes must partition the set"
    return {
        "train": idx[:n_train],
        "val": idx[n_train : n_train + n_val],
        "test": idx[n_train + n_val :],
    }


def breed_name(folder_name: str) -> str:
    """'n02085620-Chihuahua' -> 'Chihuahua'. Leaves unprefixed names alone."""
    return folder_name.split("-", 1)[1] if "-" in folder_name else folder_name


# ── Stage ─────────────────────────────────────────────────────────────────────
def build(params: dict, summary_path: Path = SUMMARY_PATH) -> dict:
    raw_dir = Path(params["raw_dir"])
    out_dir = Path(params["out_dir"])
    images_dir = raw_dir / IMAGES_SUBDIR
    annotations_dir = raw_dir / ANNOTATIONS_SUBDIR

    if not images_dir.exists():
        raise SystemExit(
            f"[prepare] {images_dir} not found. Run `dvc pull` to fetch the raw "
            f"data, or `python src/prepare.py --download` to build it from scratch."
        )
    use_bbox = params.get("use_bbox_crop", True)
    if use_bbox and not annotations_dir.exists():
        raise SystemExit(
            f"[prepare] use_bbox_crop is on but {annotations_dir} is missing. "
            f"Re-run --download, or set data.use_bbox_crop: false in params.yaml."
        )

    size = params["img_size"]
    margin = params.get("bbox_margin", 0.0)
    quality = params.get("jpeg_quality", 95)
    seed = params["seed"]
    ratios = split_ratios(params)

    # Fresh start so re-runs never mix old and new crops.
    for split in SPLITS:
        if (out_dir / split).exists():
            shutil.rmtree(out_dir / split)

    breed_dirs = sorted(p for p in images_dir.iterdir() if p.is_dir())
    if not breed_dirs:
        raise SystemExit(f"[prepare] No breed folders inside {images_dir}.")

    counts = {s: 0 for s in SPLITS}
    per_breed: dict[str, dict[str, int]] = {}
    missing_boxes = 0

    for breed_dir in tqdm(breed_dirs, desc="[prepare] breeds", unit="breed"):
        label = breed_name(breed_dir.name)
        # Sorted first, so the shuffle depends only on the seed and not on
        # whatever order the filesystem happens to hand back.
        paths = sorted(p for p in breed_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        if not paths:
            continue

        # Per-breed seed keeps each class's shuffle independent, so adding a
        # breed later does not reshuffle every other breed. crc32, not hash():
        # Python randomises str hashing per process, which would make the split
        # differ between runs and defeat the point of tracking it in DVC.
        breed_seed = seed + zlib.crc32(label.encode("utf-8"))
        assignment = split_indices(len(paths), ratios, breed_seed)
        per_breed[label] = {s: len(v) for s, v in assignment.items()}

        for split, indices in assignment.items():
            for i in indices:
                src = paths[i]
                box = None
                if use_bbox:
                    annotation = annotations_dir / breed_dir.name / src.stem
                    if annotation.exists():
                        box = largest_box(parse_boxes(annotation))
                    if box is None:
                        missing_boxes += 1

                with Image.open(src) as img:
                    out = crop_and_resize(img, box, size, margin)

                dest = out_dir / split / label / f"{src.stem}.jpg"
                dest.parent.mkdir(parents=True, exist_ok=True)
                out.save(dest, "JPEG", quality=quality)
                counts[split] += 1

    summary = {
        "n_breeds": len(per_breed),
        "n_images": sum(counts.values()),
        "counts": counts,
        "ratios": {k: round(v, 4) for k, v in ratios.items()},
        "img_size": size,
        "use_bbox_crop": use_bbox,
        "bbox_margin": margin,
        "seed": seed,
        "images_without_bbox": missing_boxes,
        "per_breed": per_breed,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Stanford Dogs: bbox crop + stratified train/val/test split."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch and extract the raw tarballs, then exit. Run this once.",
    )
    args = parser.parse_args()

    params = load_params()

    if args.download:
        download_raw(Path(params["raw_dir"]), params["sources"])
        return

    summary = build(params)
    counts = summary["counts"]
    n = summary["n_images"]
    print(f"\n[prepare] {summary['n_breeds']} breeds, {n} images -> {params['out_dir']}")
    for split in SPLITS:
        share = counts[split] / n if n else 0
        print(f"[prepare]   {split:5s} {counts[split]:6d}  ({share:.1%})")
    if summary["images_without_bbox"]:
        print(
            f"[prepare] {summary['images_without_bbox']} image(s) had no usable "
            f"bounding box and were kept uncropped."
        )
    if summary["n_breeds"] != EXPECTED_BREEDS or n != EXPECTED_IMAGES:
        print(
            f"[prepare] NOTE: expected {EXPECTED_BREEDS} breeds / "
            f"{EXPECTED_IMAGES} images for the full dataset."
        )
    print(f"[prepare] Summary written to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
