"""
tests/test_prepare.py
Unit + end-to-end tests for the `prepare` stage.

Everything runs against a tiny synthetic Stanford-Dogs-shaped fixture (3 breeds,
solid-colour JPEGs, hand-written annotation XML), so the suite finishes in about
a second and needs no network and no 757 MB download.

What we actually care about:
  - the largest bounding box wins when an image contains several dogs
  - out-of-bounds annotations get clamped instead of crashing PIL
  - the split is really 70/15/15, really per breed, and really disjoint
  - the same seed gives the same split twice -- this is what makes `dvc repro`
    meaningful, and it is the thing most likely to break silently
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from PIL import Image

from src.prepare import (
    breed_name,
    build,
    crop_and_resize,
    expand_and_clamp,
    largest_box,
    parse_boxes,
    split_indices,
    split_ratios,
)

ANNOTATION_TEMPLATE = """<annotation>
  <folder>{folder}</folder>
  <filename>{stem}</filename>
  <size><width>{w}</width><height>{h}</height><depth>3</depth></size>
{objects}</annotation>
"""

OBJECT_TEMPLATE = """  <object>
    <name>{name}</name>
    <bndbox>
      <xmin>{x1}</xmin><ymin>{y1}</ymin><xmax>{x2}</xmax><ymax>{y2}</ymax>
    </bndbox>
  </object>
"""

RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def write_annotation(path: Path, w: int, h: int, boxes, name: str = "dog") -> None:
    objects = "".join(
        OBJECT_TEMPLATE.format(name=name, x1=b[0], y1=b[1], x2=b[2], y2=b[3])
        for b in boxes
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        ANNOTATION_TEMPLATE.format(
            folder=path.parent.name, stem=path.name, w=w, h=h, objects=objects
        )
    )


# ── Ratios come from val_split / test_split ───────────────────────────────────
def test_train_ratio_is_the_remainder():
    assert split_ratios({"val_split": 0.15, "test_split": 0.15}) == pytest.approx(
        {"train": 0.70, "val": 0.15, "test": 0.15}
    )


def test_ratios_that_leave_no_training_data_are_rejected():
    with pytest.raises(SystemExit, match="must be < 1.0"):
        split_ratios({"val_split": 0.6, "test_split": 0.5})


# ── Bounding box parsing ──────────────────────────────────────────────────────
def test_parse_boxes_reads_every_object(tmp_path):
    ann = tmp_path / "n02085620-Chihuahua" / "n02085620_7"
    write_annotation(ann, 250, 188, [(10, 10, 50, 50), (60, 60, 200, 180)])
    assert parse_boxes(ann) == [(10, 10, 50, 50), (60, 60, 200, 180)]


def test_parse_boxes_skips_malformed_entry(tmp_path):
    """A junk coordinate should drop that one object, not kill a 20k-image run."""
    ann = tmp_path / "breed" / "img"
    ann.parent.mkdir(parents=True)
    ann.write_text(
        "<annotation><object><bndbox>"
        "<xmin>x</xmin><ymin>0</ymin><xmax>5</xmax><ymax>5</ymax>"
        "</bndbox></object>"
        "<object><bndbox>"
        "<xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax>"
        "</bndbox></object></annotation>"
    )
    assert parse_boxes(ann) == [(1, 1, 9, 9)]


def test_largest_box_picks_biggest_area():
    assert largest_box([(0, 0, 10, 10), (0, 0, 30, 30), (0, 0, 20, 20)]) == (
        0,
        0,
        30,
        30,
    )


def test_largest_box_ignores_degenerate_boxes():
    assert largest_box([(5, 5, 5, 20), (0, 0, 4, 4)]) == (0, 0, 4, 4)


def test_largest_box_returns_none_when_nothing_usable():
    assert largest_box([]) is None
    assert largest_box([(5, 5, 5, 5)]) is None


# ── Clamping ──────────────────────────────────────────────────────────────────
def test_clamp_keeps_box_inside_image():
    assert expand_and_clamp((-20, -5, 500, 400), 250, 188) == (0, 0, 250, 188)


def test_margin_expands_by_fraction_of_box_size():
    # 100x100 box, 10% margin -> 10px on each side
    assert expand_and_clamp((100, 100, 200, 200), 400, 400, margin=0.1) == (
        90,
        90,
        210,
        210,
    )


def test_box_entirely_outside_image_falls_back_to_full_image():
    assert expand_and_clamp((300, 300, 400, 400), 100, 100) == (0, 0, 100, 100)


def test_crop_and_resize_always_returns_square_rgb():
    img = Image.new("L", (250, 188), color=128)  # greyscale on purpose
    out = crop_and_resize(img, (10, 10, 60, 30), size=224)
    assert out.size == (224, 224)
    assert out.mode == "RGB"


def test_crop_with_no_box_uses_whole_image():
    img = Image.new("RGB", (64, 32), color=(255, 0, 0))
    assert crop_and_resize(img, None, size=16).size == (16, 16)


# ── Splitting ─────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("n", [3, 10, 47, 100, 152, 171, 20580])
def test_split_partitions_exactly(n):
    parts = split_indices(n, RATIOS, seed=42)
    flat = [i for part in parts.values() for i in part]

    assert len(flat) == n, "every item must land somewhere"
    assert sorted(flat) == list(range(n)), "no duplicates, nothing dropped"
    assert all(len(p) > 0 for p in parts.values()), "no empty split"


def test_split_hits_the_requested_ratios():
    parts = split_indices(10_000, RATIOS, seed=42)
    assert len(parts["train"]) == pytest.approx(7000, abs=2)
    assert len(parts["val"]) == pytest.approx(1500, abs=2)
    assert len(parts["test"]) == pytest.approx(1500, abs=2)


def test_split_is_deterministic_for_a_given_seed():
    assert split_indices(200, RATIOS, seed=42) == split_indices(200, RATIOS, seed=42)


def test_different_seeds_give_different_splits():
    assert split_indices(200, RATIOS, seed=42) != split_indices(200, RATIOS, seed=7)


def test_breed_name_strips_the_wnid():
    assert breed_name("n02085620-Chihuahua") == "Chihuahua"
    assert breed_name("n02085936-Maltese_dog") == "Maltese_dog"
    assert breed_name("Chihuahua") == "Chihuahua"


# ── End to end on a synthetic dataset ─────────────────────────────────────────
@pytest.fixture
def fake_dataset(tmp_path):
    """Three breeds x 20 images, each with a decoy box and a larger real box."""
    raw = tmp_path / "raw"
    for folder in (
        "n02085620-Chihuahua",
        "n02085936-Maltese_dog",
        "n02088364-beagle",
    ):
        for i in range(20):
            stem = f"{folder.split('-')[0]}_{i}"
            img_path = raw / "Images" / folder / f"{stem}.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (200, 150), color=(i * 5 % 256, 100, 50)).save(img_path)
            write_annotation(
                raw / "Annotation" / folder / stem,
                200,
                150,
                [(0, 0, 20, 20), (40, 30, 180, 140)],
            )
    return raw


def _params(raw: Path, out: Path, **overrides) -> dict:
    params = {
        "raw_dir": str(raw),
        "out_dir": str(out),
        "img_size": 32,
        "use_bbox_crop": True,
        "bbox_margin": 0.0,
        "jpeg_quality": 90,
        "val_split": 0.15,
        "test_split": 0.15,
        "seed": 42,
    }
    params.update(overrides)
    return params


def test_build_produces_all_splits_for_all_breeds(fake_dataset, tmp_path):
    out = tmp_path / "processed"
    summary = build(_params(fake_dataset, out), tmp_path / "s.json")

    assert summary["n_breeds"] == 3
    assert summary["n_images"] == 60
    for split in ("train", "val", "test"):
        breeds = sorted(p.name for p in (out / split).iterdir() if p.is_dir())
        assert breeds == ["Chihuahua", "Maltese_dog", "beagle"], (
            f"{split} is missing breeds -- the split is not stratified"
        )


def test_build_split_sizes_match_ratios_per_breed(fake_dataset, tmp_path):
    summary = build(
        _params(fake_dataset, tmp_path / "processed"), tmp_path / "s.json"
    )
    for breed, counts in summary["per_breed"].items():
        assert counts == {"train": 14, "val": 3, "test": 3}, breed


def test_build_writes_cropped_images_at_the_right_size(fake_dataset, tmp_path):
    out = tmp_path / "processed"
    build(_params(fake_dataset, out, img_size=64), tmp_path / "s.json")
    written = list(out.rglob("*.jpg"))
    assert len(written) == 60
    with Image.open(written[0]) as img:
        assert img.size == (64, 64)


def test_no_image_appears_in_two_splits(fake_dataset, tmp_path):
    out = tmp_path / "processed"
    build(_params(fake_dataset, out), tmp_path / "s.json")
    seen: dict[tuple[str, str], str] = {}
    for split in ("train", "val", "test"):
        for path in (out / split).rglob("*.jpg"):
            key = (path.parent.name, path.name)
            assert key not in seen, f"{key} is in both {seen.get(key)} and {split}"
            seen[key] = split


def test_class_lists_match_across_splits(fake_dataset, tmp_path):
    """train.py asserts this -- if it ever fails, label ids silently shift."""
    out = tmp_path / "processed"
    build(_params(fake_dataset, out), tmp_path / "s.json")

    def classes(split):
        return sorted(p.name for p in (out / split).iterdir() if p.is_dir())

    assert classes("train") == classes("val") == classes("test")


def test_build_is_reproducible(fake_dataset, tmp_path):
    """Same inputs and params -> same assignment, in two separate output dirs."""
    first = build(_params(fake_dataset, tmp_path / "a"), tmp_path / "s1.json")
    second = build(_params(fake_dataset, tmp_path / "b"), tmp_path / "s2.json")

    def layout(root):
        return sorted(str(p.relative_to(root)) for p in root.rglob("*.jpg"))

    assert layout(tmp_path / "a") == layout(tmp_path / "b")
    assert first["per_breed"] == second["per_breed"]


def test_build_rerun_does_not_accumulate_stale_files(fake_dataset, tmp_path):
    out = tmp_path / "processed"
    build(_params(fake_dataset, out), tmp_path / "s.json")
    (out / "train" / "Chihuahua" / "stale.jpg").write_bytes(b"not a real jpeg")
    build(_params(fake_dataset, out), tmp_path / "s.json")
    assert not (out / "train" / "Chihuahua" / "stale.jpg").exists()


def test_summary_json_is_written_and_valid(fake_dataset, tmp_path):
    summary_path = tmp_path / "reports" / "split_summary.json"
    build(_params(fake_dataset, tmp_path / "processed"), summary_path)
    summary = json.loads(summary_path.read_text())
    assert summary["n_images"] == 60
    assert sum(summary["counts"].values()) == summary["n_images"]
    assert summary["images_without_bbox"] == 0


def test_missing_annotation_keeps_image_uncropped(fake_dataset, tmp_path):
    """Deleting an annotation must not lose the image -- just skip the crop."""
    (fake_dataset / "Annotation" / "n02088364-beagle" / "n02088364_0").unlink()
    summary = build(
        _params(fake_dataset, tmp_path / "processed"), tmp_path / "s.json"
    )
    assert summary["images_without_bbox"] == 1
    assert summary["n_images"] == 60


def test_bbox_crop_disabled_skips_annotations_entirely(fake_dataset, tmp_path):
    shutil.rmtree(fake_dataset / "Annotation")
    summary = build(
        _params(fake_dataset, tmp_path / "processed", use_bbox_crop=False),
        tmp_path / "s.json",
    )
    assert summary["n_images"] == 60
    assert summary["use_bbox_crop"] is False


def test_bbox_crop_enabled_without_annotations_is_an_actionable_error(
    fake_dataset, tmp_path
):
    shutil.rmtree(fake_dataset / "Annotation")
    with pytest.raises(SystemExit, match="use_bbox_crop"):
        build(_params(fake_dataset, tmp_path / "processed"), tmp_path / "s.json")


def test_missing_raw_dir_is_an_actionable_error(tmp_path):
    with pytest.raises(SystemExit, match="dvc pull"):
        build(_params(tmp_path / "nope", tmp_path / "out"), tmp_path / "s.json")
