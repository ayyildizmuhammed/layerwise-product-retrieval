#!/usr/bin/env python3
"""Rebuild the SKU110K crops from an official SKU110K download.

Every crop is defined by an SKU110K image name and the row number (``box_index``) of its box among
that image's rows in the official annotation files. The box is converted to five-decimal
normalised coordinates, expanded by 5 % of its size on each side, rounded outward to whole pixels,
clamped to the image and saved as JPEG (quality 95).

Download SKU110K from its authors (https://github.com/eg4000/SKU110K_CVPR19) and extract it; the
folder must contain ``images/`` and ``annotations/``.

Example::

    python scripts/rebuild_sku110k_crops.py --sku110k-root SKU110K_fixed --output-dir data/crops
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
PAD = 0.05
JPEG_QUALITY = 95
ANNOTATION_FILES = ("annotations_train.csv", "annotations_val.csv", "annotations_test.csv")
Box = tuple[float, float, float, float]


def normalised_box(row: list[str]) -> Box:
    """Centre x, centre y, width and height relative to the image, rounded to five decimals."""
    x1, y1, x2, y2 = (float(value) for value in row[1:5])
    width, height = float(row[6]), float(row[7])
    left, top, right, bottom = x1 / width, y1 / height, x2 / width, y2 / height
    values = ((left + right) / 2, (top + bottom) / 2, right - left, bottom - top)
    return tuple(float(f"{value:.5f}") for value in values)  # type: ignore[return-value]


def crop_box(box: Box, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Pixel box expanded by ``PAD`` on each side, rounded outward and clamped to the image."""
    image_width, image_height = size
    xc, yc, width, height = (min(max(value, 0.0), 1.0) for value in box)
    box_width, box_height = width * image_width, height * image_height
    centre_x, centre_y = xc * image_width, yc * image_height
    left = math.floor(centre_x - box_width / 2 - PAD * box_width)
    top = math.floor(centre_y - box_height / 2 - PAD * box_height)
    right = math.ceil(centre_x + box_width / 2 + PAD * box_width)
    bottom = math.ceil(centre_y + box_height / 2 + PAD * box_height)
    return (
        max(0, min(left, image_width)),
        max(0, min(top, image_height)),
        max(0, min(right, image_width)),
        max(0, min(bottom, image_height)),
    )


def load_boxes(annotations_dir: Path, images: set[str]) -> dict[str, list[Box]]:
    """Boxes of the requested images, in the row order of the official annotation files."""
    boxes: dict[str, list[Box]] = defaultdict(list)
    for name in ANNOTATION_FILES:
        with (annotations_dir / name).open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if row and row[0] in images:
                    boxes[row[0]].append(normalised_box(row))
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild the SKU110K crops.")
    parser.add_argument("--sku110k-root", required=True, type=Path,
                        help="folder with images/ and annotations/")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annotations", type=Path,
                        default=REPO_ROOT / "data" / "sku110k" / "annotations.csv")
    args = parser.parse_args()

    with args.annotations.open(newline="", encoding="utf-8") as handle:
        crops = list(csv.DictReader(handle))
    per_image: dict[str, list[dict[str, str]]] = defaultdict(list)
    for crop in crops:
        per_image[crop["sku110k_image"]].append(crop)
    boxes = load_boxes(args.sku110k_root / "annotations", set(per_image))

    written = 0
    for number, (image_name, rows) in enumerate(sorted(per_image.items()), start=1):
        with Image.open(args.sku110k_root / "images" / image_name) as opened:
            image = opened.convert("RGB")
        for row in rows:
            target = args.output_dir / row["filepath"]
            target.parent.mkdir(parents=True, exist_ok=True)
            box = crop_box(boxes[image_name][int(row["box_index"])], image.size)
            image.crop(box).save(target, format="JPEG", quality=JPEG_QUALITY, optimize=True)
            written += 1
        if number % 250 == 0:
            print(f"{number}/{len(per_image)} images, {written} crops", flush=True)
    print(f"Wrote {written} crops to {args.output_dir}")


if __name__ == "__main__":
    main()
