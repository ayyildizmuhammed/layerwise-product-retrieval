#!/usr/bin/env python3
"""Check rebuilt SKU110K crops against ``data/sku110k/annotations.csv``.

Reports missing files, size mismatches and files whose SHA-256 differs from the published one.

Example::

    python scripts/verify_sku110k_crops.py --crops-dir data/crops
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify rebuilt SKU110K crops.")
    parser.add_argument("--crops-dir", required=True, type=Path)
    parser.add_argument("--annotations", type=Path,
                        default=REPO_ROOT / "data" / "sku110k" / "annotations.csv")
    args = parser.parse_args()

    with args.annotations.open(newline="", encoding="utf-8") as handle:
        crops = list(csv.DictReader(handle))
    missing, wrong_size, different = [], [], []
    for crop in crops:
        path = args.crops_dir / crop["filepath"]
        if not path.is_file():
            missing.append(crop["filepath"])
            continue
        with Image.open(path) as image:
            if image.size != (int(crop["width"]), int(crop["height"])):
                wrong_size.append(crop["filepath"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != crop["sha256"]:
            different.append(crop["filepath"])
    identical = len(crops) - len(missing) - len(different)
    print(f"{len(crops)} crops: {identical} byte-identical, {len(different)} different "
          f"({len(wrong_size)} of them with a different size), {len(missing)} missing")
    for name, items in (("missing", missing), ("different", different)):
        for item in items[:10]:
            print(f"  {name}: {item}")
    return 1 if missing or wrong_size else 0


if __name__ == "__main__":
    raise SystemExit(main())
