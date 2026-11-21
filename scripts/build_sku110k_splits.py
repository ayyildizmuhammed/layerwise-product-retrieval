#!/usr/bin/env python3
"""Build the SKU110K evaluation split from the annotations and the rebuilt crops.

Family = brand. Source group = the SKU110K shelf photograph a crop was cut from. The rules are
those of :mod:`lpr.data.splits`: 259 catalogue SKUs, 10 validation-unknown and 91 test-unknown
SKUs (63 of them cross-family), source-group-disjoint roles and byte-identical query-gallery
duplicates removed. The published split in ``data/sku110k/splits`` was built with the defaults.

Example::

    python scripts/build_sku110k_splits.py --crops-dir data/crops --output-dir data/sku110k/splits
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from lpr.data.manifest import Sample, write_manifest
from lpr.data.splits import SplitConfig, build_evaluation_split

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SKU110K evaluation split.")
    parser.add_argument("--annotations", type=Path,
                        default=REPO_ROOT / "data" / "sku110k" / "annotations.csv")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with args.annotations.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    samples = [Sample(r["filepath"], r["sku_id"], r["brand_id"], r["sku110k_image"]) for r in rows]
    sha256 = {r["filepath"]: r["sha256"] for r in rows}
    result = build_evaluation_split(
        samples, SplitConfig(seed=args.seed), content_hash=lambda s: sha256[s.path]
    )
    for role, role_samples in result.roles.items():
        write_manifest(role_samples, args.output_dir / f"{role}.csv")
    summary = result.summary()
    (args.output_dir / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
