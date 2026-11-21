"""Split manifests: one CSV per role.

Every split file is a CSV with at least these columns:

=================  ==============================================================
``path``           image path, relative to an image root (or absolute)
``sku_id``         SKU label
``family_id``      product family of the SKU (optional, used for family metrics)
``source_group``   source photograph the crop was cut from (optional; used for
                   group-level statistics and to check that roles are disjoint)
=================  ==============================================================

A split directory holds the roles of one benchmark:

* fine-tuning: ``train.csv`` and ``val.csv``;
* evaluation: ``gallery.csv``, ``test_known.csv``, ``val_known.csv``, ``val_unknown.csv`` and
  ``test_unknown.csv``. Unknown roles contain SKUs that are absent from the gallery.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

REQUIRED_COLUMNS = ("path", "sku_id")
EVALUATION_ROLES = ("gallery", "test_known", "val_known", "val_unknown", "test_unknown")
TRAINING_ROLES = ("train", "val")


@dataclass(frozen=True)
class Sample:
    """One labelled image."""

    path: str
    sku_id: str
    family_id: str = ""
    source_group: str = ""


def read_manifest(csv_path: str | Path, image_root: str | Path | None = None) -> list[Sample]:
    """Read a split CSV; relative paths are resolved against ``image_root`` when given."""
    csv_path = Path(csv_path)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} lacks required columns: {missing}")
        rows = list(reader)
    root = Path(image_root) if image_root is not None else None
    samples = []
    for row in rows:
        path = Path(row["path"])
        if root is not None and not path.is_absolute():
            path = root / path
        samples.append(
            Sample(
                path=str(path),
                sku_id=row["sku_id"],
                family_id=row.get("family_id", "") or "",
                source_group=row.get("source_group", "") or "",
            )
        )
    return samples


def write_manifest(samples: Iterable[Sample], csv_path: str | Path) -> None:
    """Write samples to a split CSV."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sku_id", "family_id", "source_group"])
        for sample in samples:
            writer.writerow([sample.path, sample.sku_id, sample.family_id, sample.source_group])


def load_split(
    split_dir: str | Path,
    roles: Sequence[str],
    image_root: str | Path | None = None,
) -> dict[str, list[Sample]]:
    """Read the given roles of a split directory; missing optional files are skipped."""
    split_dir = Path(split_dir)
    found = {}
    for role in roles:
        path = split_dir / f"{role}.csv"
        if path.is_file():
            found[role] = read_manifest(path, image_root)
    return found


def family_map(samples: Iterable[Sample]) -> dict[str, str]:
    """``sku_id -> family_id`` for every sample that has a family."""
    return {s.sku_id: s.family_id for s in samples if s.family_id}


def check_disjoint_groups(roles: dict[str, list[Sample]]) -> None:
    """Raise if a source group appears in more than one role."""
    owner: dict[str, str] = {}
    for role, samples in roles.items():
        for sample in samples:
            if not sample.source_group:
                continue
            previous = owner.setdefault(sample.source_group, role)
            if previous != role:
                raise ValueError(
                    f"Source group {sample.source_group!r} is in both {previous} and {role}"
                )
