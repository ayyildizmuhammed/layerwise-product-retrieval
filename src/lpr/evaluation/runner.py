"""Evaluate several representations on one split and write the results.

Outputs in ``output_dir``:

* ``results.json`` - every metric of every representation and reference count;
* ``summary.csv`` - one row per representation and reference count;
* ``per_query/<representation>_ref<N>.csv`` - per-query outcomes of the known test queries;
* ``paired.csv`` - paired comparisons (bootstrap CIs and randomisation p-values), when pairs
  are requested.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import torch

from lpr.data.manifest import Sample
from lpr.evaluation.benchmark import ReferenceResult, Rule, evaluate_representation
from lpr.models.representations import Representation
from lpr.retrieval.embed import embed_samples
from lpr.stats.resampling import compare

log = logging.getLogger(__name__)


def evaluate_all(
    representations: Sequence[Representation],
    split: dict[str, list[Sample]],
    output_dir: str | Path,
    references: Sequence[int] = (10, 1),
    gallery_seed: int = 42,
    rule: Rule = "score",
    target_far: float = 5.0,
    pairs: Sequence[tuple[str, str]] = (),
    prototypes: bool = False,
    bootstrap_repeats: int = 10_000,
    device: torch.device | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
) -> dict[str, list[ReferenceResult]]:
    """Embed every role with every representation, evaluate, and write the outputs."""
    output_dir = Path(output_dir)
    (output_dir / "per_query").mkdir(parents=True, exist_ok=True)
    results: dict[str, list[ReferenceResult]] = {}
    for representation in representations:
        run_on = device
        on_mps = device is not None and device.type == "mps"
        if on_mps and getattr(representation, "cpu_on_mps", False):
            run_on = torch.device("cpu")
        if run_on is not None:
            representation.to(run_on)
        representation.eval()
        embeddings = {
            role: embed_samples(
                samples, representation.transform, representation.encode, device=run_on,
                batch_size=batch_size, num_workers=num_workers,
                description=f"{representation.names[0]}: {role}",
            )
            for role, samples in split.items()
        }
        for name in representation.names:
            by_role = {role: vectors[name] for role, vectors in embeddings.items()}
            results[name] = evaluate_representation(
                by_role, split, references, gallery_seed, rule, target_far, prototypes
            )
            recall = {r.references: r.metrics["recall_at_1"]
                      for r in results[name] if r.gallery == "crops"}
            log.info("%s R@1 by reference count: %s", name, recall)
        representation.cpu()

    _write(results, output_dir, pairs, bootstrap_repeats)
    return results


def _write(
    results: dict[str, list[ReferenceResult]],
    output_dir: Path,
    pairs: Sequence[tuple[str, str]],
    bootstrap_repeats: int,
) -> None:
    payload = {
        name: [
            {"references": r.references, "gallery": r.gallery, "metrics": r.metrics,
             "threshold": r.threshold, "rejection": r.rejection, "auroc": r.auroc}
            for r in per_reference
        ]
        for name, per_reference in results.items()
    }
    (output_dir / "results.json").write_text(json.dumps(payload, indent=2))

    with (output_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["representation", "gallery", "references", "R@1", "R@5", "R@10", "mAP@10",
                         "macro_R@1", "FCR_all", "FCR_error", "threshold", "FAR_all",
                         "FAR_same_family", "FAR_cross_family", "known_coverage"])
        for name, per_reference in results.items():
            for r in per_reference:
                m, rej = r.metrics, r.rejection
                writer.writerow([
                    name, r.gallery, r.references, m["recall_at_1"], m["recall_at_5"],
                    m["recall_at_10"], m["map_at_10"], m["macro_recall_at_1"],
                    m["fcr_all"], m["fcr_error"],
                    r.threshold["score"] if r.threshold else "",
                    rej.get("all", {}).get("far", ""),
                    rej.get("same_family", {}).get("far", ""),
                    rej.get("cross_family", {}).get("far", ""),
                    rej.get("all", {}).get("known_coverage", ""),
                ])

    for name, per_reference in results.items():
        for r in per_reference:
            if r.records is None:
                continue
            suffix = "" if r.gallery == "crops" else f"_{r.gallery}"
            path = output_dir / "per_query" / f"{name}_ref{r.references}{suffix}.csv"
            rows = r.records.rows()
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

    if not pairs:
        return
    with (output_dir / "paired.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["a", "b", "references", "R@1_a", "R@1_b", "difference",
                         "ci_low", "ci_high", "p_value"])
        for a, b in pairs:
            crops_a = [r for r in results[a] if r.gallery == "crops"]
            crops_b = [r for r in results[b] if r.gallery == "crops"]
            for ra, rb in zip(crops_a, crops_b, strict=True):
                groups = [g or p for g, p in zip(ra.records.source_groups, ra.records.paths,
                                                 strict=True)]
                paired = compare(ra.records.correct, rb.records.correct, groups,
                                 repeats=bootstrap_repeats)
                writer.writerow([a, b, ra.references, paired.accuracy_a, paired.accuracy_b,
                                 paired.difference, *paired.ci_difference, paired.p_value])
