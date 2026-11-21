"""Full evaluation of one representation on one benchmark split.

Given embeddings of every role of a split (``gallery``, ``test_known`` and, when present,
``val_known``, ``val_unknown`` and ``test_unknown``), :func:`evaluate_representation` reports for
each reference count:

* closed-set metrics on the known test queries (:mod:`lpr.evaluation.metrics`);
* the rejection threshold selected on the validation queries and the operating point on the test
  queries, overall and per unknown segment (:mod:`lpr.evaluation.rejection`);
* AUROC of the top-1 score and of the top-1/top-2 margin for known-query correctness;
* per-query records for paired statistics (:mod:`lpr.stats.resampling`).

An unknown test SKU is a *same-family* unknown when another SKU of its family is in the gallery
and a *cross-family* unknown otherwise.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch

from lpr.data.manifest import Sample, family_map
from lpr.evaluation.galleries import nested_gallery_indices, prototype_gallery
from lpr.evaluation.metrics import closed_set_metrics, rank_of_truth
from lpr.evaluation.rejection import (
    Threshold,
    auroc,
    operating_point,
    select_score_margin_threshold,
    select_score_threshold,
)
from lpr.retrieval.search import SkuRanking, rank_queries
from lpr.stats.resampling import clustered_bootstrap_rate_ci

Rule = Literal["score", "score_margin"]


@dataclass
class QueryRecords:
    """Per-query outcome of the known test queries of one representation and gallery."""

    paths: list[str]
    sku_ids: list[str]
    source_groups: list[str]
    predicted: list[str]
    correct: np.ndarray
    s1: np.ndarray
    s2: np.ndarray

    def rows(self) -> list[dict]:
        return [
            {"path": p, "sku_id": y, "source_group": g, "predicted": q,
             "correct": int(c), "s1": float(a), "s2": float(b)}
            for p, y, g, q, c, a, b in zip(
                self.paths, self.sku_ids, self.source_groups, self.predicted,
                self.correct, self.s1, self.s2, strict=True,
            )
        ]


@dataclass
class ReferenceResult:
    references: int
    metrics: dict
    gallery: str = "crops"
    threshold: dict | None = None
    rejection: dict = field(default_factory=dict)
    auroc: dict = field(default_factory=dict)
    records: QueryRecords | None = None


def _rank(
    embeddings: dict[str, torch.Tensor],
    split: dict[str, list[Sample]],
    role: str,
    gallery_index: list[int],
) -> SkuRanking:
    gallery = embeddings["gallery"][gallery_index]
    labels = [split["gallery"][i].sku_id for i in gallery_index]
    return rank_queries(embeddings[role], gallery, labels)


def _unknown_segments(
    unknown: Sequence[Sample],
    gallery_skus: set[str],
    family_of: dict[str, str],
) -> dict[str, np.ndarray]:
    gallery_families = {family_of[s] for s in gallery_skus if s in family_of}
    same = np.array([bool(s.family_id) and s.family_id in gallery_families for s in unknown])
    segments = {"all": np.ones(len(unknown), dtype=bool)}
    if family_of:
        segments["same_family"] = same
        segments["cross_family"] = ~same
    return segments


def evaluate_representation(
    embeddings: dict[str, torch.Tensor],
    split: dict[str, list[Sample]],
    references: Sequence[int] = (10, 1),
    gallery_seed: int = 42,
    rule: Rule = "score",
    target_far: float = 5.0,
    prototypes: bool = False,
    bootstrap_repeats: int = 10_000,
) -> list[ReferenceResult]:
    """Evaluate one representation; ``embeddings`` and ``split`` are keyed by role.

    With ``prototypes``, every reference count is also evaluated with a prototype gallery (one
    mean embedding per SKU); those results carry closed-set metrics only.
    """
    family_of = family_map(s for samples in split.values() for s in samples)
    results = []
    for count in references:
        gallery_index = nested_gallery_indices(split["gallery"], count, seed=gallery_seed)
        gallery_skus = {split["gallery"][i].sku_id for i in gallery_index}

        known = split["test_known"]
        ranking = _rank(embeddings, split, "test_known", gallery_index)
        truth = [s.sku_id for s in known]
        correct = rank_of_truth(ranking, truth, max_rank=1) == 1
        s1, s2 = ranking.top_two()
        result = ReferenceResult(
            references=count,
            metrics=closed_set_metrics(ranking, truth, family_of or None).as_dict(),
            auroc={"score": auroc(s1, correct), "margin": auroc(s1 - s2, correct)},
            records=QueryRecords(
                paths=[s.path for s in known],
                sku_ids=truth,
                source_groups=[s.source_group for s in known],
                predicted=list(ranking.ranked_labels(1)[:, 0]),
                correct=correct,
                s1=s1,
                s2=s2,
            ),
        )

        if "val_unknown" in split and "test_unknown" in split:
            v_unknown = _rank(embeddings, split, "val_unknown", gallery_index)
            vu_s1, vu_s2 = v_unknown.top_two()
            if rule == "score":
                threshold = select_score_threshold(vu_s1, target_far)
            else:
                v_known = _rank(embeddings, split, "val_known", gallery_index)
                vk_truth = [s.sku_id for s in split["val_known"]]
                vk_correct = rank_of_truth(v_known, vk_truth, max_rank=1) == 1
                vk_s1, vk_s2 = v_known.top_two()
                threshold = select_score_margin_threshold(
                    vk_s1, vk_s2, vk_correct, vu_s1, vu_s2, target_far
                )
            result.threshold = {"rule": rule, "score": threshold.score, "margin": threshold.margin}
            t_unknown = _rank(embeddings, split, "test_unknown", gallery_index)
            tu_s1, tu_s2 = t_unknown.top_two()
            segments = _unknown_segments(split["test_unknown"], gallery_skus, family_of)
            unknown_skus = np.asarray([s.sku_id for s in split["test_unknown"]], dtype=object)
            for segment, mask in segments.items():
                point = _operating_point(threshold, s1, s2, correct, tu_s1[mask], tu_s2[mask])
                accepted = threshold.accepts(tu_s1[mask], tu_s2[mask])
                # FAR interval: resample the unknown SKUs of the segment.
                point["far_ci"] = clustered_bootstrap_rate_ci(
                    accepted, list(unknown_skus[mask]), repeats=bootstrap_repeats
                )
                result.rejection[segment] = point
        results.append(result)
        if prototypes:
            vectors, sku_ids = prototype_gallery(
                embeddings["gallery"][gallery_index],
                [split["gallery"][i].sku_id for i in gallery_index],
            )
            proto_ranking = rank_queries(embeddings["test_known"], vectors, sku_ids)
            results.append(ReferenceResult(
                references=count,
                gallery="prototype",
                metrics=closed_set_metrics(proto_ranking, truth, family_of or None).as_dict(),
            ))
    return results


def _operating_point(
    threshold: Threshold,
    known_s1: np.ndarray,
    known_s2: np.ndarray,
    known_correct: np.ndarray,
    unknown_s1: np.ndarray,
    unknown_s2: np.ndarray,
) -> dict:
    point = operating_point(
        threshold, known_s1, known_correct, unknown_s1, known_s2=known_s2, unknown_s2=unknown_s2
    )
    return point.as_dict()
