"""Closed-set retrieval metrics on the SKU-level ranking.

* ``R@K`` - share of queries whose SKU is among the first ``K`` ranked SKUs.
* ``mAP@10`` - with one relevant SKU per query after the collapse, average precision at cutoff
  ten is the reciprocal rank, so mAP@10 is the mean reciprocal rank truncated at ten.
* macro-SKU ``R@1`` - the mean over SKUs of the per-SKU top-1 accuracy, so every SKU weighs the
  same regardless of its number of queries.
* family confusion rate (FCR) - an incorrect top-1 prediction is a *family confusion* when the
  predicted and the true SKU share a family. ``FCR_all`` divides by all queries, ``FCR_error`` by
  the incorrect queries only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from lpr.retrieval.search import SkuRanking


@dataclass
class RetrievalMetrics:
    """Closed-set metrics of one representation on one query set, in percent."""

    num_queries: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    map_at_10: float
    macro_recall_at_1: float
    fcr_all: float | None = None
    fcr_error: float | None = None

    def as_dict(self) -> dict[str, float | int | None]:
        return asdict(self)


def rank_of_truth(
    ranking: SkuRanking, true_labels: Sequence[str], max_rank: int = 10
) -> np.ndarray:
    """1-based rank of the true SKU for every query, or 0 when it is not in the first ``max_rank``.

    A query whose SKU is absent from the gallery always gets 0.
    """
    ranked = ranking.ranked_labels(max_rank)
    truth = np.asarray(true_labels, dtype=object)[:, None]
    hit = ranked == truth
    ranks = np.where(hit.any(axis=1), hit.argmax(axis=1) + 1, 0)
    return ranks.astype(np.int64)


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    """Percentage of queries whose true SKU is ranked within the first ``k``."""
    return float(100.0 * np.mean((ranks >= 1) & (ranks <= k)))


def map_at_10(ranks: np.ndarray) -> float:
    """Mean reciprocal rank truncated at ten, in percent (equal to mAP@10 here)."""
    reciprocal = np.where((ranks >= 1) & (ranks <= 10), 1.0 / np.maximum(ranks, 1), 0.0)
    return float(100.0 * reciprocal.mean())


def macro_recall_at_1(ranks: np.ndarray, true_labels: Sequence[str]) -> float:
    """Mean over SKUs of the per-SKU top-1 accuracy, in percent."""
    labels = np.asarray(true_labels, dtype=object)
    correct = ranks == 1
    per_sku = [correct[labels == sku].mean() for sku in np.unique(labels)]
    return float(100.0 * np.mean(per_sku))


def family_confusion_rates(
    predicted: Sequence[str],
    true_labels: Sequence[str],
    family_of: Mapping[str, str],
) -> tuple[float, float]:
    """``(FCR_all, FCR_error)`` in percent.

    ``family_of`` maps every SKU id to its family id. ``FCR_error`` is 0 when there is no error.
    """
    predicted = np.asarray(predicted, dtype=object)
    truth = np.asarray(true_labels, dtype=object)
    wrong = predicted != truth
    same_family = np.array(
        [family_of.get(p) is not None and family_of.get(p) == family_of.get(t)
         for p, t in zip(predicted, truth, strict=True)]
    )
    confusion = wrong & same_family
    fcr_all = 100.0 * confusion.mean()
    fcr_error = 100.0 * confusion.sum() / wrong.sum() if wrong.any() else 0.0
    return float(fcr_all), float(fcr_error)


def closed_set_metrics(
    ranking: SkuRanking,
    true_labels: Sequence[str],
    family_of: Mapping[str, str] | None = None,
) -> RetrievalMetrics:
    """All closed-set metrics of one ranking against the true SKUs of its queries."""
    ranks = rank_of_truth(ranking, true_labels, max_rank=10)
    metrics = RetrievalMetrics(
        num_queries=len(true_labels),
        recall_at_1=recall_at_k(ranks, 1),
        recall_at_5=recall_at_k(ranks, 5),
        recall_at_10=recall_at_k(ranks, 10),
        map_at_10=map_at_10(ranks),
        macro_recall_at_1=macro_recall_at_1(ranks, true_labels),
    )
    if family_of is not None:
        predicted = ranking.ranked_labels(1)[:, 0]
        metrics.fcr_all, metrics.fcr_error = family_confusion_rates(
            predicted, true_labels, family_of
        )
    return metrics
