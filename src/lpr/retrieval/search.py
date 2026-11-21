"""Exact cosine retrieval with an SKU-level collapse.

Every crop score is the cosine similarity of two L2-normalised embeddings. Because an SKU may
have several gallery crops, crop scores are collapsed to one score per SKU,

    S(q, c) = max_{i : y_i = c} s(q, g_i),

and SKUs are ranked by ``S``. Every metric works on this SKU-level ranking.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class SkuRanking:
    """SKU-level scores of a set of queries against a gallery.

    Attributes:
        sku_ids: The gallery SKUs, in column order of ``scores``.
        scores: ``[num_queries, num_skus]`` collapsed scores ``S(q, c)``.
        best_gallery_index: ``[num_queries, num_skus]`` index of the gallery crop that produced
            each collapsed score, for qualitative inspection.
    """

    sku_ids: list[str]
    scores: np.ndarray
    best_gallery_index: np.ndarray

    def top_k(self, k: int) -> np.ndarray:
        """Column indices of the ``k`` highest-scoring SKUs per query, best first."""
        k = min(k, self.scores.shape[1])
        part = np.argpartition(-self.scores, kth=k - 1, axis=1)[:, :k]
        order = np.take_along_axis(self.scores, part, axis=1).argsort(axis=1)[:, ::-1]
        return np.take_along_axis(part, order, axis=1)

    def ranked_labels(self, k: int) -> np.ndarray:
        """SKU ids of the ``k`` best SKUs per query, as an object array ``[num_queries, k]``."""
        ids = np.asarray(self.sku_ids, dtype=object)
        return ids[self.top_k(k)]

    def top_two(self) -> tuple[np.ndarray, np.ndarray]:
        """Top-1 score ``s1`` and second-highest score ``s2`` of every query."""
        if self.scores.shape[1] < 2:
            raise ValueError("At least two gallery SKUs are needed for a top-1/top-2 margin.")
        best_two = -np.partition(-self.scores, kth=1, axis=1)[:, :2]
        best_two.sort(axis=1)
        return best_two[:, 1], best_two[:, 0]


def cosine_scores(queries: torch.Tensor, gallery: torch.Tensor) -> torch.Tensor:
    """Cosine similarity matrix ``[num_queries, num_gallery]`` (inputs are normalised here)."""
    return F.normalize(queries.float(), dim=-1) @ F.normalize(gallery.float(), dim=-1).T


def collapse_to_skus(
    scores: torch.Tensor,
    gallery_labels: Sequence[str],
    chunk_size: int = 4096,
) -> SkuRanking:
    """Collapse crop scores to SKU scores by taking the best crop of each SKU.

    Args:
        scores: ``[num_queries, num_gallery]`` crop-level cosine scores.
        gallery_labels: SKU id of every gallery crop.
        chunk_size: Number of queries processed at once, to bound memory.
    """
    if scores.shape[1] != len(gallery_labels):
        raise ValueError("scores and gallery_labels disagree on the gallery size")
    sku_ids = sorted(set(gallery_labels))
    column = {sku: index for index, sku in enumerate(sku_ids)}
    owner = torch.tensor([column[label] for label in gallery_labels], device=scores.device)

    num_queries, num_skus = scores.shape[0], len(sku_ids)
    collapsed = np.empty((num_queries, num_skus), dtype=np.float32)
    best_index = np.empty((num_queries, num_skus), dtype=np.int64)
    for start in range(0, num_queries, chunk_size):
        block = scores[start : start + chunk_size].float()
        sku_max = torch.full((block.shape[0], num_skus), -torch.inf, device=block.device)
        sku_max = sku_max.scatter_reduce(
            1, owner.expand_as(block), block, reduce="amax", include_self=True
        )
        # The gallery crop that reached the maximum for each SKU.
        is_max = block == sku_max.gather(1, owner.expand_as(block))
        candidate = torch.where(is_max, torch.arange(block.shape[1], device=block.device), -1)
        arg = torch.full((block.shape[0], num_skus), -1, device=block.device, dtype=torch.long)
        arg = arg.scatter_reduce(
            1, owner.expand_as(block), candidate, reduce="amax", include_self=True
        )
        collapsed[start : start + block.shape[0]] = sku_max.cpu().numpy()
        best_index[start : start + block.shape[0]] = arg.cpu().numpy()
    return SkuRanking(sku_ids=sku_ids, scores=collapsed, best_gallery_index=best_index)


def rank_queries(
    query_embeddings: torch.Tensor,
    gallery_embeddings: torch.Tensor,
    gallery_labels: Sequence[str],
) -> SkuRanking:
    """Score queries against a gallery and return the SKU-level ranking."""
    return collapse_to_skus(cosine_scores(query_embeddings, gallery_embeddings), gallery_labels)
