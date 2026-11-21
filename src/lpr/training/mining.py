"""Iterative hard-negative mining.

At the start of every mining cycle the current model embeds the training set (evaluation
transform) and retrieves each crop's nearest neighbours by exact cosine search. The first cycle
uses the pretrained stop-layer readout, because the head is not trained yet; later cycles use the
projected embedding ``z``.
Neighbours are then sorted into buckets *by rank*, not by a margin criterion:

* different SKU, similarity >= ``min_negative_similarity``: the first ``num_hard`` form the
  **hard** bucket and the following ones (up to ``top_k`` candidates in total) the **semi-hard**
  bucket;
* same SKU, similarity <= ``max_positive_similarity`` (near-copies are skipped): the first
  ``num_positives`` form the **hard** positive bucket and the rest the **semi-hard** one.

The buckets are handed to :class:`lpr.data.sampler.PKSampler`. Because the embedding changes
during training, mining is repeated every cycle, so a sibling that becomes the nearest impostor
late in training is still found.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


def _buckets() -> dict[str, list[int]]:
    return {"hard": [], "semi_hard": []}


@dataclass
class MinedExamples:
    """Mined dataset indices of one anchor, by bucket."""

    positives: dict[str, list[int]] = field(default_factory=_buckets)
    negatives: dict[str, list[int]] = field(default_factory=_buckets)


@dataclass
class MiningConfig:
    """Mining settings."""

    top_k: int = 80
    num_hard: int = 4
    num_positives: int = 2
    min_negative_similarity: float = 0.50
    max_positive_similarity: float = 0.95
    query_chunk: int = 1024


@torch.no_grad()
def mine_neighbours(
    embeddings: torch.Tensor,
    labels: Sequence[int],
    config: MiningConfig | None = None,
) -> dict[int, MinedExamples]:
    """Bucket the nearest neighbours of every training crop.

    Args:
        embeddings: ``[N, dim]`` embeddings of the training set (normalised inside).
        labels: SKU label of every row.
        config: Mining settings.

    Returns:
        ``index -> MinedExamples`` for every anchor that has at least one mined negative.
    """
    config = config or MiningConfig()
    embeddings = F.normalize(embeddings.float(), dim=-1)
    label_tensor = torch.as_tensor(list(labels), device=embeddings.device)
    num_items = embeddings.shape[0]
    k = min(num_items, config.top_k + 1)  # +1 for the anchor itself

    mined: dict[int, MinedExamples] = {}
    for start in range(0, num_items, config.query_chunk):
        block = embeddings[start : start + config.query_chunk]
        scores, indices = (block @ embeddings.T).topk(k, dim=1)
        for row in range(block.shape[0]):
            anchor = start + row
            anchor_label = label_tensor[anchor]
            examples = MinedExamples()
            num_negatives = 0
            for score, index in zip(scores[row].tolist(), indices[row].tolist(), strict=True):
                if index == anchor:
                    continue
                if label_tensor[index] == anchor_label:
                    if score <= config.max_positive_similarity:
                        positives = examples.positives
                        is_hard = len(positives["hard"]) < config.num_positives
                        positives["hard" if is_hard else "semi_hard"].append(index)
                elif score >= config.min_negative_similarity:
                    bucket = "hard" if num_negatives < config.num_hard else "semi_hard"
                    examples.negatives[bucket].append(index)
                    num_negatives += 1
            if num_negatives:
                mined[anchor] = examples
    return mined
