"""Reference-image configurations.

The primary operating point is a crop gallery with up to ten references per SKU (Ref. = 10);
the single-reference condition (Ref. = 1) is reported alongside it. Galleries for 1, 3, 5 and 10
references are *nested*: within each SKU the crops are put in a seeded pseudo-random order and
the Ref. = N gallery keeps the first N, so every smaller gallery is contained in the larger ones
and the query set stays fixed. Different gallery seeds change which reference images are drawn.

A *prototype* gallery replaces the crops of each SKU by their mean embedding.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence

import torch
import torch.nn.functional as F

from lpr.data.manifest import Sample


def _order_key(seed: int, sample: Sample) -> int:
    digest = hashlib.md5(f"{seed}:{sample.sku_id}:{sample.path}".encode()).hexdigest()
    return int(digest[:16], 16)


def nested_gallery_indices(
    gallery: Sequence[Sample],
    references: int,
    seed: int = 42,
) -> list[int]:
    """Indices of the Ref. = ``references`` gallery, nested across reference counts.

    SKUs with fewer crops keep all of them.
    """
    by_sku: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(gallery):
        by_sku[sample.sku_id].append(index)
    chosen = []
    for indices in by_sku.values():
        ordered = sorted(indices, key=lambda i: _order_key(seed, gallery[i]))
        chosen.extend(ordered[:references])
    return sorted(chosen)


def prototype_gallery(
    embeddings: torch.Tensor,
    labels: Sequence[str],
) -> tuple[torch.Tensor, list[str]]:
    """Mean embedding per SKU, L2-normalised; returns ``(prototypes, sku_ids)``."""
    sku_ids = sorted(set(labels))
    row = {sku: i for i, sku in enumerate(sku_ids)}
    index = torch.tensor([row[label] for label in labels])
    sums = torch.zeros(len(sku_ids), embeddings.shape[1]).index_add_(0, index, embeddings.float())
    return F.normalize(sums, dim=-1), sku_ids
