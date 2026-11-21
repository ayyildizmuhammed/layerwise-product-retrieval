"""Batch-hard triplet loss with cosine distance.

For every anchor ``i`` of a batch, the hardest positive is the same-SKU sample farthest from it
and the hardest negative is the different-SKU sample closest to it:

    L = mean_i max(0, d(i, hardest positive) - d(i, hardest negative) + m),

with ``d(u, v) = 1 - u^T v`` on L2-normalised embeddings and margin ``m = 0.3``
(Hermans et al., 2017).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_hard_triplet_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Batch-hard triplet loss.

    Args:
        embeddings: ``[batch, dim]`` embeddings (normalised inside).
        labels: ``[batch]`` integer SKU labels.
        margin: Triplet margin ``m``.

    Anchors without a positive or without a negative in the batch do not contribute. Returns a
    scalar; zero when no anchor is valid.
    """
    embeddings = F.normalize(embeddings, dim=-1)
    distance = 1.0 - embeddings @ embeddings.T

    same = labels[:, None] == labels[None, :]
    eye = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = same & ~eye
    negative = ~same

    hardest_positive = distance.masked_fill(~positive, -torch.inf).amax(dim=1)
    hardest_negative = distance.masked_fill(~negative, torch.inf).amin(dim=1)

    valid = positive.any(dim=1) & negative.any(dim=1)
    if not valid.any():
        return embeddings.sum() * 0.0
    losses = F.relu(hardest_positive[valid] - hardest_negative[valid] + margin)
    return losses.mean()


class BatchHardTripletLoss(nn.Module):
    """Module form of :func:`batch_hard_triplet_loss`."""

    def __init__(self, margin: float = 0.3) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return batch_hard_triplet_loss(embeddings, labels, self.margin)
