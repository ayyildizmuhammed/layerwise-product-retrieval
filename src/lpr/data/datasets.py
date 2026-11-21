"""PyTorch dataset over manifest samples."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch.utils.data import Dataset

from lpr.data.manifest import Sample
from lpr.data.transforms import load_rgb


class ImageDataset(Dataset):
    """Images of a manifest with integer SKU labels.

    Args:
        samples: Manifest rows.
        transform: PIL image to tensor.
        label_index: Optional fixed ``sku_id -> int`` mapping; built from ``samples`` otherwise.

    Items are ``(image, label, index)``; ``index`` is the position in ``samples``.
    """

    def __init__(
        self,
        samples: Sequence[Sample],
        transform: Callable,
        label_index: dict[str, int] | None = None,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        if label_index is None:
            label_index = {sku: i for i, sku in enumerate(sorted({s.sku_id for s in samples}))}
        self.label_index = label_index
        self.labels = [label_index[s.sku_id] for s in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        sample = self.samples[index]
        return self.transform(load_rgb(sample.path)), self.labels[index], index
