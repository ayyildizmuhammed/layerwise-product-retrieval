"""Embedding a list of samples with one or more representations."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lpr.data.datasets import ImageDataset
from lpr.data.manifest import Sample


def default_device() -> torch.device:
    """CUDA, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def embed_samples(
    samples: Sequence[Sample],
    transform: Callable,
    encode: Callable[[torch.Tensor], dict[str, torch.Tensor]],
    device: torch.device | None = None,
    batch_size: int = 64,
    num_workers: int = 4,
    description: str = "embedding",
) -> dict[str, torch.Tensor]:
    """Embed every sample; returns ``{representation name: [len(samples), dim]}`` on the CPU.

    Args:
        samples: Images to embed, in order.
        transform: PIL image to tensor (the representation's preprocessing).
        encode: Batch to named, L2-normalised embeddings.
        device: Device for the forward pass.
        batch_size: Images per forward pass.
        num_workers: DataLoader workers.
    """
    device = device or default_device()
    loader = DataLoader(
        ImageDataset(samples, transform, label_index={s.sku_id: 0 for s in samples}),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    parts: dict[str, list[torch.Tensor]] = defaultdict(list)
    for images, _, _ in tqdm(loader, desc=description, leave=False):
        for name, vectors in encode(images.to(device, non_blocking=True)).items():
            parts[name].append(vectors.float().cpu())
    return {name: torch.cat(chunks) for name, chunks in parts.items()}
