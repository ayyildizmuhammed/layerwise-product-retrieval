"""Final-output baselines: CLIP B/16 and SigLIP2 B/16 through ``open_clip``.

The raw final embeddings of CLIP ViT-B/16 (Radford et al., 2021) and SigLIP2 ViT-B/16
(Tschannen et al., 2025) serve as baselines. Like every representation, each crop is padded to
a square and resized to the encoder's input size, then normalised with the encoder's own image
statistics. Install the optional dependency with ``pip install -e ".[baselines]"``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from lpr.data.transforms import ImageTransform
from lpr.models.representations import Representation

#: name -> (open_clip architecture, pretrained tag)
BASELINES: dict[str, tuple[str, str]] = {
    "clip_b16": ("ViT-B-16", "openai"),
    "siglip2_b16": ("ViT-B-16-SigLIP2", "webli"),
}


class OpenClipBaseline(Representation):
    """Raw final image embedding of an ``open_clip`` model."""

    def __init__(self, name: str) -> None:
        super().__init__()
        if name not in BASELINES:
            raise ValueError(f"Unknown baseline {name!r}; choose from {sorted(BASELINES)}")
        try:
            import open_clip
        except ImportError as error:  # pragma: no cover - optional dependency
            raise ImportError('Baselines need open_clip: pip install -e ".[baselines]"') from error
        architecture, pretrained = BASELINES[name]
        self.name = name
        # SigLIP2's attention-pooling head fails on Apple MPS in current PyTorch releases.
        self.cpu_on_mps = name == "siglip2_b16"
        self.model, _, reference = open_clip.create_model_and_transforms(
            architecture, pretrained=pretrained
        )
        self.model.eval()
        self.transform = _padded_transform(self.model, reference)

    @property
    def names(self) -> list[str]:
        return [self.name]

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {self.name: F.normalize(self.model.encode_image(images), dim=-1)}


def _padded_transform(model, reference) -> ImageTransform:
    """Pad-and-resize transform with the encoder's input size and image statistics."""
    from torchvision.transforms import Normalize

    normalize = next(t for t in reference.transforms if isinstance(t, Normalize))
    size = model.visual.image_size
    size = size[0] if isinstance(size, (tuple, list)) else size
    return ImageTransform(size=int(size), mean=tuple(normalize.mean), std=tuple(normalize.std))
