"""The compared representations, behind one interface.

A :class:`Representation` turns a batch of preprocessed images into embeddings and knows its own
preprocessing. One encoder pass may produce several representations (for example all 24 frozen
readouts), so ``encode`` returns a dictionary ``{name: [batch, dim]}``; every vector is
L2-normalised.

==============================  ==============================================  =========
Representation                  Class                                           Dim.
==============================  ==============================================  =========
PE raw final embedding          :class:`PERawFinal`                             1024
PE frozen readouts (layer l)    :class:`PEFrozenReadouts`                       768
PE trained readout ``r_l``      :class:`WindowRepresentations`                  768
PE projected embedding ``z_l``  :class:`WindowRepresentations`                  768
CLIP B/16, SigLIP2 B/16         :class:`lpr.models.baselines.OpenClipBaseline`  512 / 768
==============================  ==============================================  =========

All of them use the same input: the crop padded to a square and resized to 224x224
(:mod:`lpr.data.transforms`), normalised with the statistics of the respective encoder.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from lpr.data.transforms import pe_eval_transform
from lpr.models.pe import POOLINGS, PerceptionEncoder
from lpr.models.window import WindowModel


class Representation(nn.Module):
    """Base class: ``transform`` (PIL to tensor) and ``encode`` (batch to named embeddings)."""

    transform: Callable

    @property
    def names(self) -> list[str]:
        raise NotImplementedError

    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        raise NotImplementedError


def frozen_name(layer: int, pooling: str) -> str:
    """Name of a frozen readout, e.g. ``pe_frozen_l08_attn``."""
    return f"pe_frozen_l{layer:02d}_{pooling}"


class PERawFinal(Representation):
    """The encoder's standard output: attention-pooled layer 12 after PE's projection."""

    def __init__(self, encoder: PerceptionEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.transform = pe_eval_transform()

    @property
    def names(self) -> list[str]:
        return ["pe_raw_final"]

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"pe_raw_final": F.normalize(self.encoder.final_embedding(images), dim=-1)}


class PEFrozenReadouts(Representation):
    """Frozen readouts ``f_l`` of the pretrained encoder.

    Args:
        encoder: Pretrained PE.
        layers: Layers to read out (all twelve by default).
        poolings: ``attn`` and/or ``mean``.
    """

    def __init__(
        self,
        encoder: PerceptionEncoder,
        layers: Sequence[int] | None = None,
        poolings: Sequence[str] = POOLINGS,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.layers = list(layers or range(1, encoder.num_layers + 1))
        self.poolings = tuple(poolings)
        self.transform = pe_eval_transform()

    @property
    def names(self) -> list[str]:
        return [frozen_name(layer, pool) for layer in self.layers for pool in self.poolings]

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        readouts = self.encoder.layer_readouts(images, self.layers, self.poolings)
        return {frozen_name(layer, pool): F.normalize(vector, dim=-1)
                for (layer, pool), vector in readouts.items()}


class WindowRepresentations(Representation):
    """Trained readout ``r_l`` and projected embedding ``z_l`` of a fine-tuned window."""

    def __init__(self, model: WindowModel, name: str | None = None) -> None:
        super().__init__()
        self.model = model
        layers = model.window_layers
        self.prefix = name or f"pe_l{layers[0]}_l{layers[-1]}"
        self.transform = pe_eval_transform()

    @property
    def names(self) -> list[str]:
        return [f"{self.prefix}_trained_readout", f"{self.prefix}_projected"]

    @torch.no_grad()
    def encode(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.model(images)
        return {
            f"{self.prefix}_trained_readout": output.readout,
            f"{self.prefix}_projected": output.projected,
        }
