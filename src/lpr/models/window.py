"""Depth-specific fine-tuning windows.

A window of ``window_size`` consecutive blocks ending at ``stop_layer`` is trained together with
the post-layer normalisation, the attention pool and a bias-free linear head
``W in R^{768 x 768}``. All other encoder weights are frozen, and blocks above the stop layer are
never executed. One forward pass yields two embeddings:

* the *trained readout* ``r_l = p~_l / ||p~_l||`` - the attention-pooled stop-layer feature, and
* the *projected embedding* ``z_l = W p~_l / ||W p~_l||`` - the same feature after the head.

The triplet loss is optimised on ``z_l``; comparing ``r_l`` with ``z_l`` separates the
information available at the trained layer from the effect of the learned projection.

The provided configs train three windows of three blocks: L3-5, L6-8 and L9-11.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from lpr.models.pe import PerceptionEncoder


@dataclass
class WindowOutput:
    """Both embeddings of one forward pass, L2-normalised."""

    readout: torch.Tensor  # r_l, [batch, 768]
    projected: torch.Tensor  # z_l, [batch, 768]


class WindowModel(nn.Module):
    """A PE encoder whose blocks ``stop_layer - window_size + 1 .. stop_layer`` are trainable.

    Args:
        encoder: The pretrained PE vision tower.
        stop_layer: Last block of the window and the layer that is read out (5, 8 or 11).
        window_size: Number of trainable blocks ending at ``stop_layer`` (3 by default).
        embed_dim: Output dimension of the linear head (768).
    """

    def __init__(
        self,
        encoder: PerceptionEncoder,
        stop_layer: int,
        window_size: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        if not 1 <= window_size <= stop_layer <= encoder.num_layers:
            raise ValueError(
                f"Need 1 <= window_size <= stop_layer <= {encoder.num_layers}, "
                f"got window_size={window_size}, stop_layer={stop_layer}"
            )
        self.encoder = encoder
        self.stop_layer = stop_layer
        self.window_size = window_size
        # PyTorch's default (Kaiming-uniform) initialisation, no bias.
        self.head = nn.Linear(encoder.width, embed_dim, bias=False)
        self._set_trainable()

    @property
    def window_layers(self) -> list[int]:
        """1-based indices of the trainable blocks."""
        return list(range(self.stop_layer - self.window_size + 1, self.stop_layer + 1))

    def _set_trainable(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        trainable = [self.encoder.blocks[layer - 1] for layer in self.window_layers]
        trainable += [self.encoder.visual.ln_post, self.encoder.visual.attn_pool]
        for module in trainable:
            for parameter in module.parameters():
                parameter.requires_grad = True

    def encoder_parameters(self) -> list[nn.Parameter]:
        """Trainable encoder parameters (window blocks, post-LN, attention pool)."""
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def pooled(self, images: torch.Tensor) -> torch.Tensor:
        """Unnormalised attention-pooled stop-layer feature ``p~_l``."""
        return self.encoder.readout(images, self.stop_layer, pooling="attn")

    def forward(self, images: torch.Tensor) -> WindowOutput:
        pooled = self.pooled(images)
        return WindowOutput(
            readout=F.normalize(pooled, dim=-1),
            projected=F.normalize(self.head(pooled), dim=-1),
        )

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Only the parameters that fine-tuning changes, for compact checkpoints."""
        names = {name for name, p in self.named_parameters() if p.requires_grad}
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if k in names}

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        missing = {name for name, p in self.named_parameters() if p.requires_grad} - set(state)
        if missing:
            raise KeyError(f"Checkpoint lacks trainable parameters: {sorted(missing)[:5]} ...")
        self.load_state_dict(state, strict=False)
