"""Perception Encoder (PE) vision tower with layer-wise readouts.

The encoder is ``PE-Core-B16-224``, a 12-block vision transformer with 16x16 patches at
224x224 resolution. This module wraps the official implementation from
``facebookresearch/perception_models`` and exposes three things:

* :meth:`PerceptionEncoder.tokens` - the token sequence after block ``l`` followed by the
  pretrained post-layer normalisation (``H_l``);
* :meth:`PerceptionEncoder.readout` - a pooled readout ``p_l`` of those tokens, either with the
  pretrained attention pool or with the mean of the patch tokens;
* :meth:`PerceptionEncoder.final_embedding` - the encoder's standard 1024-dimensional output
  (the *raw final embedding*), i.e. the attention-pooled layer-12 feature after PE's projection.

Blocks above the requested layer are not executed.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

Pooling = Literal["attn", "mean"]
POOLINGS: tuple[str, ...] = ("attn", "mean")
DEFAULT_MODEL = "PE-Core-B16-224"


def _import_pe():
    try:
        from core.vision_encoder import pe  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The Perception Encoder code is not installed. Install it with:\n"
            '  pip install --no-deps "git+https://github.com/facebookresearch/perception_models'
            '@3e352cca660658d4b5c90f42a7808b11469e4c66"'
        ) from error
    return pe


class PerceptionEncoder(nn.Module):
    """PE vision tower with access to every block.

    Args:
        model_name: PE configuration name, ``PE-Core-B16-224`` by default.
        pretrained: Load the pretrained weights (downloaded from Hugging Face on first use).
        checkpoint_path: Optional local checkpoint instead of the Hugging Face download.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        pretrained: bool = True,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__()
        pe = _import_pe()
        self.model_name = model_name
        self.visual = pe.VisionTransformer.from_config(
            model_name, pretrained=pretrained, checkpoint_path=checkpoint_path
        )
        if self.visual.attn_pool is None:
            raise ValueError(f"{model_name} has no attention pool; layer-wise readouts need one.")

    # ------------------------------------------------------------------ properties
    @property
    def num_layers(self) -> int:
        """Number of transformer blocks (12 for PE-Core-B16-224)."""
        return len(self.visual.transformer.resblocks)

    @property
    def width(self) -> int:
        """Token width, which is also the dimension of every readout (768)."""
        return int(self.visual.width)

    @property
    def output_dim(self) -> int:
        """Dimension of the raw final embedding (1024)."""
        return int(self.visual.output_dim)

    @property
    def blocks(self) -> nn.ModuleList:
        return self.visual.transformer.resblocks

    @property
    def has_cls_token(self) -> bool:
        return bool(self.visual.use_cls_token)

    # ------------------------------------------------------------------ forward passes
    def _check_layer(self, layer: int) -> int:
        if not 1 <= layer <= self.num_layers:
            raise ValueError(f"layer must be in [1, {self.num_layers}], got {layer}")
        return layer

    def tokens(self, images: torch.Tensor, layer: int) -> torch.Tensor:
        """Tokens after block ``layer`` (1-based) and the pretrained post-layer normalisation.

        Returns a tensor of shape ``[batch, tokens, width]``; the class token, when present, is
        the first token.
        """
        layer = self._check_layer(layer)
        # PE counts layers from the end: -1 is the last block, -num_layers the first.
        layer_idx = layer - self.num_layers - 1
        return self.visual.forward_features(images, norm=True, layer_idx=layer_idx)

    def pool(self, tokens: torch.Tensor, pooling: Pooling) -> torch.Tensor:
        """Pool a token sequence into one vector per image.

        ``attn`` applies the pretrained attention pool (the same pool at every depth); ``mean``
        averages the patch tokens, excluding the class token.
        """
        if pooling == "attn":
            return self.visual.attn_pool(tokens).squeeze(1)
        if pooling == "mean":
            patches = tokens[:, 1:] if self.has_cls_token else tokens
            return patches.mean(dim=1)
        raise ValueError(f"pooling must be one of {POOLINGS}, got {pooling!r}")

    def readout(self, images: torch.Tensor, layer: int, pooling: Pooling = "attn") -> torch.Tensor:
        """Unnormalised readout ``p_l`` of block ``layer`` with the given pooling (768-D)."""
        return self.pool(self.tokens(images, layer), pooling)

    def final_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """Unnormalised raw final embedding: attention-pooled layer 12 after PE's projection."""
        return self.visual(images)

    def layer_readouts(
        self,
        images: torch.Tensor,
        layers: list[int] | None = None,
        poolings: tuple[str, ...] = POOLINGS,
    ) -> dict[tuple[int, str], torch.Tensor]:
        """Readouts of several layers from one forward pass.

        Runs the encoder once up to the deepest requested layer, captures every requested
        block's output, applies the pretrained post-layer normalisation and pools it.
        Returns ``{(layer, pooling): [batch, width]}``, unnormalised.
        """
        layers = sorted(set(layers or range(1, self.num_layers + 1)))
        for layer in layers:
            self._check_layer(layer)
        captured: dict[int, torch.Tensor] = {}
        handles = [
            self.blocks[layer - 1].register_forward_hook(
                lambda _module, _inputs, output, layer=layer: captured.__setitem__(layer, output)
            )
            for layer in layers
        ]
        try:
            deepest_idx = layers[-1] - self.num_layers - 1
            self.visual.forward_features(images, norm=False, layer_idx=deepest_idx)
        finally:
            for handle in handles:
                handle.remove()
        readouts = {}
        for layer in layers:
            tokens = self.visual.ln_post(captured[layer])
            for pooling in poolings:
                readouts[(layer, pooling)] = self.pool(tokens, pooling)  # type: ignore[arg-type]
        return readouts
