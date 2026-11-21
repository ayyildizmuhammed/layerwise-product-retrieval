"""Iterative hard-negative fine-tuning of one window.

Training proceeds in mining cycles. Each cycle first mines hard and semi-hard neighbours with the
current model and then trains for a fixed number of epochs with PK batches that include the
mined examples. Before the first cycle the head is still untrained, so the first mining uses the
pretrained stop-layer readout; later cycles use the projected embedding of the model trained so
far. Validation R@1 of the projected embedding
is computed after epoch 1 and every second epoch; the best checkpoint is kept, and training stops
after ``patience`` evaluations without improvement.

Optimisation: AdamW with learning rate 1e-5 for the window (blocks, post-LN, attention pool) and
1e-3 for the head, weight decay 1e-4, and a per-step cosine schedule with 200 linear warm-up
steps.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from lpr.data.datasets import ImageDataset
from lpr.data.manifest import Sample
from lpr.data.sampler import PKSampler
from lpr.data.transforms import pe_eval_transform, pe_train_transform
from lpr.models.window import WindowModel
from lpr.retrieval.embed import default_device
from lpr.retrieval.search import rank_queries
from lpr.training.loss import batch_hard_triplet_loss
from lpr.training.mining import MiningConfig, mine_neighbours

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Fine-tuning settings."""

    stop_layer: int = 11
    window_size: int = 3
    P: int = 16
    K: int = 4
    margin: float = 0.3
    lr_encoder: float = 1e-5
    lr_head: float = 1e-3
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    cycles: int = 5
    epochs_per_cycle: int = 8
    eval_every: int = 2
    patience: int = 4
    num_positives: int = 2
    num_negatives: int = 4
    positive_buckets: tuple[str, ...] = ("hard", "semi_hard")
    negative_buckets: tuple[str, ...] = ("semi_hard", "hard")
    mining: MiningConfig = field(default_factory=MiningConfig)
    val_gallery_fraction: float = 0.2
    seed: int = 54
    num_workers: int = 4
    eval_batch_size: int = 128
    amp: bool = True  # CUDA only

    @classmethod
    def from_dict(cls, values: dict) -> TrainConfig:
        values = dict(values)
        if isinstance(values.get("mining"), dict):
            values["mining"] = MiningConfig(**values["mining"])
        for key in ("positive_buckets", "negative_buckets"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_train_val(
    samples: list[Sample], val_ratio: float = 0.1, seed: int = 54
) -> tuple[list[Sample], list[Sample]]:
    """Per-SKU image split of a fine-tuning collection.

    Each SKU contributes ``round(n * val_ratio)`` (at least one) of its images to validation, but
    always keeps at least one image for training.
    """
    by_sku: dict[str, list[Sample]] = {}
    for sample in samples:
        by_sku.setdefault(sample.sku_id, []).append(sample)
    rng = random.Random(seed)
    train, val = [], []
    for sku in sorted(by_sku):
        items = list(by_sku[sku])
        rng.shuffle(items)
        n_val = min(max(1, round(len(items) * val_ratio)), len(items) - 1)
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    return train, val


class WarmupCosine:
    """Linear warm-up for ``warmup`` steps, then cosine decay to zero at ``total`` steps."""

    def __init__(self, optimizer: torch.optim.Optimizer, warmup: int, total: int) -> None:
        self.optimizer = optimizer
        self.warmup, self.total = warmup, max(total, warmup + 1)
        self.base = [group["lr"] for group in optimizer.param_groups]
        self.step_count = 0
        self._apply()

    def factor(self) -> float:
        if self.step_count < self.warmup:
            return (self.step_count + 1) / self.warmup
        progress = min(1.0, (self.step_count - self.warmup) / (self.total - self.warmup))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _apply(self) -> None:
        for group, base in zip(self.optimizer.param_groups, self.base, strict=True):
            group["lr"] = base * self.factor()

    def step(self) -> None:
        self.step_count += 1
        self._apply()


class Trainer:
    """Fine-tunes one :class:`WindowModel`.

    Args:
        model: The window model.
        train_samples: Fine-tuning images.
        val_samples: Validation images (SKUs may overlap with training; images do not).
        output_dir: Where checkpoints and the training log are written.
        config: Training settings.
        device: Training device.
    """

    def __init__(
        self,
        model: WindowModel,
        train_samples: list[Sample],
        val_samples: list[Sample],
        output_dir: str | Path,
        config: TrainConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or TrainConfig()
        self.device = device or default_device()
        self.model = model.to(self.device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_samples = train_samples
        self.val_samples = val_samples
        skus = sorted({s.sku_id for s in train_samples})
        self.label_index = {sku: i for i, sku in enumerate(skus)}
        self.train_set = ImageDataset(train_samples, pe_train_transform(), self.label_index)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": model.encoder_parameters(), "lr": self.config.lr_encoder},
                {"params": model.head_parameters(), "lr": self.config.lr_head},
            ],
            weight_decay=self.config.weight_decay,
        )
        plain_batches = max(1, len(train_samples) // (self.config.P * self.config.K))
        total = plain_batches * self.config.epochs_per_cycle * self.config.cycles
        self.scheduler = WarmupCosine(self.optimizer, self.config.warmup_steps, total)
        use_amp = self.config.amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        self.use_amp = use_amp
        self.history: list[dict] = []
        if self.device.type == "mps":
            log.warning(
                "Training on Apple MPS: batches with mined examples vary in size, which makes "
                "MPS recompile its kernels and can slow training down sharply. CUDA or CPU is "
                "recommended."
            )

    # ------------------------------------------------------------------ helpers
    @torch.no_grad()
    def _embed(self, samples: list[Sample], readout: bool = False) -> torch.Tensor:
        """Projected embeddings ``z`` of ``samples``, or readouts ``r`` with ``readout=True``."""
        self.model.eval()
        transform = pe_eval_transform()
        dataset = ImageDataset(samples, transform, {s.sku_id: 0 for s in samples})
        loader = DataLoader(
            dataset, batch_size=self.config.eval_batch_size, num_workers=self.config.num_workers
        )
        chunks = []
        for images, _, _ in loader:
            output = self.model(images.to(self.device))
            chunks.append((output.readout if readout else output.projected).float().cpu())
        return torch.cat(chunks)

    def validate(self) -> float:
        """Validation R@1 (%) of the projected embedding.

        Within each SKU, the first ``val_gallery_fraction`` of the validation images (at least
        one) form the gallery and the rest are queries; an SKU without a remaining image uses
        its last image as a query as well.
        """
        embeddings = self._embed(self.val_samples)
        by_sku: dict[str, list[int]] = {}
        for index, sample in enumerate(self.val_samples):
            by_sku.setdefault(sample.sku_id, []).append(index)
        gallery, queries = [], []
        for indices in by_sku.values():
            cut = max(1, int(len(indices) * self.config.val_gallery_fraction))
            gallery.extend(indices[:cut])
            queries.extend(indices[cut:] or indices[-1:])
        labels = [s.sku_id for s in self.val_samples]
        ranking = rank_queries(
            embeddings[queries], embeddings[gallery], [labels[i] for i in gallery]
        )
        predicted = ranking.ranked_labels(1)[:, 0]
        truth = np.asarray([labels[i] for i in queries], dtype=object)
        return float(100.0 * np.mean(predicted == truth))

    def _mine(self, readout: bool = False) -> dict:
        embeddings = self._embed(self.train_samples, readout=readout)
        return mine_neighbours(embeddings, self.train_set.labels, self.config.mining)

    def _save(self, name: str, epoch: int, metric: float) -> None:
        torch.save(
            {
                "trainable_state_dict": self.model.trainable_state_dict(),
                "stop_layer": self.model.stop_layer,
                "window_size": self.model.window_size,
                "epoch": epoch,
                "val_recall_at_1": metric,
                "config": asdict(self.config),
            },
            self.output_dir / name,
        )

    # ------------------------------------------------------------------ training
    def _train_epoch(self, sampler: PKSampler) -> float:
        self.model.train()
        loader = DataLoader(
            self.train_set, batch_sampler=sampler, num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
        )
        losses = []
        for images, labels, _ in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            autocast = torch.autocast("cuda") if self.use_amp else contextlib.nullcontext()
            with autocast:
                loss = batch_hard_triplet_loss(
                    self.model(images).projected.float(), labels, self.config.margin
                )
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            losses.append(loss.item())
        return float(np.mean(losses)) if losses else float("nan")

    def fit(self) -> dict:
        """Run all mining cycles; returns a summary with the best validation R@1."""
        config = self.config
        seed_everything(config.seed)
        best, stale, epoch = -1.0, 0, 0
        stopped = False
        for cycle in range(1, config.cycles + 1):
            # The head is untrained before cycle 1, so the first mining uses the readout.
            mined = self._mine(readout=cycle == 1)
            sampler = PKSampler(
                self.train_set.labels, P=config.P, K=config.K, mined=mined,
                num_positives=config.num_positives, num_negatives=config.num_negatives,
                positive_buckets=config.positive_buckets,
                negative_buckets=config.negative_buckets,
                seed=config.seed + cycle,
            )
            for _ in range(config.epochs_per_cycle):
                epoch += 1
                loss = self._train_epoch(sampler)
                record = {"cycle": cycle, "epoch": epoch, "loss": loss,
                          "lr_encoder": self.optimizer.param_groups[0]["lr"]}
                if epoch == 1 or epoch % config.eval_every == 0:
                    metric = self.validate()
                    record["val_recall_at_1"] = metric
                    if metric > best:
                        best, stale = metric, 0
                        self._save("best.pt", epoch, metric)
                    else:
                        stale += 1
                self._save("last.pt", epoch, record.get("val_recall_at_1", float("nan")))
                self.history.append(record)
                log.info("cycle %d epoch %d: %s", cycle, epoch, record)
                (self.output_dir / "history.json").write_text(json.dumps(self.history, indent=2))
                if stale >= config.patience:
                    stopped = True
                    break
            if stopped:
                break
        summary = {"best_val_recall_at_1": best, "epochs": epoch, "early_stopped": stopped}
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def load_window(checkpoint_path: str | Path, encoder) -> WindowModel:
    """Rebuild a :class:`WindowModel` from a checkpoint written by :class:`Trainer`."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = WindowModel(encoder, stop_layer=state["stop_layer"], window_size=state["window_size"])
    model.load_trainable_state_dict(state["trainable_state_dict"])
    return model.eval()
