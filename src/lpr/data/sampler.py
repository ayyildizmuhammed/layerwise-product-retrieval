"""PK batch sampling with optional mined examples.

Every batch starts from ``P`` SKUs with ``K`` images each (``P = 16``, ``K = 4``). An SKU with
fewer than ``K`` images is sampled with replacement; otherwise its images are visited round-robin
in a shuffled order, so every image is seen before any repeats.

After a mining cycle, each of these anchors also brings up to ``num_positives`` mined positives
and up to ``num_negatives`` mined negatives. Candidates are taken bucket by bucket in a
configurable order; only the first ``2 * num_negatives`` negative candidates are scanned. A batch
is de-duplicated, so its size varies.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterator, Sequence

from torch.utils.data import Sampler

from lpr.training.mining import MinedExamples


class PKSampler(Sampler[list[int]]):
    """Batch sampler yielding lists of dataset indices.

    Args:
        labels: Integer label of every dataset item.
        P: SKUs per batch.
        K: Images per SKU.
        mined: Optional mined positives and negatives per dataset index.
        num_positives: Mined positives added per anchor.
        num_negatives: Mined negatives added per anchor.
        positive_buckets: Order in which positive buckets are used.
        negative_buckets: Order in which negative buckets are used.
        batches_per_epoch: Batches per epoch. Defaults to ``N // (P * K)`` without mining and
            ``N // (2 * P * K)`` with mining, whose batches are about twice as large.
        seed: Random seed.
    """

    def __init__(
        self,
        labels: Sequence[int],
        P: int = 16,
        K: int = 4,
        mined: dict[int, MinedExamples] | None = None,
        num_positives: int = 2,
        num_negatives: int = 4,
        positive_buckets: Sequence[str] = ("hard", "semi_hard"),
        negative_buckets: Sequence[str] = ("semi_hard", "hard"),
        batches_per_epoch: int | None = None,
        seed: int = 0,
    ) -> None:
        self.labels = list(labels)
        self.P, self.K = P, K
        self.mined = mined
        self.num_positives = num_positives
        self.num_negatives = num_negatives
        self.positive_buckets = tuple(positive_buckets)
        self.negative_buckets = tuple(negative_buckets)
        self.rng = random.Random(seed)

        by_label: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            by_label[label].append(index)
        for indices in by_label.values():
            self.rng.shuffle(indices)
        self.by_label = dict(by_label)
        if len(self.by_label) < P:
            raise ValueError(f"P={P} needs at least {P} SKUs, found {len(self.by_label)}")

        if batches_per_epoch is None:
            divisor = P * K * (2 if mined else 1)
            batches_per_epoch = max(1, len(self.labels) // divisor)
        self.batches_per_epoch = batches_per_epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def _take(self, label: int) -> list[int]:
        indices = self.by_label[label]
        if len(indices) < self.K:
            return [self.rng.choice(indices) for _ in range(self.K)]
        chosen = indices[: self.K]
        self.by_label[label] = indices[self.K :] + chosen
        return chosen

    def _mined_for(self, anchor: int, in_batch: set[int]) -> list[int]:
        examples = self.mined.get(anchor) if self.mined else None
        if examples is None:
            return []
        label = self.labels[anchor]
        added: list[int] = []

        positives = [i for bucket in self.positive_buckets for i in examples.positives[bucket]]
        self.rng.shuffle(positives)
        for index in positives:
            if len(added) >= self.num_positives:
                break
            if index not in in_batch and self.labels[index] == label:
                added.append(index)
                in_batch.add(index)

        negatives = [i for bucket in self.negative_buckets for i in examples.negatives[bucket]]
        count = 0
        for index in negatives[: 2 * self.num_negatives]:
            if count >= self.num_negatives:
                break
            if index not in in_batch and self.labels[index] != label:
                added.append(index)
                in_batch.add(index)
                count += 1
        return added

    def __iter__(self) -> Iterator[list[int]]:
        labels = list(self.by_label)
        for _ in range(self.batches_per_epoch):
            anchors: list[int] = []
            for label in self.rng.sample(labels, self.P):
                anchors.extend(self._take(label))
            batch = list(dict.fromkeys(anchors))
            in_batch = set(batch)
            if self.mined:
                for anchor in list(dict.fromkeys(anchors)):
                    batch.extend(self._mined_for(anchor, in_batch))
            yield batch
