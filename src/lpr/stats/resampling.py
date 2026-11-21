"""Group-level uncertainty and paired tests.

Crops cut from one source photograph are dependent, so the source group is the resampling unit:

* :func:`clustered_bootstrap_ci` - percentile confidence intervals from source-group clustered
  bootstrap samples (10,000 by default). Groups are drawn with replacement and each
  representation's accuracy is the ratio of correct queries to queries over the drawn groups.
  Both representations use the same draws, so the interval of their difference is paired.
* :func:`paired_randomisation_test` - a source-group paired randomisation (sign-flip) test of
  ``H0: equal accuracy``. Each group's summed difference ``sum(a_correct - b_correct)`` gets an
  independent random sign; the two-sided p-value is the share of sign patterns whose absolute
  total is at least the observed one, ``p = (extreme + 1) / (repeats + 1)``.

When no grouping exists (RP2K query images, Products-6K query SKUs, unknown SKUs for FAR), pass
the appropriate unit as ``groups``: one group per image, per SKU, and so on.

These intervals hold the trained model, the gallery and the threshold fixed; they describe query
sampling variation, not variation across training runs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class PairedResult:
    """Accuracies (%), their difference ``a - b`` and the 95 % CIs and p-value."""

    accuracy_a: float
    accuracy_b: float
    difference: float
    ci_a: tuple[float, float]
    ci_b: tuple[float, float]
    ci_difference: tuple[float, float]
    p_value: float | None = None


def _group_sums(values: np.ndarray, groups: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Per-group sums of ``values`` (rows) and group sizes."""
    _, inverse = np.unique(np.asarray(groups, dtype=object).astype(str), return_inverse=True)
    num_groups = inverse.max() + 1
    sizes = np.bincount(inverse, minlength=num_groups).astype(np.float64)
    sums = np.stack(
        [np.bincount(inverse, weights=column, minlength=num_groups) for column in values.T],
        axis=1,
    )
    return sums, sizes


def clustered_bootstrap_ci(
    correct_a: np.ndarray,
    groups: Sequence[str],
    correct_b: np.ndarray | None = None,
    repeats: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> PairedResult:
    """Source-group clustered bootstrap of one accuracy, or of two paired accuracies.

    Args:
        correct_a: Per-query correctness (0/1) of representation A.
        groups: Source group of every query.
        correct_b: Optional per-query correctness of representation B on the same queries.
        repeats: Bootstrap samples.
        confidence: Interval coverage.
        seed: Random seed.
    """
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64) if correct_b is not None else a
    sums, sizes = _group_sums(np.stack([a, b], axis=1), groups)
    rng = np.random.default_rng(seed)
    num_groups = len(sizes)
    uniform = np.full(num_groups, 1.0 / num_groups)
    acc_a, acc_b = np.empty(repeats), np.empty(repeats)
    for start in range(0, repeats, 256):  # chunks bound the memory of the count matrix
        stop = min(start + 256, repeats)
        # How often each group is drawn when num_groups groups are drawn with replacement.
        counts = rng.multinomial(num_groups, uniform, size=stop - start).astype(np.float64)
        totals = counts @ sizes
        acc_a[start:stop] = 100.0 * (counts @ sums[:, 0]) / totals
        acc_b[start:stop] = 100.0 * (counts @ sums[:, 1]) / totals
    low, high = 50.0 * (1 - confidence), 50.0 * (1 + confidence)

    def interval(x: np.ndarray) -> tuple[float, float]:
        return float(np.percentile(x, low)), float(np.percentile(x, high))

    return PairedResult(
        accuracy_a=float(100.0 * a.mean()),
        accuracy_b=float(100.0 * b.mean()),
        difference=float(100.0 * (a.mean() - b.mean())),
        ci_a=interval(acc_a),
        ci_b=interval(acc_b),
        ci_difference=interval(acc_a - acc_b),
    )


def clustered_bootstrap_rate_ci(
    events: np.ndarray,
    groups: Sequence[str],
    repeats: int = 10_000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile interval (%) of a rate, resampling groups with replacement.

    Used for the false accept rate of unknown queries, with the unknown SKU as the group.
    """
    events = np.asarray(events, dtype=np.float64)
    if events.size == 0:
        return float("nan"), float("nan")
    result = clustered_bootstrap_ci(events, groups, repeats=repeats, confidence=confidence,
                                    seed=seed)
    return result.ci_a


def paired_randomisation_test(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    groups: Sequence[str],
    repeats: int = 10_000,
    seed: int = 42,
) -> float:
    """Two-sided p-value of the source-group paired sign-flip test."""
    diff = np.asarray(correct_a, dtype=np.float64) - np.asarray(correct_b, dtype=np.float64)
    sums, _ = _group_sums(diff[:, None], groups)
    group_diff = sums[:, 0]
    observed = abs(group_diff.sum())
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, repeats, 256):
        signs = rng.choice((-1.0, 1.0), size=(min(256, repeats - start), group_diff.size))
        extreme += int((np.abs(signs @ group_diff) + 1e-12 >= observed).sum())
    return (extreme + 1) / (repeats + 1)


def compare(
    correct_a: np.ndarray,
    correct_b: np.ndarray,
    groups: Sequence[str],
    repeats: int = 10_000,
    seed: int = 42,
) -> PairedResult:
    """Paired bootstrap intervals and randomisation p-value of two representations."""
    result = clustered_bootstrap_ci(correct_a, groups, correct_b, repeats=repeats, seed=seed)
    result.p_value = paired_randomisation_test(correct_a, correct_b, groups, repeats, seed)
    return result
