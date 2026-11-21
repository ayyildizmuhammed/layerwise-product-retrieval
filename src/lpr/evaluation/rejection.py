"""Unknown-product rejection.

A nearest-neighbour retriever always returns an SKU, so a query is accepted only when its top-1
SKU score ``s1`` reaches a threshold ``tau``; otherwise it is rejected as an unknown product.

Threshold selection uses validation queries only and is then frozen:

* **score rule** (internal benchmark): for each representation, ``tau`` is the lowest top-1
  score threshold whose false accept rate (FAR) on the validation-unknown queries does not exceed
  the target (5 %). Validation-known queries are not used.
* **score-and-margin rule** (RP2K): a pair ``(tau_s, tau_m)`` from a fixed grid; a query is
  accepted when ``s1 >= tau_s`` and ``s1 - s2 >= tau_m``. Among the grid points whose
  validation-unknown FAR meets the target, the one that accepts the most correct
  validation-known queries is kept (ties: higher coverage, then lower thresholds).

On the test set the module reports, per unknown segment, the FAR together with *known coverage*
(share of known test queries accepted) and *accepted accuracy* (R@1 among the accepted ones).
FAR is an operating point that depends on the threshold and on the composition of the unknown
set, not a fixed property of a model.

The margin ``s1 - s2`` is a second confidence signal; :func:`auroc` measures how well a signal
separates correct from incorrect known predictions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

#: Score and margin grid of the score-and-margin rule.
SCORE_GRID: tuple[float, ...] = (
    0.20, 0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
)
MARGIN_GRID: tuple[float, ...] = (0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20)


@dataclass
class Threshold:
    """A frozen operating point. ``margin`` is 0 for the score rule."""

    score: float
    margin: float = 0.0

    def accepts(self, s1: np.ndarray, s2: np.ndarray | None = None) -> np.ndarray:
        accepted = np.asarray(s1) >= self.score
        if self.margin > 0.0:
            if s2 is None:
                raise ValueError("The margin rule needs the second-highest scores s2")
            accepted &= (np.asarray(s1) - np.asarray(s2)) >= self.margin
        return accepted


@dataclass
class OperatingPoint:
    """Rejection results of one threshold on one known set and one unknown set, in percent."""

    far: float
    known_coverage: float
    accepted_accuracy: float
    num_unknown: int
    num_known: int

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def far(
    unknown_s1: np.ndarray, threshold: Threshold, unknown_s2: np.ndarray | None = None
) -> float:
    """False accept rate: percentage of unknown queries accepted."""
    if len(unknown_s1) == 0:
        return 0.0
    return float(100.0 * threshold.accepts(unknown_s1, unknown_s2).mean())


def select_score_threshold(val_unknown_s1: np.ndarray, target_far: float = 5.0) -> Threshold:
    """Lowest top-1 score threshold whose validation-unknown FAR is at most ``target_far`` %.

    Candidates are the observed validation-unknown scores and a value just above the maximum
    (which rejects every validation unknown). The lowest admissible candidate accepts the most
    known queries.
    """
    scores = np.sort(np.asarray(val_unknown_s1, dtype=np.float64))
    if scores.size == 0:
        raise ValueError("No validation-unknown queries")
    candidates = np.append(np.unique(scores), np.nextafter(scores[-1], np.inf))
    # FAR(t) = share of scores >= t; searchsorted gives the number of scores below t.
    rates = 100.0 * (scores.size - np.searchsorted(scores, candidates, side="left")) / scores.size
    admissible = candidates[rates <= target_far + 1e-12]
    return Threshold(score=float(admissible.min()))


def select_score_margin_threshold(
    val_known_s1: np.ndarray,
    val_known_s2: np.ndarray,
    val_known_correct: np.ndarray,
    val_unknown_s1: np.ndarray,
    val_unknown_s2: np.ndarray,
    target_far: float = 5.0,
    score_grid: Sequence[float] = SCORE_GRID,
    margin_grid: Sequence[float] = MARGIN_GRID,
) -> Threshold:
    """Grid point meeting the FAR target that accepts the most correct validation-known queries.

    Ties are broken by higher known coverage, then lower score and lower margin thresholds.
    """
    correct = np.asarray(val_known_correct, dtype=bool)
    best_key, best = None, None
    for score in score_grid:
        for margin in margin_grid:
            point = Threshold(score=score, margin=margin)
            if far(val_unknown_s1, point, val_unknown_s2) > target_far + 1e-12:
                continue
            accepted = point.accepts(val_known_s1, val_known_s2)
            key = ((accepted & correct).sum(), accepted.sum(), -score, -margin)
            if best_key is None or key > best_key:
                best_key, best = key, point
    if best is None:
        raise ValueError("No grid point meets the FAR target")
    return best


def operating_point(
    threshold: Threshold,
    known_s1: np.ndarray,
    known_correct: np.ndarray,
    unknown_s1: np.ndarray,
    known_s2: np.ndarray | None = None,
    unknown_s2: np.ndarray | None = None,
) -> OperatingPoint:
    """FAR on the unknown queries and coverage and accepted accuracy on the known queries."""
    accepted = threshold.accepts(known_s1, known_s2)
    correct = np.asarray(known_correct, dtype=bool)
    coverage = 100.0 * accepted.mean() if accepted.size else 0.0
    accuracy = 100.0 * correct[accepted].mean() if accepted.any() else 0.0
    return OperatingPoint(
        far=far(unknown_s1, threshold, unknown_s2),
        known_coverage=float(coverage),
        accepted_accuracy=float(accuracy),
        num_unknown=int(len(unknown_s1)),
        num_known=int(len(known_s1)),
    )


def auroc(signal: np.ndarray, positive: np.ndarray) -> float:
    """Area under the ROC curve of ``signal`` for separating positives (ties averaged), in %.

    Used with the top-1 score or the top-1/top-2 margin as ``signal`` and the correctness of
    known-query predictions as ``positive``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    positive = np.asarray(positive, dtype=bool)
    num_pos, num_neg = positive.sum(), (~positive).sum()
    if num_pos == 0 or num_neg == 0:
        return float("nan")
    order = signal.argsort(kind="mergesort")
    ranks = np.empty_like(signal)
    sorted_signal = signal[order]
    # Average ranks over ties.
    boundaries = np.flatnonzero(np.diff(sorted_signal)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [signal.size]))
    for start, end in zip(starts, ends, strict=True):
        ranks[order[start:end]] = (start + end + 1) / 2.0
    rank_sum = ranks[positive].sum()
    return float(100.0 * (rank_sum - num_pos * (num_pos + 1) / 2.0) / (num_pos * num_neg))
