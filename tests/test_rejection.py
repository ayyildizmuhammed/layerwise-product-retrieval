import numpy as np
import pytest

from lpr.evaluation.rejection import (
    Threshold,
    auroc,
    far,
    operating_point,
    select_score_margin_threshold,
    select_score_threshold,
)


def test_score_threshold_is_lowest_meeting_target():
    val_unknown = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # FAR <= 20 % allows at most two accepted unknowns: threshold 0.9.
    assert select_score_threshold(val_unknown, target_far=20.0).score == pytest.approx(0.9)
    # A 0 % target needs a value above the maximum.
    assert select_score_threshold(val_unknown, target_far=0.0).score > 1.0


def test_far_and_operating_point():
    threshold = Threshold(score=0.5)
    assert far(np.array([0.4, 0.6, 0.7, 0.2]), threshold) == 50.0
    point = operating_point(
        threshold,
        known_s1=np.array([0.9, 0.8, 0.3, 0.6]),
        known_correct=np.array([1, 0, 1, 1]),
        unknown_s1=np.array([0.4, 0.6]),
    )
    assert point.known_coverage == 75.0
    assert point.accepted_accuracy == pytest.approx(200 / 3)
    assert point.far == 50.0


def test_margin_rule_prefers_most_correct_accepted():
    known_s1 = np.array([0.9, 0.9, 0.9, 0.9])
    known_s2 = np.array([0.5, 0.85, 0.5, 0.88])
    correct = np.array([1, 0, 1, 0])
    unknown_s1 = np.array([0.95, 0.95])
    unknown_s2 = np.array([0.94, 0.93])  # small margins
    threshold = select_score_margin_threshold(
        known_s1, known_s2, correct, unknown_s1, unknown_s2, target_far=0.0,
        score_grid=(0.5, 0.95), margin_grid=(0.0, 0.05),
    )
    assert threshold == Threshold(score=0.5, margin=0.05)


def test_auroc():
    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 100.0
    assert auroc(np.array([0.5, 0.5]), np.array([0, 1])) == 50.0
