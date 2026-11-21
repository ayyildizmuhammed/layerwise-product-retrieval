import numpy as np

from lpr.stats.resampling import clustered_bootstrap_ci, compare, paired_randomisation_test


def test_identical_representations_are_not_different():
    rng = np.random.default_rng(0)
    correct = rng.integers(0, 2, 200)
    groups = [f"g{i // 4}" for i in range(200)]
    assert paired_randomisation_test(correct, correct, groups, repeats=999) == 1.0


def test_clear_difference_is_significant():
    a = np.ones(200)
    b = np.zeros(200)
    groups = [f"g{i // 4}" for i in range(200)]
    result = compare(a, b, groups, repeats=999)
    assert result.difference == 100.0
    assert result.p_value <= 0.002
    assert result.ci_difference == (100.0, 100.0)


def test_bootstrap_interval_contains_estimate():
    rng = np.random.default_rng(1)
    correct = rng.random(500) < 0.8
    groups = [f"g{i // 5}" for i in range(500)]
    result = clustered_bootstrap_ci(correct, groups, repeats=2000)
    low, high = result.ci_a
    assert low < result.accuracy_a < high
