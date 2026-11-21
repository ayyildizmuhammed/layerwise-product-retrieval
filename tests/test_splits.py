"""Split rules on a synthetic catalogue (no images needed)."""

import random

from lpr.data.manifest import Sample, check_disjoint_groups
from lpr.data.splits import SplitConfig, build_evaluation_split


def _catalogue(num_families=12, skus_per_family=4, photos=40, crops_per_photo=3, seed=0):
    rng = random.Random(seed)
    skus = [(f"f{f}_s{s}", f"f{f}") for f in range(num_families) for s in range(skus_per_family)]
    samples = []
    for photo in range(photos * 4):
        family = rng.randrange(num_families)
        siblings = [sku for sku in skus if sku[1] == f"f{family}"]
        for crop in range(crops_per_photo):
            sku, fam = rng.choice(siblings)
            samples.append(Sample(f"p{photo}_c{crop}.jpg", sku, fam, f"photo{photo}"))
    return samples


def test_split_sizes_segments_and_disjoint_groups():
    samples = _catalogue()
    config = SplitConfig(num_val_unknown=3, num_test_unknown=10,
                         num_cross_family_test_unknown=4, seed=1)
    result = build_evaluation_split(samples, config)
    roles = result.roles
    check_disjoint_groups(roles)
    summary = result.summary()
    assert summary["skus"]["val_unknown"] == 3
    assert summary["skus"]["test_unknown"] == 10
    assert summary["test_unknown_cross_family_skus"] == 4
    assert summary["test_unknown_same_family_skus"] == 6
    catalogue = {s.sku_id for s in roles["gallery"]}
    assert catalogue == {s.sku_id for s in roles["test_known"]}
    unknown = {s.sku_id for s in roles["val_unknown"] + roles["test_unknown"]}
    assert not catalogue & unknown
    assert len(catalogue) + len(unknown) + len(result.unused_skus) == 48


def test_duplicate_queries_are_removed():
    samples = _catalogue()
    config = SplitConfig(num_val_unknown=2, num_test_unknown=4,
                         num_cross_family_test_unknown=None, seed=3)
    # Every crop of an SKU has the same content: all its queries duplicate its gallery crops.
    result = build_evaluation_split(samples, config, content_hash=lambda s: s.sku_id)
    assert result.removed_duplicates > 0
    gallery = {s.sku_id for s in result.roles["gallery"]}
    assert not gallery & {s.sku_id for s in result.roles["test_known"]}


def test_split_is_deterministic():
    samples = _catalogue()
    config = SplitConfig(num_val_unknown=2, num_test_unknown=6,
                         num_cross_family_test_unknown=None, seed=7)
    first = build_evaluation_split(samples, config).roles
    second = build_evaluation_split(samples, config).roles
    assert first == second
