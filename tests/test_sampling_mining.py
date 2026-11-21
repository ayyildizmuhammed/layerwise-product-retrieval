import torch

from lpr.data.manifest import Sample
from lpr.data.sampler import PKSampler
from lpr.evaluation.galleries import nested_gallery_indices
from lpr.training.mining import MiningConfig, mine_neighbours


def test_pk_batches_have_p_classes():
    labels = [i // 5 for i in range(100)]  # 20 SKUs with 5 images
    sampler = PKSampler(labels, P=4, K=3, seed=0)
    assert len(sampler) == 100 // 12
    for batch in sampler:
        assert len({labels[i] for i in batch}) == 4
        assert len(batch) == 12


def test_pk_samples_with_replacement_for_small_classes():
    labels = [0, 1, 1, 1, 1, 2, 2, 2, 2]
    batch = next(iter(PKSampler(labels, P=3, K=4, seed=0)))
    assert 0 in batch  # deduplicated batch still contains the single image


def test_mining_buckets_by_rank_and_sampler_adds_negatives():
    torch.manual_seed(0)
    centres = torch.randn(10, 16)
    embeddings = centres.repeat_interleave(4, dim=0) + 0.05 * torch.randn(40, 16)
    labels = [i // 4 for i in range(40)]
    config = MiningConfig(
        top_k=10, num_hard=2, num_positives=1,
        min_negative_similarity=-1.0, max_positive_similarity=1.0,
    )
    mined = mine_neighbours(embeddings, labels, config)
    examples = mined[0]
    assert len(examples.negatives["hard"]) == 2
    assert all(labels[i] != 0 for i in examples.negatives["hard"] + examples.negatives["semi_hard"])
    assert len(examples.positives["hard"]) == 1
    sampler = PKSampler(labels, P=4, K=2, mined=mined, num_negatives=2, seed=0)
    batch = next(iter(sampler))
    assert len(batch) > 8


def test_nested_galleries():
    gallery = [Sample(path=f"{s}/{i}.jpg", sku_id=s) for s in "ab" for i in range(12)]
    one = set(nested_gallery_indices(gallery, 1))
    five = set(nested_gallery_indices(gallery, 5))
    ten = set(nested_gallery_indices(gallery, 10))
    assert one <= five <= ten
    assert len(one) == 2 and len(ten) == 20
