import numpy as np
import pytest
import torch

from lpr.evaluation.metrics import closed_set_metrics, family_confusion_rates, rank_of_truth
from lpr.retrieval.search import collapse_to_skus, cosine_scores


def test_collapse_takes_best_crop_per_sku():
    scores = torch.tensor([[0.1, 0.9, 0.3, 0.2], [0.8, 0.1, 0.0, 0.7]])
    ranking = collapse_to_skus(scores, ["a", "a", "b", "c"])
    assert ranking.sku_ids == ["a", "b", "c"]
    np.testing.assert_allclose(ranking.scores, [[0.9, 0.3, 0.2], [0.8, 0.0, 0.7]], rtol=1e-6)
    np.testing.assert_array_equal(ranking.best_gallery_index[:, 0], [1, 0])


def test_collapse_chunks_match_single_pass():
    generator = torch.Generator().manual_seed(0)
    scores = torch.rand(50, 30, generator=generator)
    labels = [f"s{i % 7}" for i in range(30)]
    np.testing.assert_allclose(
        collapse_to_skus(scores, labels, chunk_size=7).scores,
        collapse_to_skus(scores, labels).scores,
    )


def test_top_two_and_ranked_labels():
    ranking = collapse_to_skus(torch.tensor([[0.2, 0.9, 0.5]]), ["x", "y", "z"])
    s1, s2 = ranking.top_two()
    assert s1[0] == np.float32(0.9) and s2[0] == np.float32(0.5)
    assert list(ranking.ranked_labels(3)[0]) == ["y", "z", "x"]


def test_cosine_scores_normalises():
    queries = torch.tensor([[2.0, 0.0]])
    gallery = torch.tensor([[0.0, 3.0], [5.0, 0.0]])
    np.testing.assert_allclose(cosine_scores(queries, gallery).numpy(), [[0.0, 1.0]])


def test_metrics_on_known_ranking():
    # Gallery SKUs a, b, c. Query 1 ranks a first; query 2 (true b) ranks b second;
    # query 3 (true d) has no SKU in the gallery.
    scores = torch.tensor([[0.9, 0.2, 0.1], [0.8, 0.7, 0.1], [0.5, 0.4, 0.3]])
    ranking = collapse_to_skus(scores, ["a", "b", "c"])
    truth = ["a", "b", "d"]
    np.testing.assert_array_equal(rank_of_truth(ranking, truth), [1, 2, 0])
    metrics = closed_set_metrics(ranking, truth, family_of={"a": "f", "b": "f", "c": "g", "d": "g"})
    assert metrics.recall_at_1 == pytest.approx(100 / 3)
    assert metrics.recall_at_5 == pytest.approx(200 / 3)
    assert abs(metrics.map_at_10 - 100 * (1 + 0.5) / 3) < 1e-9
    assert metrics.macro_recall_at_1 == pytest.approx(100 / 3)
    # Query 2 predicts a (same family f as b): one family confusion among two errors.
    assert metrics.fcr_all == pytest.approx(100 / 3)
    assert metrics.fcr_error == pytest.approx(50.0)


def test_family_confusion_without_errors():
    assert family_confusion_rates(["a"], ["a"], {"a": "f"}) == (0.0, 0.0)
