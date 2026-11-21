"""End-to-end evaluation on synthetic embeddings (no model weights needed)."""

import torch

from lpr.data.manifest import Sample
from lpr.evaluation.benchmark import evaluate_representation


def _split_and_embeddings():
    torch.manual_seed(0)
    known = [f"k{i}" for i in range(6)]
    family = {"k0": "f0", "k1": "f0", "k2": "f1", "k3": "f1", "k4": "f2", "k5": "f2",
              "u0": "f0", "u1": "f9", "u2": "f9"}
    centres = {sku: torch.nn.functional.normalize(torch.randn(32), dim=0) for sku in family}

    def make(role, skus, per_sku, noise):
        samples, vectors = [], []
        for sku in skus:
            for i in range(per_sku):
                path = f"{role}/{sku}/{i}.jpg"
                samples.append(Sample(path, sku, family[sku], f"{role}-{sku}-{i}"))
                vectors.append(centres[sku] + noise * torch.randn(32))
        return samples, torch.stack(vectors)

    split, embeddings = {}, {}
    for role, skus, per_sku, noise in [
        ("gallery", known, 10, 0.05), ("test_known", known, 5, 0.1), ("val_known", known, 2, 0.1),
        ("val_unknown", ["u1"], 4, 0.1), ("test_unknown", ["u0", "u2"], 4, 0.1),
    ]:
        split[role], embeddings[role] = make(role, skus, per_sku, noise)
    return split, embeddings


def test_evaluate_representation_end_to_end():
    split, embeddings = _split_and_embeddings()
    for rule in ("score", "score_margin"):
        results = evaluate_representation(embeddings, split, references=(10, 1), rule=rule)
        assert [r.references for r in results] == [10, 1]
        top = results[0]
        assert top.metrics["recall_at_1"] == 100.0
        assert top.metrics["num_queries"] == 30
        assert set(top.rejection) == {"all", "same_family", "cross_family"}
        # u0 shares family f0 with gallery SKUs; u2 does not.
        assert top.rejection["same_family"]["num_unknown"] == 4
        assert top.rejection["cross_family"]["num_unknown"] == 4
        assert top.threshold["rule"] == rule
        low, high = top.rejection["all"]["far_ci"]
        assert low <= top.rejection["all"]["far"] <= high
        assert len(top.records.rows()) == 30


def test_prototype_gallery_results():
    split, embeddings = _split_and_embeddings()
    results = evaluate_representation(embeddings, split, references=(10,), prototypes=True)
    assert [r.gallery for r in results] == ["crops", "prototype"]
    assert results[1].metrics["recall_at_1"] == 100.0
    assert results[1].rejection == {}
