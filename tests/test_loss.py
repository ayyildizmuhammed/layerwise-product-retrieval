import torch

from lpr.training.loss import batch_hard_triplet_loss


def test_loss_uses_hardest_pairs():
    embeddings = torch.tensor([[1.0, 0.0], [0.6, 0.8], [0.8, 0.6], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    # Anchor 0: hardest positive d=0.4, hardest negative d=0.2 -> 0.4 - 0.2 + 0.3 = 0.5.
    # Anchor 1: pos d=0.4, neg min(d(1,2)=0.04, d(1,3)=0.2)=0.04 -> 0.66.
    # Anchor 2: pos d(2,3)=0.4, neg min(0.2, 0.04)=0.04 -> 0.66.
    # Anchor 3: pos 0.4, neg min(1.0, 0.2)=0.2 -> 0.5.
    expected = (0.5 + 0.66 + 0.66 + 0.5) / 4
    assert abs(batch_hard_triplet_loss(embeddings, labels).item() - expected) < 1e-5


def test_loss_is_zero_when_well_separated():
    embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.01], [-1.0, 0.0], [-1.0, 0.01]])
    labels = torch.tensor([0, 0, 1, 1])
    assert batch_hard_triplet_loss(embeddings, labels).item() == 0.0


def test_loss_without_valid_anchor_is_zero_and_differentiable():
    embeddings = torch.randn(3, 4, requires_grad=True)
    loss = batch_hard_triplet_loss(embeddings, torch.tensor([0, 1, 2]))
    loss.backward()
    assert loss.item() == 0.0
