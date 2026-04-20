import torch
from ml.point_model.losses import compute_losses

def test_losses_run():
    B, T = 2, 32
    logits = {
        "contact_logits": torch.zeros(B, T),
        "bounce_logits": torch.zeros(B, T),
        "hitter_per_frame_logits": torch.zeros(B, T, 2),
        "stroke_logits": torch.zeros(B, 4, 9),
        "inout_logits": torch.zeros(B, 4),
        "outcome_logits": torch.zeros(B, 5),
    }
    targets = {
        "contact": torch.zeros(B, T, dtype=torch.long),
        "contact_strong": torch.tensor([1.0, 0.0]),
        "hitter_per_frame": torch.full((B, T), -1, dtype=torch.long),
        "stroke": torch.full((B, 4), -100, dtype=torch.long),
        "outcome": torch.zeros(B, dtype=torch.long),
        "mask": torch.ones(B, T),
    }
    losses = compute_losses(logits, targets)
    assert "total" in losses and torch.is_tensor(losses["total"])
