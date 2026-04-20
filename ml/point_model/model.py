from __future__ import annotations
from dataclasses import dataclass
import torch, torch.nn as nn
from .vocab import NUM_STROKE, NUM_OUTCOME
from .features import FEATURE_DIM
from .labels import MAX_SHOTS

@dataclass
class PointModelConfig:
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 8
    ffn_dim: int = 1024
    dropout: float = 0.1
    max_T: int = 300
    n_shots: int = MAX_SHOTS

class PointModel(nn.Module):
    def __init__(self, cfg: PointModelConfig):
        super().__init__()
        self.cfg = cfg
        self.input_proj = nn.Linear(FEATURE_DIM, cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_T, cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.ffn_dim,
            dropout=cfg.dropout, batch_first=True, activation="gelu", norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)

        self.head_contact = nn.Linear(cfg.d_model, 1)
        self.head_bounce  = nn.Linear(cfg.d_model, 1)
        self.head_hitter  = nn.Linear(cfg.d_model, 2)

        self.shot_query = nn.Parameter(torch.randn(cfg.n_shots, cfg.d_model) * 0.02)
        self.shot_attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                               dropout=cfg.dropout, batch_first=True)
        self.head_stroke = nn.Linear(cfg.d_model, NUM_STROKE)
        self.head_inout  = nn.Linear(cfg.d_model, 1)

        self.head_outcome = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.GELU(),
            nn.Linear(cfg.d_model, NUM_OUTCOME),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        B, T, _ = x.shape
        h = self.input_proj(x) + self.pos[:, :T]
        key_padding = mask < 0.5
        h = self.encoder(h, src_key_padding_mask=key_padding)

        contact = self.head_contact(h).squeeze(-1)
        bounce = self.head_bounce(h).squeeze(-1)
        hit = self.head_hitter(h)

        q = self.shot_query.unsqueeze(0).expand(B, -1, -1)
        shot_h, _ = self.shot_attn(q, h, h, key_padding_mask=key_padding)
        stroke = self.head_stroke(shot_h)
        inout = self.head_inout(shot_h).squeeze(-1)

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        outcome = self.head_outcome(pooled)

        return {
            "contact_logits": contact, "bounce_logits": bounce,
            "hitter_per_frame_logits": hit,
            "stroke_logits": stroke, "inout_logits": inout,
            "outcome_logits": outcome,
        }
