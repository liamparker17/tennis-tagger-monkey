from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from ml.feature_extractor.process import process_clip
from ml.feature_extractor.schema import load_npz
from ml.point_model.model import PointModel, PointModelConfig
from ml.point_model.features import build_feature_tensor
from ml.point_model.fusion import fuse, detect_audio_impulses, FusedPointPrediction

class Predictor:
    def __init__(self, ckpt: Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        sd = torch.load(ckpt, map_location=self.device)
        self.model = PointModel(PointModelConfig(**sd["config"])).to(self.device)
        self.model.load_state_dict(sd["model"]); self.model.eval()

    @torch.no_grad()
    def predict_clip(self, clip_path: Path, tmp_dir: Path) -> FusedPointPrediction:
        npz_path = tmp_dir / (clip_path.stem + ".npz")
        process_clip(clip_path, npz_path)
        fs = load_npz(npz_path)
        feats = build_feature_tensor(
            pose_px=fs.pose_px, pose_court=fs.pose_court,
            pose_conf=fs.pose_conf, pose_valid=fs.pose_valid,
            ball=fs.ball, audio_mel=fs.audio_mel, clip_meta=fs.clip_meta,
        )
        x = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        mask = torch.ones(1, feats.shape[0], device=self.device)
        out = self.model(x, mask)
        T = feats.shape[0]
        impulses = detect_audio_impulses(fs.audio_mel, T)
        return fuse(
            out["contact_logits"][0], out["hitter_per_frame_logits"][0],
            out["stroke_logits"][0], out["inout_logits"][0],
            out["outcome_logits"][0],
            audio_impulse=impulses, pose_court=fs.pose_court, fps=int(fs.fps),
        )
