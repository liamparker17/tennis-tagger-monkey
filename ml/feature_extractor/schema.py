from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

SCHEMA_VERSION = 1


@dataclass
class FeatureSet:
    schema: int
    fps: int
    pose_px: np.ndarray      # (T, 2, 17, 2) float32 - keypoint pixels
    pose_conf: np.ndarray    # (T, 2, 17)    float32
    pose_court: np.ndarray   # (T, 2, 17, 2) float32 - keypoint in court coords
    pose_valid: np.ndarray   # (T, 2)        bool
    court_h: np.ndarray      # (3, 3)        float32 - homography image->court
    ball: np.ndarray         # (T, 3)        float32 - (x, y, conf), 0 where missing
    audio_mel: np.ndarray    # (n_mels, F)   float32 - log-mel
    clip_meta: np.ndarray    # (4,)          float32 - [duration_s, w, h, fps_native]


def save_npz(path: Path, fs: FeatureSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        **{k: v for k, v in asdict(fs).items() if isinstance(v, np.ndarray)},
        schema=np.int32(fs.schema),
        fps=np.int32(fs.fps),
    )


def load_npz(path: Path) -> FeatureSet:
    z = np.load(path)
    return FeatureSet(
        schema=int(z["schema"]),
        fps=int(z["fps"]),
        pose_px=z["pose_px"],
        pose_conf=z["pose_conf"],
        pose_court=z["pose_court"],
        pose_valid=z["pose_valid"],
        court_h=z["court_h"],
        ball=z["ball"],
        audio_mel=z["audio_mel"],
        clip_meta=z["clip_meta"],
    )
