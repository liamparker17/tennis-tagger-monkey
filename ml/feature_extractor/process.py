from __future__ import annotations
from pathlib import Path
import numpy as np
from .schema import FeatureSet, save_npz, SCHEMA_VERSION
from .frames import iter_frames, probe_wh
from .pose import PoseExtractor
from .court import estimate_homography, project_to_court
from .audio import extract_log_mel
from .ball import BallDetector

POSE_MODEL = Path("models/yolov8s-pose.pt")
BALL_MODEL = Path("models/yolov8s.pt")


def process_clip(clip: Path, out_npz: Path, fps: int = 30,
                 pose: PoseExtractor | None = None,
                 ball: BallDetector | None = None) -> None:
    pose = pose or PoseExtractor(POSE_MODEL)
    ball = ball or BallDetector(BALL_MODEL)

    w, h, native_fps = probe_wh(clip)
    frames = list(iter_frames(clip, fps=fps))
    T = len(frames)
    pose_px = np.zeros((T, 2, 17, 2), np.float32)
    pose_conf = np.zeros((T, 2, 17), np.float32)
    ball_arr = np.zeros((T, 3), np.float32)
    for t, fr in enumerate(frames):
        pose_px[t], pose_conf[t] = pose.extract(fr)
        ball_arr[t] = ball.detect(fr)
    H = estimate_homography(frames[T // 2]) if T else np.eye(3, np.float32)
    pose_court = project_to_court(pose_px, H)
    pose_valid = pose_conf.max(axis=-1) > 0.2  # (T, 2)
    mel = extract_log_mel(clip)
    duration_s = T / float(fps)
    fs = FeatureSet(
        schema=SCHEMA_VERSION, fps=fps,
        pose_px=pose_px, pose_conf=pose_conf,
        pose_court=pose_court, pose_valid=pose_valid,
        court_h=H.astype(np.float32), ball=ball_arr, audio_mel=mel,
        clip_meta=np.array([duration_s, w, h, native_fps], np.float32),
    )
    save_npz(out_npz, fs)
