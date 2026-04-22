from __future__ import annotations
from pathlib import Path
import numpy as np
from .schema import FeatureSet, save_npz, SCHEMA_VERSION
from .frames import iter_frames, probe_wh
from .pose import PoseExtractor
from .court import estimate_homography, homography_from_setup, project_to_court
from .audio import extract_log_mel
from .ball import BallDetector

POSE_MODEL = Path("models/yolov8s-pose.pt")
# Prefer the fine-tuned ball detector (trained from preflight ball_labels via
# ml.train_yolo) when it exists; stock yolov8s.pt otherwise. This lets users
# upgrade ball detection just by training — no code change required.
_BALL_FINETUNED = Path("files/models/yolo_ball/best.pt")
BALL_MODEL = _BALL_FINETUNED if _BALL_FINETUNED.is_file() else Path("models/yolov8s.pt")


DARTFISH_COURT_W = 300.0
DARTFISH_COURT_H = 300.0


def _court_xy_to_pixel(xy_court: list[float], H_img_to_court: np.ndarray,
                       det_w: float, det_h: float,
                       frame_w: int, frame_h: int) -> tuple[float, float] | None:
    try:
        nx = float(xy_court[0]) / DARTFISH_COURT_W
        ny = float(xy_court[1]) / DARTFISH_COURT_H
        H_inv = np.linalg.inv(H_img_to_court)
        v = H_inv @ np.array([nx, ny, 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-9: return None
        det_x = v[0] / v[2]; det_y = v[1] / v[2]
        px = det_x * (frame_w / det_w)
        py = det_y * (frame_h / det_h)
        if not (0 <= px < frame_w and 0 <= py < frame_h): return None
        return float(px), float(py)
    except Exception:
        return None


def process_clip(clip: Path, out_npz: Path, fps: int = 30,
                 pose: PoseExtractor | None = None,
                 ball: BallDetector | None = None,
                 setup: dict | None = None,
                 ball_anchors: list[dict] | None = None,
                 ball_window_frames: int = 15) -> None:
    pose = pose or PoseExtractor(POSE_MODEL)
    ball = ball or BallDetector(BALL_MODEL)

    w, h, native_fps = probe_wh(clip)

    # We don't know T up front; `_build_run_ball` is grown lazily in the
    # streaming loop when the decoder outruns the current mask size.
    T_est = 0

    anchor_frames = [int(a["frame"]) for a in (ball_anchors or [])
                     if int(a["frame"]) >= 0]
    # Only run the ball detector near human-confirmed anchor frames. Those
    # are the only timestamps where we have ground truth to supervise
    # against, so running the detector on unlabelled frames produces noise
    # the downstream model can't actually learn from. Fall back to running
    # everywhere only when a clip has no anchors at all.
    def _build_run_ball(T: int) -> np.ndarray:
        if anchor_frames:
            rb = np.zeros(T, dtype=bool)
            for af in anchor_frames:
                if af >= T: continue
                lo = max(0, af - ball_window_frames)
                hi = min(T, af + ball_window_frames + 1)
                rb[lo:hi] = True
            return rb
        return np.ones(T, dtype=bool)
    run_ball = _build_run_ball(T_est)

    # Stream frames one at a time — never hold the full clip in memory.
    pose_px_list: list[np.ndarray] = []
    pose_conf_list: list[np.ndarray] = []
    ball_list: list[np.ndarray] = []
    mid_frame: np.ndarray | None = None
    mid_target = T_est // 2 if T_est > 0 else -1
    T = 0
    for t, fr in enumerate(iter_frames(clip, fps=fps)):
        p_px, p_conf = pose.extract(fr)
        pose_px_list.append(p_px)
        pose_conf_list.append(p_conf)
        # If the stream outruns our duration-based mask, extend it.
        if t >= run_ball.shape[0]:
            run_ball = _build_run_ball(t + 1)
        if run_ball[t]:
            ball_list.append(ball.detect(fr))
        else:
            ball_list.append(np.zeros(3, dtype=np.float32))
        # Retain exactly one frame for the homography fallback; if the
        # user's preflight setup is present we won't need it and release it.
        if t == mid_target:
            mid_frame = fr.copy()
        T = t + 1

    pose_px = np.stack(pose_px_list, axis=0).astype(np.float32) if pose_px_list \
        else np.zeros((0, 2, 17, 2), np.float32)
    pose_conf = np.stack(pose_conf_list, axis=0).astype(np.float32) if pose_conf_list \
        else np.zeros((0, 2, 17), np.float32)
    ball_arr = np.stack(ball_list, axis=0).astype(np.float32) if ball_list \
        else np.zeros((0, 3), np.float32)
    # Re-mask ball output in case T differs from T_est (rare, but keeps
    # per-clip behaviour identical to the old listing implementation).
    if ball_arr.shape[0] > 0:
        run_ball = _build_run_ball(ball_arr.shape[0])
        ball_arr[~run_ball] = 0.0

    H = None
    if setup is not None:
        H = homography_from_setup(setup, w, h)
    if H is None:
        H = estimate_homography(mid_frame) if mid_frame is not None else np.eye(3, np.float32)
    mid_frame = None  # release the single retained frame asap

    if ball_anchors:
        for a in ball_anchors:
            f = int(a.get("frame", -1))
            xy = a.get("xy_court")
            if not (0 <= f < T) or xy is None: continue
            px = _court_xy_to_pixel(xy, H, 640.0, 360.0, w, h)
            if px is None: continue
            ball_arr[f] = (px[0], px[1], 1.0)

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
