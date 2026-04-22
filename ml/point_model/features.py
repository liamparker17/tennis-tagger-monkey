from __future__ import annotations
import numpy as np

FEATURE_DIM = 242  # 68 pose_px + 68 pose_court + 34 conf + 2 valid + 3 ball + 64 mel + 3 meta

def _resample_mel_to_T(mel: np.ndarray, T: int) -> np.ndarray:
    n_mels, F = mel.shape
    if F == T: return mel.T.astype(np.float32)
    idx = np.linspace(0, F - 1, num=T).astype(np.int64) if F > 0 else np.zeros(T, np.int64)
    return mel[:, idx].T.astype(np.float32)

def build_feature_tensor(pose_px, pose_court, pose_conf, pose_valid,
                         ball, audio_mel, clip_meta) -> np.ndarray:
    T = pose_px.shape[0]
    W = max(float(clip_meta[1]), 1.0); H = max(float(clip_meta[2]), 1.0)
    px_n  = (pose_px / np.array([W, H], np.float32)).reshape(T, -1)
    crt_n = pose_court.reshape(T, -1)
    conf  = pose_conf.reshape(T, -1)
    valid = pose_valid.astype(np.float32).reshape(T, -1)
    ball_n = np.stack([
        ball[:, 0] / W, ball[:, 1] / H, ball[:, 2]
    ], axis=1).astype(np.float32)
    mel = _resample_mel_to_T(audio_mel, T)
    # clip_meta = [duration_s, w, h, native_fps]. Raw values (30 s, 1920, 1080,
    # 30 fps) would dominate the feature vector with huge un-normalized
    # numbers, so we rescale into roughly O(1): duration over 30 s, the w/h
    # contribute only as aspect ratio (W and H are already used as pose
    # normalizers), fps over 30.
    dur_s, w_px, h_px, fps = [float(x) for x in clip_meta.tolist()]
    aspect = (w_px / max(h_px, 1.0))
    meta_norm = np.array([dur_s / 30.0, aspect, fps / 30.0], np.float32)
    meta = np.tile(meta_norm.reshape(1, -1), (T, 1)).astype(np.float32)
    out = np.concatenate([px_n, crt_n, conf, valid, ball_n, mel, meta], axis=1)
    assert out.shape[1] == FEATURE_DIM, out.shape
    return out
