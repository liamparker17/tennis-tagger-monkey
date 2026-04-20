from __future__ import annotations
import numpy as np

def estimate_homography(middle_frame_bgr: np.ndarray) -> np.ndarray:
    try:
        from ml.analyzer import detect_court_homography
    except Exception:
        return np.eye(3, dtype=np.float32)
    try:
        H = detect_court_homography(middle_frame_bgr)
        if H is None: return np.eye(3, dtype=np.float32)
        return H.astype(np.float32)
    except Exception:
        return np.eye(3, dtype=np.float32)

def project_to_court(px: np.ndarray, H: np.ndarray) -> np.ndarray:
    flat = px.reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1), np.float32)
    homo = np.concatenate([flat, ones], axis=1)
    proj = (H @ homo.T).T
    proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)
    return proj.reshape(px.shape)
