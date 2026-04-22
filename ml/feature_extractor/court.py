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


def homography_from_setup(setup: dict, frame_w: int, frame_h: int) -> np.ndarray | None:
    """Compute image->court homography from a preflight sidecar.

    Sidecar corners are in the setup reference frame's native pixel space.
    Pipeline operates on the clip's native frame, so the scale is per-clip.
    """
    try:
        import cv2
        corners = setup.get("court_corners_pixel") or []
        if len(corners) != 4: return None
        src_w = float(setup.get("frame_width") or frame_w)
        src_h = float(setup.get("frame_height") or frame_h)
        sx = float(frame_w) / src_w
        sy = float(frame_h) / src_h
        pts_native = np.array([[c["x"] * sx, c["y"] * sy] for c in corners], dtype=np.float32)
        det_w, det_h = 640.0, 360.0
        pts_det = pts_native * np.array([det_w / frame_w, det_h / frame_h], dtype=np.float32)
        standard = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
        H, _ = cv2.findHomography(pts_det, standard)
        if H is None: return None
        return H.astype(np.float32)
    except Exception:
        return None

def project_to_court(px: np.ndarray, H: np.ndarray) -> np.ndarray:
    flat = px.reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1), np.float32)
    homo = np.concatenate([flat, ones], axis=1)
    proj = (H @ homo.T).T
    proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)
    return proj.reshape(px.shape)
