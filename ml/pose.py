"""
Pose Estimator + Serve Detector.

Merged from files/detection/pose_estimator.py and
files/detection/serve_detector.py into a single module.

Primary: YOLOv8n-pose (GPU).  Fallback: MediaPipe Pose (CPU).
"""

import logging
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO

    _YOLO_OK = True
except ImportError:
    _YOLO_OK = False

try:
    import mediapipe as mp

    _MP_OK = True
except ImportError:
    _MP_OK = False

logger = logging.getLogger(__name__)

# Serve detection parameters
_MIN_ARM_ANGLE = 120  # degrees at elbow
_SERVE_DEDUP_FRAMES = 30  # suppress duplicates within this window

# COCO keypoint indices (17-point layout)
_SHOULDER_R = 6
_ELBOW_R = 8
_WRIST_R = 10
_SHOULDER_L = 5
_ELBOW_L = 7
_WRIST_L = 9


class PoseEstimator:
    """Player pose estimation with optional serve detection.

    Loads YOLOv8n-pose when available (GPU-accelerated batch processing),
    falling back to MediaPipe Pose in lite mode on CPU.
    """

    def __init__(self, device: str = "auto"):
        """
        Args:
            device: ``"auto"`` picks CUDA when available, else CPU.
        """
        self._model: Optional[object] = None
        self._mp_pose: Optional[object] = None
        self._mode: str = "none"
        self.device: str = "cpu"

        # --- Try YOLO-pose first ---
        if _YOLO_OK:
            try:
                import torch

                if device == "auto" and torch.cuda.is_available():
                    self.device = "0"
                elif device not in ("auto", "cpu"):
                    self.device = device

                self._model = YOLO("yolov8n-pose.pt")
                if self.device != "cpu":
                    self._model.to("cuda")

                # Warmup
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self._model.predict(dummy, verbose=False, device=self.device)

                self._mode = "yolo"
                logger.info("YOLOv8-pose ready (device=%s)", self.device)
            except Exception:
                logger.warning("YOLOv8-pose unavailable, trying MediaPipe", exc_info=True)
                self._model = None

        # --- Fallback: MediaPipe ---
        if self._model is None and _MP_OK:
            self._init_mediapipe()

        if self._mode == "none":
            logger.warning("No pose backend available")

    # ------------------------------------------------------------------
    # MediaPipe init
    # ------------------------------------------------------------------

    def _init_mediapipe(self) -> None:
        mp_pose = mp.solutions.pose
        self._mp_pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=0,  # lite
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._mode = "mediapipe"
        logger.info("MediaPipe Pose ready (lite, CPU)")

    # ------------------------------------------------------------------
    # Public API — batch pose estimation
    # ------------------------------------------------------------------

    def estimate_batch(
        self,
        frames: list[np.ndarray],
        players: list[list[dict]],
    ) -> list[list[dict]]:
        """Estimate poses for every player in every frame.

        Args:
            frames: List of BGR frames.
            players: Per-frame list of player dicts (each must have ``"bbox"``).

        Returns:
            Per-frame, per-player keypoint dicts::

                [
                    [  # frame 0
                        {"player_id": int, "keypoints": [{"x","y","confidence"}, ...], "bbox": [...]},
                        ...
                    ],
                    ...
                ]
        """
        if not frames:
            return []

        if self._mode == "yolo" and self._model is not None:
            return self._yolo_batch(frames, players)
        if self._mode == "mediapipe" and self._mp_pose is not None:
            return [
                self._mediapipe_frame(f, p)
                for f, p in zip(frames, players)
            ]
        return [[] for _ in frames]

    # ------------------------------------------------------------------
    # YOLO-pose batch
    # ------------------------------------------------------------------

    def _yolo_batch(
        self,
        frames: list[np.ndarray],
        players: list[list[dict]],
    ) -> list[list[dict]]:
        results_batch = self._model.predict(
            frames,
            verbose=False,
            device=self.device,
            conf=0.5,
            half=(self.device != "cpu"),
            augment=False,
        )

        all_poses: list[list[dict]] = []
        for results in results_batch:
            frame_poses: list[dict] = []
            if results.keypoints is not None:
                kp_data = results.keypoints.data.cpu().numpy()  # (N, 17, 3)
                for i, kps in enumerate(kp_data):
                    keypoints = []
                    for j in range(17):
                        keypoints.append({
                            "x": float(kps[j, 0]),
                            "y": float(kps[j, 1]),
                            "confidence": float(kps[j, 2]),
                        })
                    bbox = None
                    if i < len(results.boxes):
                        bbox = [float(v) for v in results.boxes[i].xyxy[0].cpu().numpy()]
                    frame_poses.append({
                        "player_id": -1,
                        "keypoints": keypoints,
                        "bbox": bbox,
                    })
            all_poses.append(frame_poses)
        return all_poses

    # ------------------------------------------------------------------
    # MediaPipe per-frame
    # ------------------------------------------------------------------

    def _mediapipe_frame(
        self,
        frame: np.ndarray,
        player_dets: list[dict],
    ) -> list[dict]:
        poses: list[dict] = []
        for player in player_dets:
            x1, y1, x2, y2 = (int(v) for v in player["bbox"])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            result = self._mp_pose.process(rgb)

            if result.pose_landmarks:
                keypoints = []
                for lm in result.pose_landmarks.landmark:
                    keypoints.append({
                        "x": float(x1 + lm.x * (x2 - x1)),
                        "y": float(y1 + lm.y * (y2 - y1)),
                        "confidence": float(lm.visibility),
                    })
                poses.append({
                    "player_id": player.get("id", -1),
                    "keypoints": keypoints,
                    "bbox": player["bbox"],
                })
        return poses

    # ------------------------------------------------------------------
    # Serve detection
    # ------------------------------------------------------------------

    def detect_serves(
        self,
        poses: list[list[dict]],
        frame_index: int = 0,
    ) -> list[dict]:
        """Detect serve motions from pose sequences.

        A serve is flagged when the arm angle at the elbow exceeds
        ``_MIN_ARM_ANGLE`` (120 deg) **and** the wrist is above the
        shoulder.  Duplicates within ``_SERVE_DEDUP_FRAMES`` are
        suppressed.

        Args:
            poses: Output of :meth:`estimate_batch` — per-frame list of
                   pose dicts.
            frame_index: Starting frame index (offset added to local
                         indices).

        Returns:
            List of serve events::

                {"frame": int, "player_id": int, "arm_angle": float,
                 "confidence": float, "type": "serve"}
        """
        serves: list[dict] = []

        for local_idx, frame_poses in enumerate(poses):
            abs_frame = frame_index + local_idx
            for pose in frame_poses:
                kps = pose.get("keypoints", [])

                # Check right arm
                serve = self._check_arm(kps, _SHOULDER_R, _ELBOW_R, _WRIST_R, abs_frame, pose)
                if serve is not None:
                    serves.append(serve)
                    continue

                # Check left arm
                serve = self._check_arm(kps, _SHOULDER_L, _ELBOW_L, _WRIST_L, abs_frame, pose)
                if serve is not None:
                    serves.append(serve)

        return self._dedup_serves(serves)

    def _check_arm(
        self,
        kps: list[dict],
        shoulder_idx: int,
        elbow_idx: int,
        wrist_idx: int,
        frame: int,
        pose: dict,
    ) -> Optional[dict]:
        """Return a serve dict if the arm geometry indicates a serve."""
        if len(kps) <= max(shoulder_idx, elbow_idx, wrist_idx):
            return None

        shoulder = kps[shoulder_idx]
        elbow = kps[elbow_idx]
        wrist = kps[wrist_idx]

        # Minimum confidence on all three joints
        if min(shoulder["confidence"], elbow["confidence"], wrist["confidence"]) < 0.3:
            return None

        angle = self._arm_angle(shoulder, elbow, wrist)

        if angle > _MIN_ARM_ANGLE and wrist["y"] < shoulder["y"]:
            return {
                "frame": frame,
                "player_id": pose.get("player_id", -1),
                "arm_angle": round(angle, 1),
                "confidence": 0.8,
                "type": "serve",
            }
        return None

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _arm_angle(shoulder: dict, elbow: dict, wrist: dict) -> float:
        """Angle at the elbow (degrees)."""
        v1 = np.array([shoulder["x"] - elbow["x"], shoulder["y"] - elbow["y"]])
        v2 = np.array([wrist["x"] - elbow["x"], wrist["y"] - elbow["y"]])
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    @staticmethod
    def _dedup_serves(serves: list[dict], window: int = _SERVE_DEDUP_FRAMES) -> list[dict]:
        """Remove duplicate serve events within *window* frames."""
        if not serves:
            return []
        ordered = sorted(serves, key=lambda s: s["frame"])
        keep = [ordered[0]]
        for s in ordered[1:]:
            if (s["frame"] - keep[-1]["frame"] > window
                    or s["player_id"] != keep[-1]["player_id"]):
                keep.append(s)
        return keep

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self) -> None:
        if self._mp_pose is not None:
            try:
                self._mp_pose.close()
            except Exception:
                pass
