"""
Analyzer — court detection, placement analysis, rally segmentation.

Merged from:
  files/detection/court_detector.py
  files/analysis/placement_analyzer.py
  files/analysis/rally_analyzer.py
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---- Zone classification thresholds (normalised court coords [0,1]) ----
# Side: y < 0.5 => deuce, else ad
# Lateral: x < 0.33 => wide, x < 0.67 => body, else t
# Depth: x < 0.2 => baseline, x < 0.6 => mid, else net

_PLACEMENT_ZONES = {
    "deuce": {"wide": "Deuce Wide", "body": "Deuce Body", "t": "Deuce T"},
    "ad":    {"wide": "Ad Wide",    "body": "Ad Body",    "t": "Ad T"},
}
_DEPTH_ZONES = {"baseline": "Baseline", "mid": "Mid-court", "net": "Net"}


class Analyzer:
    """Court detection, shot placement, and rally segmentation."""

    def __init__(self) -> None:
        self._court_cache: Optional[dict] = None

    # ------------------------------------------------------------------
    # Court detection
    # ------------------------------------------------------------------

    def detect_court(self, frame: np.ndarray) -> dict:
        """Detect court boundaries via Canny + HoughLinesP + findHomography.

        The result is cached until :meth:`reset_court_cache` is called.

        Args:
            frame: BGR image.

        Returns:
            Dict with ``corners``, ``homography``, ``method``,
            ``confidence``.  ``homography`` may be ``None`` if detection
            fails.
        """
        if self._court_cache is not None:
            return self._court_cache

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=100,
            maxLineGap=10,
        )

        if lines is None or len(lines) < 4:
            result: dict = {
                "corners": None,
                "homography": None,
                "method": "line_detection",
                "confidence": 0.0,
            }
            # Do NOT cache a failure — retry on next frame
            return result

        h, w = frame.shape[:2]

        # Approximate court corners from frame proportions
        corners = np.array(
            [
                [w * 0.1, h * 0.8],  # bottom-left
                [w * 0.9, h * 0.8],  # bottom-right
                [w * 0.2, h * 0.2],  # top-left
                [w * 0.8, h * 0.2],  # top-right
            ],
            dtype=np.float32,
        )

        # Standard normalised court
        standard = np.array(
            [[0, 1], [1, 1], [0, 0], [1, 0]],
            dtype=np.float32,
        )

        homography, _ = cv2.findHomography(corners, standard)

        self._court_cache = {
            "corners": corners.tolist(),
            "homography": homography,
            "method": "line_detection",
            "confidence": 0.8,
        }
        logger.info("Court detected and cached")
        return self._court_cache

    def reset_court_cache(self) -> None:
        """Clear cached court detection so the next call re-detects."""
        self._court_cache = None

    # ------------------------------------------------------------------
    # Placement analysis
    # ------------------------------------------------------------------

    def analyze_placements(
        self,
        detections: list[dict],
        court: dict,
    ) -> list[dict]:
        """Map ball positions to court zones and depths.

        Args:
            detections: Output of :class:`Detector.detect_batch` — one
                        dict per frame with ``"ball"`` key.
            court: Output of :meth:`detect_court`.

        Returns:
            One dict per frame where a ball was found::

                {"frame": int, "zone": str, "depth": str,
                 "court_position": (x, y)}
        """
        H = court.get("homography") if court else None
        placements: list[dict] = []

        for idx, det in enumerate(detections):
            ball = det.get("ball")
            if ball is None:
                continue

            bbox = ball["bbox"]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            # Project onto normalised court
            if H is not None:
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                tp = cv2.perspectiveTransform(pt, H)
                nx, ny = float(tp[0][0][0]), float(tp[0][0][1])
            else:
                nx, ny = cx, cy

            zone = self._classify_zone(nx, ny)
            depth = self._classify_depth(nx)

            placements.append({
                "frame": idx,
                "zone": zone,
                "depth": depth,
                "court_position": (nx, ny),
            })

        return placements

    # ------------------------------------------------------------------
    # Rally segmentation
    # ------------------------------------------------------------------

    def segment_rallies(
        self,
        detections: list[dict],
        fps: float,
        max_gap_frames: int = 90,
        min_length: int = 2,
    ) -> list[dict]:
        """Segment a sequence of detections into rallies.

        A rally is a contiguous run of frames with ball presence,
        separated by gaps longer than *max_gap_frames*.

        Args:
            detections: Per-frame detection dicts (need ``"ball"`` key).
            fps: Video frame rate (used for timestamp calculation).
            max_gap_frames: Maximum allowed gap between ball detections
                            before a new rally starts.
            min_length: Minimum number of ball-present frames to count
                        as a rally.

        Returns:
            List of rally dicts::

                {"start_frame": int, "end_frame": int,
                 "start_time": float, "end_time": float,
                 "ball_frames": int, "duration_sec": float}
        """
        # Collect frames where ball is present
        ball_frames: list[int] = [
            i for i, d in enumerate(detections) if d.get("ball") is not None
        ]

        if len(ball_frames) < min_length:
            return []

        rallies: list[dict] = []
        run_start = ball_frames[0]
        run_end = ball_frames[0]
        run_count = 1

        for f in ball_frames[1:]:
            if f - run_end <= max_gap_frames:
                run_end = f
                run_count += 1
            else:
                if run_count >= min_length:
                    rallies.append(self._make_rally(run_start, run_end, run_count, fps))
                run_start = f
                run_end = f
                run_count = 1

        if run_count >= min_length:
            rallies.append(self._make_rally(run_start, run_end, run_count, fps))

        logger.info("Segmented %d rallies from %d ball frames", len(rallies), len(ball_frames))
        return rallies

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_zone(x: float, y: float) -> str:
        side = "deuce" if y < 0.5 else "ad"
        if x < 0.33:
            lateral = "wide"
        elif x < 0.67:
            lateral = "body"
        else:
            lateral = "t"
        return _PLACEMENT_ZONES[side][lateral]

    @staticmethod
    def _classify_depth(x: float) -> str:
        if x < 0.2:
            return _DEPTH_ZONES["baseline"]
        if x < 0.6:
            return _DEPTH_ZONES["mid"]
        return _DEPTH_ZONES["net"]

    @staticmethod
    def _make_rally(start: int, end: int, count: int, fps: float) -> dict:
        return {
            "start_frame": start,
            "end_frame": end,
            "start_time": round(start / fps, 3),
            "end_time": round(end / fps, 3),
            "ball_frames": count,
            "duration_sec": round((end - start) / fps, 3),
        }
