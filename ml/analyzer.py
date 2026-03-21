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

        Uses multi-threshold Canny edge detection, edge dilation, and
        line classification (horizontal vs vertical) for robust court
        corner estimation.

        The result is cached on success until :meth:`reset_court_cache`
        is called.  Failed detections (confidence == 0) are **not**
        cached so the next call retries.

        Args:
            frame: BGR image.

        Returns:
            Dict with ``corners``, ``homography``, ``method``,
            ``confidence``.  ``homography`` may be ``None`` if detection
            fails.
        """
        if self._court_cache is not None:
            return self._court_cache

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Multi-threshold Canny — break on first successful detection
        lines = None
        thresholds = [(30, 100), (50, 150), (75, 200)]
        for lo, hi in thresholds:
            edges = cv2.Canny(gray, lo, hi, apertureSize=3)
            edges = cv2.dilate(edges, kernel, iterations=1)

            detected = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=80,
                minLineLength=50,
                maxLineGap=20,
            )
            if detected is not None and len(detected) >= 4:
                lines = detected
                break

        if lines is None:
            return {
                "corners": None,
                "homography": None,
                "method": "line_detection",
                "confidence": 0.0,
            }

        # Classify lines into horizontal and vertical
        horizontals, verticals = self._separate_lines(lines, w, h)

        # Corner estimation — try methods in priority order
        if len(horizontals) >= 2 and len(verticals) >= 2:
            corners = self._corners_from_lines(horizontals, verticals, w, h)
            method = "line_intersection"
            confidence = 0.9
        elif len(lines) >= 4:
            corners = self._corners_from_endpoints(lines, w, h)
            method = "endpoint_extremes"
            confidence = 0.7
        else:
            corners = self._default_corners(w, h)
            method = "frame_proportion"
            confidence = 0.5

        # Standard normalised court
        standard = np.array(
            [[0, 1], [1, 1], [0, 0], [1, 0]],
            dtype=np.float32,
        )

        homography, _ = cv2.findHomography(corners, standard)

        self._court_cache = {
            "corners": corners.tolist(),
            "homography": homography,
            "method": method,
            "confidence": confidence,
        }
        logger.info("Court detected (%s, confidence=%.2f) and cached", method, confidence)
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
    # Court detection helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_corners(w: int, h: int) -> np.ndarray:
        """Frame-proportion fallback for court corners."""
        return np.array(
            [
                [w * 0.1, h * 0.8],   # bottom-left
                [w * 0.9, h * 0.8],   # bottom-right
                [w * 0.2, h * 0.2],   # top-left
                [w * 0.8, h * 0.2],   # top-right
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _separate_lines(
        lines: np.ndarray, w: int, h: int,
    ) -> tuple[list, list]:
        """Classify detected lines into horizontal and vertical groups.

        Lines shorter than 30 px are filtered out.
        Horizontal: angle < 30 deg or > 150 deg.
        Vertical: angle between 60 and 120 deg.
        """
        horizontals: list = []
        verticals: list = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if length < 30:
                continue
            angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            if angle < 30 or angle > 150:
                horizontals.append((x1, y1, x2, y2))
            elif 60 <= angle <= 120:
                verticals.append((x1, y1, x2, y2))

        return horizontals, verticals

    def _corners_from_lines(
        self,
        horizontals: list,
        verticals: list,
        w: int,
        h: int,
    ) -> np.ndarray:
        """Compute court corners from horizontal/vertical line intersections."""
        # Sort horizontals by average y (top first)
        horizontals = sorted(horizontals, key=lambda l: (l[1] + l[3]) / 2)
        # Sort verticals by average x (left first)
        verticals = sorted(verticals, key=lambda l: (l[0] + l[2]) / 2)

        top_h = horizontals[0]
        bot_h = horizontals[-1]
        left_v = verticals[0]
        right_v = verticals[-1]

        tl = self._line_intersection(top_h, left_v, w, h)
        tr = self._line_intersection(top_h, right_v, w, h)
        bl = self._line_intersection(bot_h, left_v, w, h)
        br = self._line_intersection(bot_h, right_v, w, h)

        return np.array([bl, br, tl, tr], dtype=np.float32)

    @staticmethod
    def _corners_from_endpoints(lines: np.ndarray, w: int, h: int) -> np.ndarray:
        """Estimate court corners from the extreme endpoints of all lines."""
        pts = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            pts.append((x1, y1))
            pts.append((x2, y2))

        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        return np.array(
            [
                [min_x, max_y],  # bottom-left
                [max_x, max_y],  # bottom-right
                [min_x, min_y],  # top-left
                [max_x, min_y],  # top-right
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _line_intersection(
        line1: tuple, line2: tuple, w: int, h: int,
    ) -> np.ndarray:
        """Compute intersection of two line segments, clamped to frame bounds.

        Each line is a tuple ``(x1, y1, x2, y2)``.
        Returns a 1-D array ``[x, y]``.
        """
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-6:
            # Parallel lines — return midpoint of the two midpoints
            mx = ((x1 + x2) / 2 + (x3 + x4) / 2) / 2
            my = ((y1 + y2) / 2 + (y3 + y4) / 2) / 2
            return np.array(
                [np.clip(mx, 0, w - 1), np.clip(my, 0, h - 1)],
                dtype=np.float32,
            )

        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)

        return np.array(
            [np.clip(ix, 0, w - 1), np.clip(iy, 0, h - 1)],
            dtype=np.float32,
        )

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
