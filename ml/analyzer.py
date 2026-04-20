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
        self._manual_court: Optional[dict] = None
        self.near_player: Optional[str] = None
        self.far_player: Optional[str] = None

    def set_manual_court(
        self,
        corners_pixel: list,
        frame_width: int,
        frame_height: int,
        near_player: Optional[str] = None,
        far_player: Optional[str] = None,
    ) -> None:
        """Pre-populate the court cache from user-clicked corners.

        The user clicks in **native pixel space** (whatever the video is
        recorded at, e.g. 1920x1080). The rest of the pipeline operates in
        640x360 detection space, so we scale the corners down before
        computing the homography. This matches ``rpc_detect_court``'s
        convention of storing H in 640x360 → [0,1] court coordinates.

        ``corners_pixel`` is ordered [near_left, near_right, far_right, far_left]
        — the same order ``preflight.py`` writes to the sidecar. Near = bottom
        of frame = player A. Far = top of frame = player B.

        Standard court target (normalised [0,1]):
            near_left  -> (0, 1)   near_right -> (1, 1)
            far_right  -> (1, 0)   far_left   -> (0, 0)
        """
        if len(corners_pixel) != 4:
            raise ValueError(f"expected 4 corners, got {len(corners_pixel)}")
        if frame_width <= 0 or frame_height <= 0:
            raise ValueError(f"bad frame size: {frame_width}x{frame_height}")

        # Native pixel corners (what the user clicked on)
        pts_native = np.array(corners_pixel, dtype=np.float32).reshape(-1, 2)

        # Scale into 640x360 detection space
        det_w, det_h = 640.0, 360.0
        sx = det_w / float(frame_width)
        sy = det_h / float(frame_height)
        pts_det = pts_native * np.array([sx, sy], dtype=np.float32)

        standard = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=np.float32)
        homography, _ = cv2.findHomography(pts_det, standard)
        if homography is None:
            raise ValueError("could not compute homography from corners — likely collinear")

        # Expand polygon 10% outward for player-filtering headroom.
        # Kept in native-pixel space; rpc_set_manual_court rescales it for
        # the detector (same pattern as rpc_detect_court).
        cx_m = float(pts_native[:, 0].mean())
        cy_m = float(pts_native[:, 1].mean())
        expanded = ((pts_native - [cx_m, cy_m]) * 1.10 + [cx_m, cy_m])
        expanded = np.clip(expanded, 0, [frame_width - 1, frame_height - 1]).astype(np.int32)

        self._manual_court = {
            "corners": pts_native.tolist(),         # native pixels (for debug/inspection)
            "corners_det": pts_det.tolist(),        # 640x360 pixels (for homography domain)
            "polygon": expanded.tolist(),           # native pixels
            "homography": homography,               # 640x360 → [0,1] court
            "method": "manual_preflight",
            "confidence": 1.0,
            "native_width": int(frame_width),
            "native_height": int(frame_height),
        }
        self._court_cache = self._manual_court
        self.near_player = near_player
        self.far_player = far_player
        logger.info(
            "Court set manually from pre-flight: native=%dx%d near=%r far=%r",
            frame_width, frame_height, near_player, far_player,
        )

    # ------------------------------------------------------------------
    # Court detection
    # ------------------------------------------------------------------

    def detect_court(self, frame: np.ndarray) -> dict:
        """Detect court boundaries from white lines on a coloured surface.

        Strategy:
        1. Detect court surface colour (clay/hard court via HSV).
        2. Find white pixels adjacent to the court surface (court lines).
        3. Thicken and close gaps to form enclosed regions.
        4. Find enclosed regions in the central frame area (service boxes etc).
        5. Convex hull of those regions → court polygon.

        Falls back to frame-proportion estimate if detection fails.

        The result is cached on success until :meth:`reset_court_cache`
        is called.

        Args:
            frame: BGR image.

        Returns:
            Dict with ``corners``, ``polygon``, ``homography``, ``method``,
            ``confidence``.  ``polygon`` is the expanded court boundary
            for player filtering.
        """
        if self._manual_court is not None:
            return self._manual_court
        if self._court_cache is not None:
            return self._court_cache

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # --- Step 1: Detect court surface ---
        # Try common court colours: clay (red/orange), hard (blue), grass (green)
        clay = cv2.inRange(hsv, np.array([0, 25, 50]), np.array([25, 255, 255]))
        blue = cv2.inRange(hsv, np.array([90, 30, 50]), np.array([130, 255, 255]))
        green = cv2.inRange(hsv, np.array([35, 30, 50]), np.array([85, 255, 255]))

        # Pick the dominant surface
        counts = {"clay": clay.sum(), "blue": blue.sum(), "green": green.sum()}
        surface_name = max(counts, key=counts.get)
        surface_mask = {"clay": clay, "blue": blue, "green": green}[surface_name]
        logger.info("Court surface: %s (%d%% of frame)", surface_name,
                     int(counts[surface_name] / 255 / (h * w) * 100))

        # --- Step 2: White pixels adjacent to court surface ---
        surface_dilated = cv2.dilate(
            surface_mask, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        )
        white = cv2.inRange(hsv, np.array([0, 0, 185]), np.array([180, 55, 255]))
        court_white = white & surface_dilated

        # --- Step 3: Find court boundary from line segment extreme points ---
        # Detect line segments in the white court mask
        edges = cv2.Canny(court_white, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
                                minLineLength=max(20, w // 50),
                                maxLineGap=max(10, w // 100))

        if lines is not None and len(lines) >= 4:
            # Collect all line endpoints
            pts = []
            for l in lines:
                x1, y1, x2, y2 = l[0]
                pts.extend([(x1, y1), (x2, y2)])
            pts = np.array(pts)

            # Find the 4 extreme points (court corners in perspective)
            tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]       # top-left
            tr = pts[np.argmax(pts[:, 0] - pts[:, 1])]       # top-right
            bl = pts[np.argmin(pts[:, 0] - pts[:, 1])]       # bottom-left
            br = pts[np.argmax(pts[:, 0] + pts[:, 1])]       # bottom-right

            corners_poly = np.array([tl, tr, br, bl], dtype=np.int32)
            method = "line_extremes"
            confidence = 0.85
        else:
            corners_poly = self._default_corners(w, h).astype(np.int32)
            method = "frame_proportion"
            confidence = 0.3

        # Expand polygon by 10% for players standing behind baselines
        M = cv2.moments(corners_poly.reshape(-1, 1, 2))
        if M["m00"] > 0:
            cx_m = M["m10"] / M["m00"]
            cy_m = M["m01"] / M["m00"]
            expanded = ((corners_poly.astype(float) - [cx_m, cy_m]) * 1.10
                        + [cx_m, cy_m]).astype(np.int32)
            expanded = np.clip(expanded, 0, [w - 1, h - 1])
        else:
            expanded = corners_poly

        # Pick 4 representative corners for homography (extremes of polygon)
        four = self._pick_four_corners(corners_poly, w, h)
        standard = np.array([[0, 1], [1, 1], [0, 0], [1, 0]], dtype=np.float32)
        homography, _ = cv2.findHomography(four, standard)

        self._court_cache = {
            "corners": four.tolist(),
            "polygon": expanded.tolist(),
            "homography": homography,
            "method": method,
            "confidence": confidence,
        }
        logger.info("Court detected (%s, confidence=%.2f, %d vertices) and cached",
                     method, confidence, len(expanded))
        return self._court_cache

    @staticmethod
    def _pick_four_corners(poly: np.ndarray, w: int, h: int) -> np.ndarray:
        """Select 4 corners from polygon for homography (TL, TR, BL, BR)."""
        pts = poly.reshape(-1, 2).astype(np.float32)
        # Top-left: closest to (0, 0)
        tl = pts[np.argmin(pts[:, 0] + pts[:, 1])]
        # Top-right: closest to (w, 0)
        tr = pts[np.argmin((pts[:, 0] - w) ** 2 + pts[:, 1] ** 2)]
        # Bottom-left: closest to (0, h)
        bl = pts[np.argmin(pts[:, 0] ** 2 + (pts[:, 1] - h) ** 2)]
        # Bottom-right: closest to (w, h)
        br = pts[np.argmin((pts[:, 0] - w) ** 2 + (pts[:, 1] - h) ** 2)]
        return np.array([bl, br, tl, tr], dtype=np.float32)

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
        H_raw = court.get("homography") if court else None
        # Convert list-of-lists to numpy array if needed
        if H_raw is not None and not isinstance(H_raw, np.ndarray):
            H = np.array(H_raw, dtype=np.float64)
        else:
            H = H_raw
        placements: list[dict] = []

        for idx, det in enumerate(detections):
            ball = det.get("ball")
            if ball is None:
                continue

            # Handle both formats:
            #   Python detector: {"bbox": [x1,y1,x2,y2]}
            #   Go bridge JSON:  {"x1": .., "y1": .., "x2": .., "y2": ..}
            if "bbox" in ball:
                bbox = ball["bbox"]
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
            elif "x1" in ball:
                cx = (ball["x1"] + ball["x2"]) / 2.0
                cy = (ball["y1"] + ball["y2"]) / 2.0
            else:
                continue

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


def detect_court_homography(frame_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Thin wrapper returning the 3x3 image->court homography for a BGR frame.

    Used by feature extractors that need a one-shot homography estimate. Returns
    None on any failure so callers can fall back to identity.
    """
    try:
        result = Analyzer().detect_court(frame_bgr)
        H = result.get("homography")
        if H is None:
            return None
        return np.asarray(H)
    except Exception:
        return None
