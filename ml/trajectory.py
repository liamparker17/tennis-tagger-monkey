"""Physics-based trajectory fitting for tennis ball detections.

Takes a sequence of ball detections (pixel coords at 640x360) and a court
homography matrix, and produces fitted trajectory segments with bounce points
and in/out calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from scipy.optimize import curve_fit

# ---------------------------------------------------------------------------
# Court constants — mirror of internal/point/rules.go
# ---------------------------------------------------------------------------
COURT_LENGTH = 23.77        # baseline to baseline in meters
SINGLES_WIDTH = 8.23        # singles sideline to sideline
DOUBLES_WIDTH = 10.97
NET_Y = 11.885              # net at halfway
SERVICE_BOX_DEPTH = 6.40    # net to service line
CENTER_LINE_X = 4.115       # center of singles court
CLOSE_CALL_MARGIN = 0.05    # 5 cm

# Minimum detections required to attempt a fit
_MIN_DETECTIONS = 3

# Velocity change threshold (m/s) to classify as a bounce
_BOUNCE_VELOCITY_THRESHOLD = 2.0

# Minimum time gap between bounces (seconds)
_MIN_BOUNCE_GAP = 0.10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Trajectory:
    """Fitted ball trajectory for a segment of detections."""

    start_frame: int
    end_frame: int
    positions: List[dict] = field(default_factory=list)   # [{cx, cy, t, frame_index}]
    bounces: List[dict] = field(default_factory=list)     # [{frame_index, cx, cy, in_out}]
    speed_kph: float = 0.0          # estimated peak speed in km/h
    confidence: float = 0.0         # fit quality [0, 1]
    cx0: float = 0.0                # fitted x-intercept (m)
    vx: float = 0.0                 # fitted x-velocity (m/s)
    cy0: float = 0.0                # fitted y-intercept (m)
    vy: float = 0.0                 # fitted y-velocity (m/s)

    def to_dict(self) -> dict:
        return {
            "startFrame": self.start_frame,
            "endFrame": self.end_frame,
            "positions": self.positions,
            "bounces": self.bounces,
            "speedKph": self.speed_kph,
            "confidence": self.confidence,
            "cx0": self.cx0,
            "vx": self.vx,
            "cy0": self.cy0,
            "vy": self.vy,
        }


# ---------------------------------------------------------------------------
# TrajectoryFitter
# ---------------------------------------------------------------------------

class TrajectoryFitter:
    """Fits a 2-D linear trajectory to a sequence of ball detections.

    The court is described by a 3x3 homography H that maps pixel coordinates
    (at 640x360 resolution) to normalised court coordinates in [0, 1], where
    (0, 0) is one baseline corner and (1, 1) is the diagonally opposite corner.

    Real-world distances are recovered by scaling:
        cx = nx * SINGLES_WIDTH   (meters, across the court)
        cy = ny * COURT_LENGTH    (meters, along the court)
    """

    def __init__(self, homography: np.ndarray, fps: float) -> None:
        """
        Args:
            homography: 3x3 float array mapping pixel -> normalised court coords.
            fps:        Video frame rate (frames per second).
        """
        self.H = np.asarray(homography, dtype=np.float64)
        self.fps = float(fps)

    # ------------------------------------------------------------------
    # Coordinate transform
    # ------------------------------------------------------------------

    def pixel_to_court(self, px: float, py: float) -> tuple[float, float]:
        """Map pixel (px, py) to real-world court metres (cx, cy).

        Uses the homography to obtain normalised coords, then scales to metres.
        """
        src = np.array([px, py, 1.0], dtype=np.float64)
        dst = self.H @ src
        if dst[2] == 0:
            return 0.0, 0.0
        nx = dst[0] / dst[2]
        ny = dst[1] / dst[2]
        cx = nx * SINGLES_WIDTH
        cy = ny * COURT_LENGTH
        return float(cx), float(cy)

    # ------------------------------------------------------------------
    # Landing classification
    # ------------------------------------------------------------------

    @staticmethod
    def classify_landing(cx: float, cy: float) -> str:
        """Return 'in', 'out', or 'close_call' for a landing at (cx, cy) metres.

        Mirrors ClassifyLanding in internal/point/rules.go.
        """
        in_x = 0.0 <= cx <= SINGLES_WIDTH
        in_y = 0.0 <= cy <= COURT_LENGTH
        if in_x and in_y:
            return "in"
        margin_x = -CLOSE_CALL_MARGIN <= cx <= SINGLES_WIDTH + CLOSE_CALL_MARGIN
        margin_y = -CLOSE_CALL_MARGIN <= cy <= COURT_LENGTH + CLOSE_CALL_MARGIN
        if margin_x and margin_y:
            return "close_call"
        return "out"

    # ------------------------------------------------------------------
    # Bounce detection
    # ------------------------------------------------------------------

    def find_bounces(self, detections: List[dict]) -> List[dict]:
        """Detect bounce points from a sequence of detections.

        A bounce is signalled by a sign change or sharp deceleration in the
        cy component of velocity.  Returns list of
        ``{"frame_index": int, "cx": float, "cy": float}``.
        """
        if len(detections) < _MIN_DETECTIONS:
            return []

        # Build time series
        frames = np.array([d["frame_index"] for d in detections], dtype=float)
        court_coords = [self.pixel_to_court(d["x"], d["y"]) for d in detections]
        cy_arr = np.array([c[1] for c in court_coords], dtype=float)
        cx_arr = np.array([c[0] for c in court_coords], dtype=float)
        t_arr = frames / self.fps

        if len(t_arr) < 2:
            return []

        # Compute finite-difference velocities at each interior point
        dt = np.diff(t_arr)
        dt = np.where(dt == 0, 1e-6, dt)
        vy_arr = np.diff(cy_arr) / dt

        bounces = []
        last_bounce_t = -999.0

        for i in range(1, len(vy_arr)):
            # Sign change in vy — ball reversed direction along the court axis
            sign_change = vy_arr[i - 1] * vy_arr[i] < 0
            # Sharp deceleration (magnitude drops by threshold)
            sharp_decel = abs(vy_arr[i - 1] - vy_arr[i]) > _BOUNCE_VELOCITY_THRESHOLD

            if sign_change or sharp_decel:
                t_candidate = t_arr[i]
                if t_candidate - last_bounce_t < _MIN_BOUNCE_GAP:
                    continue
                last_bounce_t = t_candidate
                # Interpolate position at bounce: average the two surrounding points
                cx_b = float((cx_arr[i] + cx_arr[i + 1]) / 2) if i + 1 < len(cx_arr) else float(cx_arr[i])
                cy_b = float((cy_arr[i] + cy_arr[i + 1]) / 2) if i + 1 < len(cy_arr) else float(cy_arr[i])
                bounces.append({
                    "frame_index": int(detections[i]["frame_index"]),
                    "cx": cx_b,
                    "cy": cy_b,
                })

        return bounces

    # ------------------------------------------------------------------
    # Trajectory fitting
    # ------------------------------------------------------------------

    def fit(self, detections: List[dict]) -> Optional[Trajectory]:
        """Fit a trajectory to a list of ball detections.

        Args:
            detections: List of dicts with keys x, y, confidence, frame_index.

        Returns:
            A Trajectory object, or None if there are too few detections to fit.
        """
        if len(detections) < _MIN_DETECTIONS:
            return None

        # Sort by frame index to ensure time-ordering
        detections = sorted(detections, key=lambda d: d["frame_index"])

        # Convert to court coords and time
        court_pts = [self.pixel_to_court(d["x"], d["y"]) for d in detections]
        cx_arr = np.array([p[0] for p in court_pts], dtype=float)
        cy_arr = np.array([p[1] for p in court_pts], dtype=float)
        t_arr = np.array([d["frame_index"] / self.fps for d in detections], dtype=float)
        t0 = t_arr[0]
        t_rel = t_arr - t0  # relative time starting at 0

        # Fit linear models: cx(t) = cx0 + vx*t, cy(t) = cy0 + vy*t
        def linear(t: np.ndarray, c0: float, v: float) -> np.ndarray:
            return c0 + v * t

        cx0, vx, cy0, vy = 0.0, 0.0, 0.0, 0.0
        fit_confidence = 0.0

        try:
            popt_x, _ = curve_fit(linear, t_rel, cx_arr)
            popt_y, _ = curve_fit(linear, t_rel, cy_arr)
            cx0, vx = float(popt_x[0]), float(popt_x[1])
            cy0, vy = float(popt_y[0]), float(popt_y[1])

            # Residual-based confidence: smaller residuals -> higher confidence
            cx_pred = linear(t_rel, cx0, vx)
            cy_pred = linear(t_rel, cy0, vy)
            rmse_x = float(np.sqrt(np.mean((cx_arr - cx_pred) ** 2)))
            rmse_y = float(np.sqrt(np.mean((cy_arr - cy_pred) ** 2)))
            rmse = (rmse_x + rmse_y) / 2.0
            # Map RMSE in metres to confidence: 0 m -> 1.0, 2 m -> ~0.0
            fit_confidence = float(np.clip(1.0 - rmse / 2.0, 0.0, 1.0))
            # Scale down for small detection counts
            count_scale = min(1.0, len(detections) / 10.0)
            fit_confidence *= count_scale

        except (RuntimeError, ValueError):
            # curve_fit failed — fall back to first-differences
            if len(t_rel) >= 2 and t_rel[-1] > 0:
                vx = float((cx_arr[-1] - cx_arr[0]) / t_rel[-1])
                vy = float((cy_arr[-1] - cy_arr[0]) / t_rel[-1])
                cx0 = float(cx_arr[0])
                cy0 = float(cy_arr[0])
            fit_confidence = 0.1

        # Build positions list
        positions = []
        for i, d in enumerate(detections):
            positions.append({
                "cx": float(cx_arr[i]),
                "cy": float(cy_arr[i]),
                "t": float(t_arr[i]),
                "frame_index": int(d["frame_index"]),
            })

        # Bounce detection
        raw_bounces = self.find_bounces(detections)
        bounces_with_inout = []
        for b in raw_bounces:
            in_out = self.classify_landing(b["cx"], b["cy"])
            bounces_with_inout.append({
                "frame_index": b["frame_index"],
                "cx": b["cx"],
                "cy": b["cy"],
                "in_out": in_out,
            })

        # Speed estimate: 2-D court-plane speed in km/h
        speed_ms = float(np.sqrt(vx ** 2 + vy ** 2))
        speed_kph = speed_ms * 3.6

        return Trajectory(
            start_frame=int(detections[0]["frame_index"]),
            end_frame=int(detections[-1]["frame_index"]),
            positions=positions,
            bounces=bounces_with_inout,
            speed_kph=speed_kph,
            confidence=fit_confidence,
            cx0=cx0,
            vx=vx,
            cy0=cy0,
            vy=vy,
        )
