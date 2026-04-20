"""
JSON-RPC bridge server for Go <-> Python ML communication.

Reads JSON-RPC requests from stdin (one per line), dispatches to ML
modules, writes JSON responses to stdout (one per line).  All logging
goes to stderr to keep the stdout channel clean.

Protocol:
    Startup:  writes {"ready": true} to stdout.
    Request:  {"method": "...", "params": {...}, "id": N}
    Response: {"result": {...}, "id": N}  or  {"error": "...", "id": N}
"""

import base64
import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Logging — all output goes to stderr so stdout stays clean for JSON-RPC
# ---------------------------------------------------------------------------
logging.basicConfig(
    stream=sys.stderr,
    format="%(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger("bridge")

# ---------------------------------------------------------------------------
# Windows stdout fix — prevent \r\n line endings
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    sys.stdout.reconfigure(newline="\n")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# numpy JSON serialiser
# ---------------------------------------------------------------------------
def _json_default(obj: Any) -> Any:
    """Convert numpy types to native Python for json.dumps."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _encode(obj: Any) -> str:
    """Encode *obj* to a single-line JSON string."""
    return json.dumps(obj, default=_json_default, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Frame decoding helper
# ---------------------------------------------------------------------------
def _decode_frame(frame_dict: dict) -> np.ndarray:
    """Decode a frame dict to a numpy array.

    Supports two modes:
    - Shared memory: keys ``shm_path``, ``offset``, ``width``, ``height``, ``size``
    - Base64 fallback: keys ``width``, ``height``, ``data_base64``
    """
    w = frame_dict["width"]
    h = frame_dict["height"]

    if "shm_path" in frame_dict:
        # Shared memory mode — read raw bytes from file at offset
        with open(frame_dict["shm_path"], "rb") as f:
            f.seek(frame_dict["offset"])
            raw = f.read(frame_dict["size"])
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        return arr
    else:
        # Base64 fallback
        raw = base64.b64decode(frame_dict["data_base64"])
        arr = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        return arr


# ---------------------------------------------------------------------------
# Bridge server
# ---------------------------------------------------------------------------
class BridgeServer:
    """JSON-RPC dispatch to ML modules."""

    def __init__(self) -> None:
        self.detector: Optional[Any] = None
        self.pose: Optional[Any] = None
        self.analyzer: Optional[Any] = None
        self.score: Optional[Any] = None
        self.trainer: Optional[Any] = None
        self._initialised = False

    # ---- RPC: __init__ ----------------------------------------------------

    def rpc_init(self, params: dict) -> dict:
        """Initialise all ML modules."""
        import torch

        torch.set_num_threads(1)

        models_dir = params.get("ModelsDir", "models")
        device = params.get("Device", "auto")

        logger.info("Initialising ML modules (models_dir=%s, device=%s)", models_dir, device)

        from ml.detector import Detector
        from ml.pose import PoseEstimator
        from ml.analyzer import Analyzer
        from ml.score import ScoreTracker
        from ml.trainer import Trainer

        detector_backend = params.get("DetectorBackend", "yolo")

        detector_models = {"yolo": "yolo12n.pt", "yolov5": "yolov5s.pt", "yolov8s": "yolov8s.pt"}

        detector_weights = detector_models.get(detector_backend, "yolo12n.pt")
        _detector_abs = os.path.join(models_dir, detector_weights)
        # If weights aren't on disk, pass the bare filename so ultralytics auto-downloads
        # from its GitHub release into its cache.
        detector_model = _detector_abs if os.path.exists(_detector_abs) else detector_weights

        self.detector = Detector(model_path=detector_model, device=device, backend=detector_backend)
        self.pose = PoseEstimator(device=device)
        self.analyzer = Analyzer()
        self.score = ScoreTracker(device="cpu" if device == "auto" else device)
        self.trainer = Trainer(models_dir=models_dir, device=device)

        self._initialised = True
        logger.info("All ML modules initialised")
        return {"status": "ok"}

    # ---- RPC: detect_batch ------------------------------------------------

    def rpc_detect_batch(self, params: dict) -> Any:
        """Decode frames (shared memory or base64) and run batch detection."""
        self._require_init()
        frames_data = params.get("frames", [])
        if not frames_data:
            return []

        t0 = time.perf_counter()
        shm_path = params.get("shm_path")
        if shm_path:
            # Shared memory mode: read all frames from single file
            batch = []
            with open(shm_path, "rb") as f:
                for meta in frames_data:
                    f.seek(meta["offset"])
                    raw = f.read(meta["size"])
                    arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (meta["height"], meta["width"], 3)
                    )
                    batch.append(arr)
        else:
            # Base64 fallback
            batch = [_decode_frame(f) for f in frames_data]
        t_decode = time.perf_counter()

        results = self.detector.detect_batch(batch)
        t_infer = time.perf_counter()

        logger.info(
            "detect_batch profile: n=%d decode_ms=%.1f infer_ms=%.1f total_ms=%.1f",
            len(batch),
            (t_decode - t0) * 1000,
            (t_infer - t_decode) * 1000,
            (t_infer - t0) * 1000,
        )
        return results

    # ---- RPC: detect_court ------------------------------------------------

    def rpc_set_manual_court(self, params: dict) -> Any:
        """Install user-clicked court corners from the pre-flight sidecar.

        Params: ``{"corners_pixel": [[x,y], ...], "frame_width": int,
                   "frame_height": int, "near_player": str, "far_player": str}``

        Corners are in NATIVE pixel space (what the user saw when clicking).
        The analyzer scales them into 640x360 detection space internally so
        the resulting homography is compatible with TrackNet ball coords,
        matching ``rpc_detect_court``'s convention.

        Corner order: near_left, near_right, far_right, far_left.
        """
        self._require_init()
        native_w = int(params["frame_width"])
        native_h = int(params["frame_height"])
        self.analyzer.set_manual_court(
            corners_pixel=params["corners_pixel"],
            frame_width=native_w,
            frame_height=native_h,
            near_player=params.get("near_player"),
            far_player=params.get("far_player"),
        )

        # Mirror rpc_detect_court: rescale the court polygon to 640-wide
        # detection space before handing it to the YOLO detector for
        # on-court player filtering.
        polygon = self.analyzer._manual_court.get("polygon")
        if polygon is not None:
            det_w = 640
            if native_w != det_w and native_w > 0:
                scale = det_w / native_w
                scaled = [[int(p[0] * scale), int(p[1] * scale)] for p in polygon]
                self.detector.set_court_polygon(scaled)
            else:
                self.detector.set_court_polygon(polygon)

        return {"ok": True, "method": "manual_preflight"}

    def rpc_detect_court(self, params: dict) -> Any:
        """Detect court boundaries in a single frame.

        If detection returns an identity (or near-identity) homography,
        substitutes a broadcast fallback derived from standard court
        dimensions and typical broadcast camera position.

        The returned homography is always in 640x360 detection space so it
        can be applied directly to TrackNet ball coordinates.
        """
        self._require_init()
        frame = _decode_frame(params["frame"])
        native_w, native_h = frame.shape[1], frame.shape[0]
        result = self.analyzer.detect_court(frame)

        # Check if homography is identity or near-identity (detection failed)
        homography = result.get("homography")
        use_fallback = False
        if homography is not None:
            H = np.array(homography, dtype=float)
            if H.shape == (3, 3) and np.allclose(H, np.eye(3), atol=1e-4):
                logger.warning("Court detection returned identity homography, using broadcast fallback")
                use_fallback = True
        else:
            logger.warning("Court detection returned no homography, using broadcast fallback")
            use_fallback = True

        if use_fallback:
            # Fallback already produces homography in 640x360 space
            result = self._broadcast_fallback(native_w, native_h, result)
            logger.info("Using broadcast fallback homography")
        else:
            # Court detection homography is in native pixel space.
            # Rescale to 640x360 detection space so it works with TrackNet coords.
            logger.info("Using detected court homography (method=%s, confidence=%.2f)",
                       result.get("method", "unknown"), result.get("confidence", 0))
            det_w, det_h = 640, 360
            if native_w != det_w or native_h != det_h:
                H = np.array(result["homography"], dtype=float)
                # Pre-multiply by a scaling matrix: pixel_640 → pixel_native → [0,1]
                # S maps 640x360 → native: S = diag(native_w/640, native_h/360, 1)
                S = np.diag([native_w / det_w, native_h / det_h, 1.0])
                H_det = H @ S  # now maps 640x360 pixels → [0,1]
                result["homography"] = H_det.tolist()

        # Set court polygon on detector for player filtering
        polygon = result.get("polygon")
        if polygon is not None:
            det_w = 640
            if native_w != det_w and native_w > 0:
                scale = det_w / native_w
                scaled = [[int(p[0] * scale), int(p[1] * scale)] for p in polygon]
                self.detector.set_court_polygon(scaled)
            else:
                self.detector.set_court_polygon(polygon)
        return result

    @staticmethod
    def _broadcast_fallback(width: int, height: int, result: dict) -> dict:
        """Compute a fallback homography for standard broadcast camera angle.

        The homography is always expressed in 640x360 detection space (the
        resolution TrackNet operates at), regardless of the native frame size.
        Maps approximate court corner pixels to normalised [0,1] court coords
        which ``TrajectoryFitter.pixel_to_court`` then scales to metres.
        """
        import cv2

        # Work in 640x360 detection space — scale native coords down
        det_w, det_h = 640.0, 360.0
        scale_x = det_w / float(width)
        scale_y = det_h / float(height)

        # Approximate court corner positions for a standard broadcast view
        # at 640x360.  These ratios work well for typical elevated side-on
        # camera angles (e.g. Wimbledon, USO, AO).
        #   far baseline: top of frame, narrower (perspective)
        #   near baseline: lower portion of frame, wider
        # The far points are placed above the actual baseline to account
        # for the ball being in flight (1-3m above court surface) which
        # shifts pixel position upward at the far end.
        src = np.float32([
            [det_w * 0.22, det_h * 0.06],   # far-left (above baseline for airborne ball)
            [det_w * 0.62, det_h * 0.06],   # far-right
            [det_w * 0.08, det_h * 0.75],   # near-left baseline corner
            [det_w * 0.84, det_h * 0.75],   # near-right baseline corner
        ])
        dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])

        H, _ = cv2.findHomography(src, dst)
        result["homography"] = H.tolist()
        result["method"] = "broadcast_fallback"
        result["confidence"] = 0.3
        # Store corners in native resolution for polygon use
        native_corners = (src / np.float32([scale_x, scale_y])).tolist()
        result["corners"] = native_corners
        return result

    # ---- RPC: analyze_placements ------------------------------------------

    def rpc_analyze_placements(self, params: dict) -> Any:
        """Map ball positions to court zones."""
        self._require_init()
        detections = params.get("detections", [])
        court = params.get("court", {})
        return self.analyzer.analyze_placements(detections, court)

    # ---- RPC: segment_rallies ---------------------------------------------

    def rpc_segment_rallies(self, params: dict) -> Any:
        """Segment detections into rallies."""
        self._require_init()
        detections = params.get("detections", [])
        fps = params.get("fps", 30.0)
        return self.analyzer.segment_rallies(detections, fps)

    # ---- RPC: train -------------------------------------------------------

    def rpc_train(self, params: dict) -> Any:
        """Train a new model version."""
        self._require_init()
        pairs = params.get("pairs", [])
        config = params.get("config", {})
        return self.trainer.train(pairs, config)

    # ---- RPC: fine_tune ---------------------------------------------------

    def rpc_fine_tune(self, params: dict) -> Any:
        """Fine-tune latest model with user corrections."""
        self._require_init()
        corrections = params.get("corrections", [])
        config = params.get("config", {})
        return self.trainer.fine_tune(corrections, config)

    # ---- RPC: get_versions ------------------------------------------------

    def rpc_get_versions(self, params: dict) -> Any:
        """List saved model versions."""
        self._require_init()
        return self.trainer.get_versions()

    # ---- RPC: rollback ----------------------------------------------------

    def rpc_rollback(self, params: dict) -> Any:
        """Rollback to a specific model version."""
        self._require_init()
        version = params.get("version", "")
        if not version:
            raise ValueError("version is required")
        return self.trainer.rollback(version)

    # ---- RPC: fit_trajectories --------------------------------------------

    def rpc_fit_trajectories(self, params: dict) -> Any:
        """Fit ball trajectories from ball position detections.

        Segments detections into individual shots, then fits one trajectory
        per segment.
        """
        from ml.trajectory import TrajectoryFitter, segment_detections

        ball_positions = params.get("ball_positions", [])
        court = params.get("court", {})
        fps = float(params.get("fps", 30.0))

        homography_raw = court.get("homography", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        homography = np.array(homography_raw, dtype=float)

        if homography.shape != (3, 3):
            raise ValueError(f"homography must be 3x3, got shape {homography.shape}")

        # Normalise key names: Go sends camelCase, Python uses snake_case
        detections = []
        for bp in ball_positions:
            detections.append({
                "x": float(bp.get("x", 0)),
                "y": float(bp.get("y", 0)),
                "confidence": float(bp.get("confidence", 0)),
                "frame_index": int(bp.get("frameIndex", bp.get("frame_index", 0))),
            })

        if not detections:
            return []

        # Segment into individual shots
        segments = segment_detections(detections, fps)
        logger.info("Segmented %d detections into %d segments", len(detections), len(segments))

        # Fit one trajectory per segment
        fitter = TrajectoryFitter(homography, fps)
        trajectories = []
        for seg in segments:
            traj = fitter.fit(seg)
            if traj is not None:
                trajectories.append(traj.to_dict())

        return trajectories

    # ---- Internal ---------------------------------------------------------

    def _require_init(self) -> None:
        if not self._initialised:
            raise RuntimeError("ML modules not initialised — call __init__ first")

    def dispatch(self, method: str, params: dict) -> Any:
        """Route a method name to the appropriate handler."""
        handlers = {
            "__init__": self.rpc_init,
            "detect_batch": self.rpc_detect_batch,
            "detect_court": self.rpc_detect_court,
            "analyze_placements": self.rpc_analyze_placements,
            "segment_rallies": self.rpc_segment_rallies,
            "train": self.rpc_train,
            "fine_tune": self.rpc_fine_tune,
            "get_versions": self.rpc_get_versions,
            "rollback": self.rpc_rollback,
            "fit_trajectories": self.rpc_fit_trajectories,
        }
        handler = handlers.get(method)
        if handler is None:
            raise ValueError(f"Unknown method: {method}")
        return handler(params)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    server = BridgeServer()

    # Signal readiness
    sys.stdout.write(_encode({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        req_id: Any = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            method = req["method"]
            params = req.get("params", {})

            result = server.dispatch(method, params)

            response = _encode({"result": result, "id": req_id})
            sys.stdout.write(response + "\n")
            sys.stdout.flush()

        except json.JSONDecodeError as exc:
            err = _encode({"error": f"Invalid JSON: {exc}", "id": None})
            sys.stdout.write(err + "\n")
            sys.stdout.flush()

        except Exception as exc:
            # Handle CUDA OOM specifically
            try:
                import torch
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    torch.cuda.empty_cache()
                    err_msg = f"CUDA_OOM: {exc}"
                    logger.error(err_msg)
                    response = _encode({"error": err_msg, "id": req_id})
                    sys.stdout.write(response + "\n")
                    sys.stdout.flush()
                    continue
            except ImportError:
                pass

            err_msg = f"{type(exc).__name__}: {exc}"
            logger.error("Request failed: %s\n%s", err_msg, traceback.format_exc())
            response = _encode({"error": err_msg, "id": req_id})
            sys.stdout.write(response + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
