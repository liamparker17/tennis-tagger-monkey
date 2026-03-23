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
        self.classifier: Optional[Any] = None
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
        from ml.classifier import StrokeClassifier
        from ml.analyzer import Analyzer
        from ml.score import ScoreTracker
        from ml.trainer import Trainer

        detector_model = os.path.join(models_dir, "yolov8s.pt")
        classifier_model = os.path.join(models_dir, "stroke_3dcnn.pt")

        self.detector = Detector(model_path=detector_model, device=device)
        self.pose = PoseEstimator(device=device)
        self.classifier = StrokeClassifier(model_path=classifier_model, device=device)
        self.analyzer = Analyzer()
        self.score = ScoreTracker(device="cpu" if device == "auto" else device)
        self.trainer = Trainer(models_dir=models_dir, device=device)

        from ml.tracknet import TrackNetDetector
        # Try pre-trained BallTrackerNet weights first, then TrackNetV2
        tracknet_model = os.path.join(models_dir, "yastrebksv_tracknet.pt")
        if not os.path.exists(tracknet_model):
            tracknet_model = os.path.join(models_dir, "tracknet_v2.pt")
        self.tracknet = TrackNetDetector(
            model_path=tracknet_model if os.path.exists(tracknet_model) else None,
            device=device,
        )

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

        return self.detector.detect_batch(batch)

    # ---- RPC: tracknet_batch ----------------------------------------------

    def rpc_tracknet_batch(self, params: dict) -> Any:
        """Run TrackNet ball detection on a batch of frames.

        Reads frames from shared memory (same layout as rpc_detect_batch),
        builds background reference on first call, assembles every-other-frame
        diff triplets [N-4, N-2, N] starting at index 4, and returns per-frame
        ball detections.

        Returns:
            List of {"frame_index": int, "ball": {"x": float, "y": float,
            "confidence": float} | null}
        """
        self._require_init()

        frames_data = params.get("frames", [])
        if not frames_data:
            return []

        shm_path = params.get("shm_path")
        if shm_path:
            batch = []
            with open(shm_path, "rb") as f:
                for meta in frames_data:
                    f.seek(meta["offset"])
                    raw = f.read(meta["size"])
                    arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (meta["height"], meta["width"], 3)
                    )
                    # frombuffer returns a read-only array — make a writable copy
                    batch.append(arr.copy())
        else:
            batch = [_decode_frame(f) for f in frames_data]

        from ml.tracknet import BackgroundSubtractor

        # Build background reference from batch ONLY if not already set
        # by a prior set_background_reference call
        if not hasattr(self, "_bg_subtractor") or self._bg_subtractor is None:
            self._bg_subtractor = BackgroundSubtractor()
        if not self._bg_subtractor.is_ready:
            logger.warning("Background reference not pre-set, building from batch (suboptimal)")
            ref_frames = batch[:30] if len(batch) >= 30 else batch
            self._bg_subtractor.build_reference(ref_frames)

        # Background-subtract every frame
        subtracted = [self._bg_subtractor.subtract(f) for f in batch]

        # Build diff triplets for every frame >= 4 using [i-4, i-2, i]
        # (triplet indices still skip by 2, matching TrackNet's temporal design)
        results = []
        for i in range(len(subtracted)):
            if i < 4:
                results.append({"frame_index": i, "ball": None})
                continue
            triplet = [subtracted[i - 4], subtracted[i - 2], subtracted[i]]
            detections = self.tracknet.detect_batch([triplet])
            ball = detections[0] if detections else None
            results.append({"frame_index": i, "ball": ball})

        return results

    # ---- RPC: set_background_reference ------------------------------------

    def rpc_set_background_reference(self, params: dict) -> Any:
        """Build background reference from frames sampled across the full video.

        Should be called once before any tracknet_batch calls, typically
        during the court detection phase.
        """
        self._require_init()
        from ml.tracknet import BackgroundSubtractor

        frames_data = params.get("frames", [])
        if not frames_data:
            return {"status": "no_frames"}

        shm_path = params.get("shm_path")
        if shm_path:
            batch = []
            with open(shm_path, "rb") as f:
                for meta in frames_data:
                    f.seek(meta["offset"])
                    raw = f.read(meta["size"])
                    arr = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (meta["height"], meta["width"], 3)
                    )
                    batch.append(arr.copy())
        else:
            batch = [_decode_frame(f) for f in frames_data]

        if not hasattr(self, "_bg_subtractor") or self._bg_subtractor is None:
            self._bg_subtractor = BackgroundSubtractor()
        self._bg_subtractor.build_reference(batch)
        logger.info("Background reference built from %d frames", len(batch))
        return {"status": "ok", "reference_frames": len(batch)}

    # ---- RPC: detect_court ------------------------------------------------

    def rpc_detect_court(self, params: dict) -> Any:
        """Detect court boundaries in a single frame.

        If detection returns an identity (or near-identity) homography,
        substitutes a broadcast fallback derived from standard court
        dimensions and typical broadcast camera position.
        """
        self._require_init()
        frame = _decode_frame(params["frame"])
        result = self.analyzer.detect_court(frame)

        # Check if homography is identity or near-identity (detection failed)
        homography = result.get("homography")
        if homography is not None:
            H = np.array(homography, dtype=float)
            if H.shape == (3, 3) and np.allclose(H, np.eye(3), atol=1e-4):
                logger.warning("Court detection returned identity homography, using broadcast fallback")
                result = self._broadcast_fallback(frame.shape[1], frame.shape[0], result)
        elif homography is None:
            logger.warning("Court detection returned no homography, using broadcast fallback")
            result = self._broadcast_fallback(frame.shape[1], frame.shape[0], result)

        # Set court polygon on detector for player filtering
        polygon = result.get("polygon")
        if polygon is not None:
            court_w = frame.shape[1]
            det_w = 640
            if court_w != det_w and court_w > 0:
                scale = det_w / court_w
                scaled = [[int(p[0] * scale), int(p[1] * scale)] for p in polygon]
                self.detector.set_court_polygon(scaled)
            else:
                self.detector.set_court_polygon(polygon)
        return result

    @staticmethod
    def _broadcast_fallback(width: int, height: int, result: dict) -> dict:
        """Compute a fallback homography for standard broadcast camera angle.

        Assumes camera is centered behind one baseline, ~10m high, ~5m back.
        Maps approximate court corner pixels to normalised [0,1] court coords.
        """
        import cv2

        w, h = float(width), float(height)
        src = np.float32([
            [w * 0.25, h * 0.17],   # top-left court corner
            [w * 0.75, h * 0.17],   # top-right court corner
            [w * 0.125, h * 0.94],  # bottom-left court corner
            [w * 0.875, h * 0.94],  # bottom-right court corner
        ])
        dst = np.float32([[0, 0], [1, 0], [0, 1], [1, 1]])

        H, _ = cv2.findHomography(src, dst)
        result["homography"] = H.tolist()
        result["method"] = "broadcast_fallback"
        result["confidence"] = 0.3
        result["corners"] = src.tolist()
        return result

    # ---- RPC: classify_strokes --------------------------------------------

    def rpc_classify_strokes(self, params: dict) -> Any:
        """Classify stroke types from video clips."""
        self._require_init()
        clips_data = params.get("clips", [])
        if not clips_data:
            return []

        # Each clip is {"frames": [frame_dict, ...]}
        clips = []
        for clip in clips_data:
            frames = [_decode_frame(f) for f in clip["frames"]]
            clips.append(np.stack(frames))

        batch = np.stack(clips)  # (N, T, H, W, 3)
        return self.classifier.classify(batch)

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
        """Fit ball trajectories from a sequence of ball position detections.

        Input params:
            ball_positions: list of {x, y, confidence, frameIndex}
            court: {homography: [[3x3 matrix]]}
            fps: float

        Returns list of trajectory dicts (see ml.trajectory.Trajectory.to_dict).
        """
        from ml.trajectory import TrajectoryFitter

        ball_positions = params.get("ball_positions", [])
        court = params.get("court", {})
        fps = float(params.get("fps", 30.0))

        # Extract homography — accept both nested list and flat list-of-lists
        homography_raw = court.get("homography", [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        homography = np.array(homography_raw, dtype=float)

        if homography.shape != (3, 3):
            raise ValueError(f"homography must be 3x3, got shape {homography.shape}")

        # Normalise detection key names: Go sends camelCase (frameIndex),
        # Python internally uses snake_case (frame_index).
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

        fitter = TrajectoryFitter(homography, fps)
        traj = fitter.fit(detections)

        if traj is None:
            return []

        return [traj.to_dict()]

    # ---- Internal ---------------------------------------------------------

    def _require_init(self) -> None:
        if not self._initialised:
            raise RuntimeError("ML modules not initialised — call __init__ first")

    def dispatch(self, method: str, params: dict) -> Any:
        """Route a method name to the appropriate handler."""
        handlers = {
            "__init__": self.rpc_init,
            "detect_batch": self.rpc_detect_batch,
            "tracknet_batch": self.rpc_tracknet_batch,
            "detect_court": self.rpc_detect_court,
            "classify_strokes": self.rpc_classify_strokes,
            "analyze_placements": self.rpc_analyze_placements,
            "segment_rallies": self.rpc_segment_rallies,
            "train": self.rpc_train,
            "fine_tune": self.rpc_fine_tune,
            "get_versions": self.rpc_get_versions,
            "rollback": self.rpc_rollback,
            "fit_trajectories": self.rpc_fit_trajectories,
            "set_background_reference": self.rpc_set_background_reference,
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
