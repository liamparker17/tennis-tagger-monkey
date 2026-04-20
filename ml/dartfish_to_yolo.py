"""Convert human-tagged Dartfish match data into a YOLO training set.

For every `<video>.mp4` + `<video>.csv` pair under the input folder, this
script:

1. Loads (or auto-detects) court corners so we can invert the homography
   and project Dartfish's 2D court-map coordinates back into pixel space.
2. Parses the CSV row-by-row (one row = one point).
3. For each point, pulls the serve bounce (XY Deuce / XY Ad) and the last
   shot bounce (XY Last Shot) plus their estimated video frames.
4. Seeks those frames, projects the court XY to pixels, and writes
   YOLO-format labels for a fine-tuning dataset.

These labels are **human-verified**, not TrackNet pseudo-labels. Two
labels per point × several hundred points per match = thousands of
high-quality labels per night. Ball detection trained on this should be
dramatically better than TrackNet.

Run:
    python ml/dartfish_to_yolo.py --pairs-dir files/data/training_pairs \
                                  --output-dir files/data/yolo_training
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.analyzer import Analyzer  # noqa: E402

logger = logging.getLogger("dartfish_to_yolo")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
BALL_BBOX_PX = 24
VAL_FRACTION = 0.2

# YOLO class IDs for the dataset we emit.
CLS_BALL = 0
CLS_PLAYER_NEAR = 1
CLS_PLAYER_FAR = 2

# Path to the pretrained YOLO used for auto-labeling players.
PLAYER_AUTO_MODEL = Path(__file__).resolve().parent.parent / "models" / "yolov8s.pt"

# COCO "person" class id.
COCO_PERSON_CLASS = 0
PLAYER_CONF_THRESH = 0.4
# Max players per frame (exclude ball boys / linesmen / umpire).
MAX_PLAYERS_PER_FRAME = 2
# Allow player feet to sit this many pixels outside the court polygon
# (shots pull players wide, but linesmen are much further out).
COURT_MARGIN_PX = 120

# Dartfish half-court XY scale. Observed values in sample (e.g. "113;74")
# sit well under 200. Assumption: 0..200 on each axis covers the half of
# the court where the bounce landed. If projected pixels look wrong after
# a training run, this is the first thing to tune — print a few labeled
# frames and compare where the dot lands vs. where the ball actually is.
DARTFISH_XY_SCALE = 200.0

# Column indices in the real Dartfish CSV. See header reference.
COL_POSITION = 1     # "MM:SS.mmm" — start of the point in the video
COL_DURATION = 2     # "MM:SS.mmm" — length of the point
COL_XY_DEUCE = 85    # Serve bounce, deuce court
COL_XY_AD = 97       # Serve bounce, ad court
COL_XY_LAST = 99     # Last shot bounce

# Seconds after the point's start when the serve bounce happens.
# Serve toss → contact → bounce is roughly 1.2s on most clips.
SERVE_BOUNCE_OFFSET_S = 1.2
# Seconds before the point's end when the last-shot bounce happens.
LAST_SHOT_BOUNCE_OFFSET_S = 0.3


def parse_timestamp(s: str) -> Optional[float]:
    """Dartfish Position/Duration is 'MM:SS.mmm'. Accept numbers too."""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    m = re.match(r"^(\d+):(\d{1,2})(?:\.(\d+))?$", s)
    if m:
        minutes = int(m.group(1))
        secs = int(m.group(2))
        frac = m.group(3) or "0"
        return minutes * 60 + secs + float("0." + frac)
    try:
        return float(s)
    except ValueError:
        return None


def parse_xy(s: str) -> Optional[tuple[float, float]]:
    """Parse Dartfish 'x;y' (or 'x,y') half-court coordinate."""
    if not s:
        return None
    s = s.strip()
    if not s:
        return None
    sep = ";" if ";" in s else ("," if "," in s else None)
    if sep is None:
        return None
    try:
        x_str, y_str = s.split(sep, 1)
        x = float(x_str.strip())
        y = float(y_str.strip())
    except (ValueError, IndexError):
        return None
    return x / DARTFISH_XY_SCALE, y / DARTFISH_XY_SCALE


def load_homography(video_path: Path, first_frame: np.ndarray) -> Optional[dict]:
    """Prefer a pre-flight sidecar; fall back to the Analyzer's auto
    court detection on the supplied frame. Returns a court dict with a
    'homography_inv' key (maps normalised court -> pixel)."""
    sidecar = Path(str(video_path) + ".setup.json")
    analyzer = Analyzer()

    native_h, native_w = first_frame.shape[:2]
    side_swaps: list[int] = []

    if sidecar.is_file():
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8"))
            corners = [[c["x"], c["y"]] for c in data["court_corners_pixel"]]
            analyzer.set_manual_court(
                corners_pixel=corners,
                frame_width=int(data.get("frame_width", native_w)),
                frame_height=int(data.get("frame_height", native_h)),
            )
            side_swaps = sorted(int(f) for f in data.get("side_swaps", []))
            source = "sidecar"
        except Exception as exc:  # noqa: BLE001
            logger.warning("Sidecar unreadable (%s) — falling back to auto-detect: %s",
                           sidecar.name, exc)
            sidecar = None  # fall through

    if not sidecar or not sidecar.is_file():
        court = analyzer.detect_court(first_frame)
        confidence = court.get("confidence", 0.0)
        if confidence < 0.5:
            return None
        source = "auto"

    court = analyzer.detect_court(first_frame)
    H = np.array(court["homography"], dtype=np.float64)
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None

    # Our homography is in 640x360 detection space. Convert the inverse
    # result back to native pixels.
    det_w, det_h = 640.0, 360.0
    sx = native_w / det_w
    sy = native_h / det_h
    scale_to_native = np.diag([sx, sy, 1.0])

    H_inv_to_native = scale_to_native @ H_inv

    # Project the four normalised court corners into native pixels so we
    # can filter persons by "is the player inside the court?".
    corners_norm = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=np.float64)
    corners_native = (H_inv_to_native @ corners_norm.T).T
    corners_native = corners_native[:, :2] / corners_native[:, 2:3]

    # Far side = small court-y ≈ top of image. Near side = large court-y.
    # Split line in pixel space is the horizontal midpoint of the court quad.
    y_split = float(np.mean(corners_native[:, 1]))

    return {
        "source": source,
        "H_inv_to_native": H_inv_to_native,
        "native_w": native_w,
        "native_h": native_h,
        "polygon": corners_native.astype(np.float32),
        "y_split": y_split,
        "side_swaps": side_swaps,
    }


def players_swapped(frame_idx: int, swaps: list[int]) -> bool:
    """After an odd number of swaps, near-side identity flips with far-side."""
    return sum(1 for s in swaps if frame_idx >= s) % 2 == 1


def project_court_to_pixel(cx: float, cy: float,
                           court: dict) -> tuple[float, float]:
    """Map normalised court [0,1] coords to native pixel coordinates."""
    pt = np.array([cx, cy, 1.0])
    out = court["H_inv_to_native"] @ pt
    out /= out[2]
    return float(out[0]), float(out[1])


def yolo_label(cx_px: float, cy_px: float, w: int, h: int,
               box_w_px: float = BALL_BBOX_PX, box_h_px: Optional[float] = None,
               cls: int = CLS_BALL) -> Optional[str]:
    """Return a YOLO-format label line, or None if the point is off-frame."""
    if not (0 <= cx_px < w and 0 <= cy_px < h):
        return None
    if box_h_px is None:
        box_h_px = box_w_px
    return (f"{cls} {cx_px / w:.6f} {cy_px / h:.6f} "
            f"{box_w_px / w:.6f} {box_h_px / h:.6f}\n")


_player_model_cache: dict[str, object] = {}


def _load_player_model():
    """Lazy-load YOLO person detector, cached across calls."""
    if "model" in _player_model_cache:
        return _player_model_cache["model"]
    try:
        from ultralytics import YOLO  # noqa: WPS433
    except ImportError:
        logger.warning("ultralytics not installed — skipping player auto-labels")
        _player_model_cache["model"] = None
        return None
    if not PLAYER_AUTO_MODEL.is_file():
        logger.warning("player model missing at %s — skipping player auto-labels",
                       PLAYER_AUTO_MODEL)
        _player_model_cache["model"] = None
        return None
    _player_model_cache["model"] = YOLO(str(PLAYER_AUTO_MODEL))
    return _player_model_cache["model"]


def detect_players(frame: np.ndarray, court: dict,
                   frame_idx: int = 0) -> list[tuple[int, float, float, float, float]]:
    """Run YOLO person detection, filter to players on court, and assign
    near/far based on the court split line.

    Returns a list of (cls, cx_px, cy_px, box_w_px, box_h_px).
    """
    model = _load_player_model()
    if model is None:
        return []

    results = model.predict(frame, classes=[COCO_PERSON_CLASS],
                            conf=PLAYER_CONF_THRESH, verbose=False)
    if not results:
        return []

    polygon = court["polygon"]
    y_split = court["y_split"]

    candidates = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        conf = float(box.conf[0].item())
        # Feet = bottom-center of the detection.
        foot_x = (x1 + x2) / 2.0
        foot_y = y2
        dist = cv2.pointPolygonTest(polygon, (foot_x, foot_y), True)
        if dist < -COURT_MARGIN_PX:
            continue
        candidates.append((conf, x1, y1, x2, y2, foot_x, foot_y))

    # Keep the top-N by confidence — excludes linesmen / ball kids.
    candidates.sort(key=lambda c: c[0], reverse=True)
    candidates = candidates[:MAX_PLAYERS_PER_FRAME]

    swapped = players_swapped(frame_idx, court.get("side_swaps", []))
    labels = []
    for _conf, x1, y1, x2, y2, foot_x, foot_y in candidates:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = x2 - x1
        bh = y2 - y1
        # Positional near/far by court half, flipped if players have swapped ends.
        on_near_side = foot_y > y_split
        if swapped:
            on_near_side = not on_near_side
        cls = CLS_PLAYER_NEAR if on_near_side else CLS_PLAYER_FAR
        labels.append((cls, cx, cy, bw, bh))
    return labels


def process_pair(
    video_path: Path,
    csv_path: Path,
    output_dir: Path,
    split_rng: random.Random,
) -> int:
    logger.info("▶ %s", video_path.name)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("  cannot open video, skipping")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Seed frame for court detection — middle of the match (past intros,
    # before hand-over interviews).
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    ok, seed_frame = cap.read()
    if not ok or seed_frame is None:
        cap.release()
        logger.warning("  could not read a seed frame")
        return 0

    court = load_homography(video_path, seed_frame)
    if court is None:
        cap.release()
        logger.warning("  no homography available (no sidecar, auto-detect failed) — skipping")
        return 0
    logger.info("  court homography: %s", court["source"])

    labels_written = 0
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            cap.release()
            return 0

        for row_idx, row in enumerate(reader, start=2):
            if len(row) < max(COL_XY_DEUCE, COL_XY_AD, COL_XY_LAST) + 1:
                continue

            pos_s = parse_timestamp(row[COL_POSITION])
            dur_s = parse_timestamp(row[COL_DURATION])
            if pos_s is None:
                continue

            # Serve bounce — could be either deuce or ad; pick whichever is filled.
            serve_xy = parse_xy(row[COL_XY_DEUCE]) or parse_xy(row[COL_XY_AD])
            last_xy = parse_xy(row[COL_XY_LAST])

            candidates: list[tuple[str, float, tuple[float, float]]] = []
            if serve_xy is not None:
                candidates.append(("serve", pos_s + SERVE_BOUNCE_OFFSET_S, serve_xy))
            if last_xy is not None and dur_s is not None and dur_s > 0:
                candidates.append(("last",
                                   pos_s + max(0.0, dur_s - LAST_SHOT_BOUNCE_OFFSET_S),
                                   last_xy))

            for kind, t_sec, (cx, cy) in candidates:
                frame_idx = int(round(t_sec * fps))
                if not (0 <= frame_idx < total_frames):
                    continue
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                h, w = frame.shape[:2]
                px, py = project_court_to_pixel(cx, cy, court)
                ball_line = yolo_label(px, py, w, h, cls=CLS_BALL)
                if ball_line is None:
                    continue

                lines = [ball_line]
                for cls, cpx, cpy, bw, bh in detect_players(frame, court, frame_idx):
                    pl_line = yolo_label(cpx, cpy, w, h,
                                         box_w_px=bw, box_h_px=bh, cls=cls)
                    if pl_line is not None:
                        lines.append(pl_line)

                stem = f"{video_path.stem}_r{row_idx:04d}_{kind}"
                split = "val" if split_rng.random() < VAL_FRACTION else "train"
                img_path = output_dir / "images" / split / f"{stem}.jpg"
                lbl_path = output_dir / "labels" / split / f"{stem}.txt"
                cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                lbl_path.write_text("".join(lines))
                labels_written += 1

    cap.release()
    logger.info("  wrote %d labels", labels_written)
    return labels_written


def find_pairs(pairs_dir: Path) -> list[tuple[Path, Path]]:
    """Every video with a sibling CSV becomes a pair."""
    pairs: list[tuple[Path, Path]] = []
    for video in sorted(pairs_dir.rglob("*")):
        if video.suffix.lower() not in VIDEO_EXTS:
            continue
        # Prefer <stem>.csv in the same folder; otherwise take the first CSV there.
        candidates = [video.with_suffix(".csv")] + sorted(video.parent.glob("*.csv"))
        csv_path = next((c for c in candidates if c.is_file()), None)
        if csv_path is None:
            logger.warning("No CSV for %s — skipping", video.name)
            continue
        pairs.append((video, csv_path))
    return pairs


def write_dataset_yaml(output_dir: Path) -> None:
    yaml = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: tennis_ball\n"
        "  1: player_near\n"
        "  2: player_far\n"
    )
    (output_dir / "dataset.yaml").write_text(yaml)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs-dir", required=True,
                        help="Folder containing <match>/<video>.mp4 + <match>/<video>.csv pairs")
    parser.add_argument("--output-dir", required=True,
                        help="Destination for the YOLO dataset")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    output_dir = Path(args.output_dir)
    if not pairs_dir.is_dir():
        logger.error("pairs dir does not exist: %s", pairs_dir)
        return 2

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(pairs_dir)
    if not pairs:
        logger.error("no video+CSV pairs found under %s", pairs_dir)
        return 2
    logger.info("found %d pair(s)", len(pairs))

    rng = random.Random(args.seed)
    total = 0
    for video, csv_path in pairs:
        try:
            total += process_pair(video, csv_path, output_dir, rng)
        except Exception:  # noqa: BLE001
            logger.exception("failed on %s — continuing", video)

    write_dataset_yaml(output_dir)
    logger.info("done — %d total labels across %s", total, output_dir)
    logger.info("next step: python ml/train_yolo_ball.py --dataset %s",
                (output_dir / "dataset.yaml").resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
