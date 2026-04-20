"""Sparse-sample videos, run TrackNet on them, and emit a YOLO-format
dataset of ball-labeled frames that can be fed to `train_yolo_ball.py`.

Designed to run unattended overnight. Sample rate defaults to one frame
per 5 seconds, which for 50 hours of footage produces ~36k candidate
frames; of those, whatever fraction TrackNet is confident on becomes the
labeled training set (typically 15-25%, i.e. 5-9k frames).

Output layout (YOLO Ultralytics convention):

    <output-dir>/
        dataset.yaml
        images/train/*.jpg
        images/val/*.jpg
        labels/train/*.txt
        labels/val/*.txt

Each .txt is one line: `0 cx cy w h` (class 0 = tennis_ball, normalised).
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

# Allow `python ml/generate_pseudo_labels.py ...` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.tracknet import TrackNetDetector, BackgroundSubtractor

logger = logging.getLogger("pseudo_labels")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
BALL_BBOX_PX = 24        # fixed-size bbox around the detected ball centre
MIN_CONFIDENCE = 0.5     # TrackNet confidence floor
VAL_FRACTION = 0.2       # 80/20 train/val split
BATCH_SIZE = 16          # triplets per TrackNet call


def find_videos(input_dir: Path) -> list[Path]:
    videos: list[Path] = []
    for p in input_dir.rglob("*"):
        if p.suffix.lower() in VIDEO_EXTS:
            videos.append(p)
    videos.sort()
    return videos


def build_background_reference(cap: cv2.VideoCapture, total_frames: int) -> BackgroundSubtractor:
    """Sample 30 evenly-spaced frames to build TrackNet's background reference."""
    bg = BackgroundSubtractor()
    sample_count = min(30, total_frames)
    if sample_count == 0:
        return bg
    sample_idxs = np.linspace(0, max(total_frames - 1, 0), sample_count, dtype=int)
    frames = []
    for idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    if frames:
        bg.build_reference(frames)
    return bg


def read_triplet(cap: cv2.VideoCapture, centre_frame: int) -> list[np.ndarray] | None:
    """Fetch frames at indices [centre-4, centre-2, centre] (TrackNet convention)."""
    triplet: list[np.ndarray] = []
    for offset in (-4, -2, 0):
        idx = centre_frame + offset
        if idx < 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        triplet.append(frame)
    return triplet


def label_line(cx: float, cy: float, img_w: int, img_h: int) -> str:
    """YOLO label: class 0, normalised cx/cy/w/h."""
    bw = BALL_BBOX_PX / img_w
    bh = BALL_BBOX_PX / img_h
    return f"0 {cx / img_w:.6f} {cy / img_h:.6f} {bw:.6f} {bh:.6f}\n"


def process_video(
    video_path: Path,
    detector: TrackNetDetector,
    sample_every_seconds: float,
    output_dir: Path,
    max_labels: int | None,
    label_count_so_far: int,
) -> int:
    """Run TrackNet over a single video at the given sampling rate.

    Returns the number of labeled frames written for this video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("cannot open %s", video_path)
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(fps * sample_every_seconds)))
    logger.info("video %s: fps=%.1f frames=%d stride=%d", video_path.name, fps, total_frames, stride)

    if total_frames < 5:
        cap.release()
        return 0

    bg = build_background_reference(cap, total_frames)

    sample_indices = list(range(4, total_frames, stride))  # need idx >= 4 for triplet
    random.shuffle(sample_indices)  # shuffle so an early interrupt still diversifies

    # Process in batches of BATCH_SIZE for TrackNet efficiency
    written = 0
    batch_triplets: list[list[np.ndarray]] = []
    batch_originals: list[np.ndarray] = []
    batch_indices: list[int] = []

    def flush_batch() -> int:
        nonlocal batch_triplets, batch_originals, batch_indices
        if not batch_triplets:
            return 0
        detections = detector.detect_batch(batch_triplets)
        flushed = 0
        for det, original, idx in zip(detections, batch_originals, batch_indices):
            if det is None:
                continue
            conf = det.get("confidence", 0.0)
            if conf < MIN_CONFIDENCE:
                continue
            img_h, img_w = original.shape[:2]
            stem = f"{video_path.stem}_f{idx:07d}"
            split = "val" if random.random() < VAL_FRACTION else "train"
            img_path = output_dir / "images" / split / f"{stem}.jpg"
            lbl_path = output_dir / "labels" / split / f"{stem}.txt"
            cv2.imwrite(str(img_path), original, [cv2.IMWRITE_JPEG_QUALITY, 90])
            lbl_path.write_text(label_line(det["x"], det["y"], img_w, img_h))
            flushed += 1
        batch_triplets = []
        batch_originals = []
        batch_indices = []
        return flushed

    for idx in sample_indices:
        if max_labels is not None and (label_count_so_far + written) >= max_labels:
            break
        triplet_raw = read_triplet(cap, idx)
        if triplet_raw is None:
            continue

        # Apply background subtraction per TrackNet's expected input
        if bg.is_ready():
            triplet = [bg.subtract(f) for f in triplet_raw]
        else:
            triplet = triplet_raw

        batch_triplets.append(triplet)
        batch_originals.append(triplet_raw[2])  # save the "current" frame (N), not subtracted
        batch_indices.append(idx)

        if len(batch_triplets) >= BATCH_SIZE:
            written += flush_batch()

    written += flush_batch()
    cap.release()
    logger.info("video %s: wrote %d labeled frames", video_path.name, written)
    return written


def write_dataset_yaml(output_dir: Path) -> None:
    yaml_text = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: tennis_ball\n"
    )
    (output_dir / "dataset.yaml").write_text(yaml_text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Folder containing tagged videos (recursed)")
    parser.add_argument("--output-dir", required=True, help="Destination for YOLO dataset")
    parser.add_argument("--tracknet-weights", required=True, help="Path to TrackNet .pt weights")
    parser.add_argument("--sample-every-seconds", type=float, default=5.0,
                        help="Seconds between sampled frames (default: 5)")
    parser.add_argument("--max-labels", type=int, default=None,
                        help="Stop after this many labeled frames (default: no cap)")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.is_dir():
        logger.error("input dir does not exist: %s", input_dir)
        return 2

    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_dir)
    if not videos:
        logger.error("no videos found under %s", input_dir)
        return 2
    logger.info("found %d videos", len(videos))

    detector = TrackNetDetector(model_path=args.tracknet_weights, device=args.device)
    if detector.model is None:
        logger.error("TrackNet failed to initialise — aborting")
        return 3

    total_written = 0
    for video in videos:
        try:
            total_written += process_video(
                video_path=video,
                detector=detector,
                sample_every_seconds=args.sample_every_seconds,
                output_dir=output_dir,
                max_labels=args.max_labels,
                label_count_so_far=total_written,
            )
        except Exception:
            logger.exception("failed processing %s — continuing", video)
        logger.info("running total: %d labeled frames", total_written)
        if args.max_labels is not None and total_written >= args.max_labels:
            logger.info("hit max-labels cap (%d)", args.max_labels)
            break

    write_dataset_yaml(output_dir)
    logger.info("done. %d labels across %s", total_written, output_dir)
    logger.info("dataset yaml: %s", output_dir / "dataset.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
