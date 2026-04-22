"""Convert user-drawn player bounding boxes (from preflight's step-6
labeling UI) into a single-class YOLO training set.

Each match's setup sidecar stores `player_labels` as a list of
`{frame, time_ms, boxes: [{x1,y1,x2,y2}, ...]}` in ORIGINAL pixel coords.
This script:

1. Reads every `<video>.setup.json` under --pairs-dir.
2. For each labeled frame, seeks the video and writes a JPG.
3. Writes the YOLO label file — one line per box, class 0 (`player`).
4. Maintains a per-match subset AND appends into a shared corpus dir so
   both per-match fine-tunes and a cross-match baseline can train off
   the same labels.

Output layout (under --output-dir):

    <match>/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
      data.yaml
    _shared/
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt
      data.yaml

Run:

    python -m ml.labels_to_yolo \\
        --pairs-dir files/data/training_pairs \\
        --output-dir files/data/yolo_player

Re-runs are idempotent: existing images/labels for a (match, frame) pair
are overwritten so edits in preflight propagate without manual cleanup.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import cv2

logger = logging.getLogger("labels_to_yolo")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v"}
PLAYER_CLASS_ID = 0  # single class "player"
VAL_FRACTION = 0.15


def _slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _split_for(frame_key: str) -> str:
    """Deterministic train/val split keyed off the match+frame name, so
    re-runs keep the same split without a random seed file."""
    h = abs(hash(frame_key)) % 1000
    return "val" if h < int(VAL_FRACTION * 1000) else "train"


def _find_setup_files(pairs_dir: Path) -> list[tuple[Path, Path]]:
    """Return (setup_json, video) pairs for every match with player_labels."""
    out: list[tuple[Path, Path]] = []
    for sub in sorted(pairs_dir.iterdir()):
        if not sub.is_dir(): continue
        for setup in sub.glob("*.setup.json"):
            video_name = setup.name[:-len(".setup.json")]
            video = sub / video_name
            if not video.is_file() or video.suffix.lower() not in VIDEO_EXTS:
                logger.warning("skip: no video next to %s", setup.name)
                continue
            out.append((setup, video))
    return out


def _load_labels(setup_path: Path) -> list[dict]:
    try:
        doc = json.loads(setup_path.read_text())
    except Exception as e:
        logger.warning("cannot read %s: %s", setup_path.name, e)
        return []
    labels = doc.get("player_labels") or []
    # Defensive: filter empty frames and malformed entries.
    clean: list[dict] = []
    for entry in labels:
        try:
            frame = int(entry["frame"])
            boxes = entry.get("boxes") or []
            normed = []
            for b in boxes:
                x1 = float(b["x1"]); y1 = float(b["y1"])
                x2 = float(b["x2"]); y2 = float(b["y2"])
                if x2 - x1 < 2 or y2 - y1 < 2: continue
                normed.append((x1, y1, x2, y2))
            if not normed: continue
            clean.append({"frame": frame, "boxes": normed})
        except Exception:
            continue
    return clean


def _yolo_line(x1: float, y1: float, x2: float, y2: float,
               w: int, h: int) -> str:
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = max(0.0, (x2 - x1)) / w
    bh = max(0.0, (y2 - y1)) / h
    cx = min(max(cx, 0.0), 1.0); cy = min(max(cy, 0.0), 1.0)
    bw = min(bw, 1.0);            bh = min(bh, 1.0)
    return f"{PLAYER_CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _ensure_tree(root: Path) -> None:
    for split in ("train", "val"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_data_yaml(root: Path) -> None:
    yaml = (
        f"path: {root.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [player]\n"
    )
    (root / "data.yaml").write_text(yaml)


def _export_frame(video: cv2.VideoCapture, frame_idx: int,
                  boxes: list[tuple[float, float, float, float]],
                  img_path: Path, lbl_path: Path) -> bool:
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = video.read()
    if not ok or frame is None:
        return False
    h, w = frame.shape[:2]
    cv2.imwrite(str(img_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    lines = [_yolo_line(*b, w=w, h=h) for b in boxes]
    lbl_path.write_text("\n".join(lines) + "\n")
    return True


def process_match(setup_path: Path, video_path: Path,
                  per_match_root: Path, shared_root: Path) -> tuple[int, int]:
    labels = _load_labels(setup_path)
    if not labels:
        logger.info("no player_labels in %s — skipping", setup_path.name)
        return 0, 0

    match_name = video_path.parent.name
    slug = _slug(match_name)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error("cannot open video %s", video_path)
        return 0, 0

    _ensure_tree(per_match_root)
    _ensure_tree(shared_root)
    written = 0
    skipped = 0
    for entry in labels:
        frame = entry["frame"]
        boxes = entry["boxes"]
        stem = f"{slug}__f{frame:07d}"
        split = _split_for(stem)

        img = per_match_root / "images" / split / f"{stem}.jpg"
        lbl = per_match_root / "labels" / split / f"{stem}.txt"
        ok = _export_frame(cap, frame, boxes, img, lbl)
        if not ok:
            logger.warning("could not read frame %d in %s", frame, video_path.name)
            skipped += 1; continue

        # Mirror into shared corpus.
        img_shared = shared_root / "images" / split / f"{stem}.jpg"
        lbl_shared = shared_root / "labels" / split / f"{stem}.txt"
        img_shared.write_bytes(img.read_bytes())
        lbl_shared.write_text(lbl.read_text())
        written += 1

    cap.release()
    _write_data_yaml(per_match_root)
    _write_data_yaml(shared_root)
    logger.info("%s: %d frames written, %d skipped", match_name, written, skipped)
    return written, skipped


def main() -> int:
    ap = argparse.ArgumentParser("labels_to_yolo")
    ap.add_argument("--pairs-dir", type=Path,
                    default=Path("files/data/training_pairs"))
    ap.add_argument("--output-dir", type=Path,
                    default=Path("files/data/yolo_player"))
    args = ap.parse_args()

    if not args.pairs_dir.is_dir():
        logger.error("pairs-dir not found: %s", args.pairs_dir)
        return 2

    shared = args.output_dir / "_shared"
    pairs = _find_setup_files(args.pairs_dir)
    if not pairs:
        logger.error("no matches with setup.json found under %s", args.pairs_dir)
        return 2

    grand_written = 0
    for setup_path, video in pairs:
        per_match = args.output_dir / _slug(video.parent.name)
        w, _ = process_match(setup_path, video, per_match, shared)
        grand_written += w

    logger.info("done. %d labeled frames written across %d matches.",
                grand_written, len(pairs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
