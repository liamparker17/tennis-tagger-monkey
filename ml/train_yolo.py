"""Fine-tune a YOLOv8 detector on a preflight-derived single-class dataset.

Use for both the `player` and `ball` corpora produced by
`ml.labels_to_yolo` / `ml.ball_labels_to_yolo`.

Run:

    # Ball detector, shared corpus, 50 epochs starting from yolov8s.pt
    python -m ml.train_yolo \\
        --data files/data/yolo_ball/_shared/data.yaml \\
        --base models/yolov8s.pt \\
        --epochs 50 \\
        --out files/models/yolo_ball

    # Player detector
    python -m ml.train_yolo \\
        --data files/data/yolo_player/_shared/data.yaml \\
        --base models/yolov8s.pt \\
        --out files/models/yolo_player

The trained `best.pt` is copied to `<out>/best.pt` so the feature extractor
can pick it up with a fixed path.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser("train_yolo")
    ap.add_argument("--data", type=Path, required=True,
                    help="path to data.yaml produced by labels_to_yolo")
    ap.add_argument("--base", type=Path, default=Path("models/yolov8s.pt"),
                    help="base weights to fine-tune from")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=960,
                    help="training image size; 960 preserves small-object detail")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True,
                    help="where to copy best.pt once training finishes")
    ap.add_argument("--project", type=Path, default=Path("files/data/yolo_runs"))
    ap.add_argument("--name", type=str, default="train")
    args = ap.parse_args()

    if not args.data.is_file():
        print(f"data.yaml not found: {args.data}", file=sys.stderr)
        return 2
    if not args.base.is_file():
        print(f"base weights not found: {args.base}", file=sys.stderr)
        return 2

    from ultralytics import YOLO

    model = YOLO(str(args.base))
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
    )

    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else \
        args.project / args.name
    src = run_dir / "weights" / "best.pt"
    if not src.is_file():
        print(f"training finished but best.pt not found at {src}", file=sys.stderr)
        return 3

    args.out.mkdir(parents=True, exist_ok=True)
    dst = args.out / "best.pt"
    shutil.copy2(src, dst)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
