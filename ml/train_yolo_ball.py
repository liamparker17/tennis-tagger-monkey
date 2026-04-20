"""Fine-tune yolo12n on a tennis-ball YOLO dataset produced by
`generate_pseudo_labels.py`. Saves the best checkpoint to
`models/yolo12n_ball.pt`.

Designed to run unattended overnight on a single GPU. Defaults are tuned
for RTX 2050-class hardware (4 GB VRAM): small batch, 640 imgsz, AMP on.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

logger = logging.getLogger("train_ball")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    default_models = repo_root / "models"

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="Path to dataset.yaml (from generate_pseudo_labels.py)")
    parser.add_argument("--base-weights", default=str(default_models / "yolo12n.pt"),
                        help="Starting weights (default: models/yolo12n.pt)")
    parser.add_argument("--output", default=str(default_models / "yolo12n_ball.pt"),
                        help="Destination for the best checkpoint")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (8 is safe for 4 GB VRAM; raise if you have more)")
    parser.add_argument("--device", default="0", help='"0" for CUDA, "cpu" for CPU')
    parser.add_argument("--name", default="yolo12n_ball",
                        help="Ultralytics run name (output lands under runs/detect/<name>)")
    parser.add_argument("--patience", type=int, default=15,
                        help="Early-stop patience (epochs without val improvement)")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    if not dataset.is_file():
        logger.error("dataset yaml not found: %s", dataset)
        return 2

    base_weights = Path(args.base_weights)
    if not base_weights.is_file():
        logger.error("base weights not found: %s", base_weights)
        logger.error("run: python -c \"from ultralytics import YOLO; YOLO('yolo12n.pt')\" then move the file to models/")
        return 2

    logger.info("loading %s", base_weights)
    model = YOLO(str(base_weights))

    logger.info("training: dataset=%s epochs=%d imgsz=%d batch=%d device=%s",
                dataset, args.epochs, args.imgsz, args.batch, args.device)
    results = model.train(
        data=str(dataset),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        amp=True,
        patience=args.patience,
        name=args.name,
        exist_ok=True,
        verbose=True,
    )

    # Ultralytics writes the best checkpoint to runs/detect/<name>/weights/best.pt
    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("runs/detect") / args.name
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        logger.error("training finished but best.pt was not produced at %s", best)
        return 3

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, output)
    logger.info("copied best checkpoint to %s", output)
    logger.info("done. swap ml/bridge_server.py detector_models 'yolo' entry to %s to use it.", output.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
