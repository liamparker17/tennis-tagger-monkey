import argparse, json, sys
from pathlib import Path
from .process import process_clip
from .pose import PoseExtractor
from .ball import BallDetector
from .schema import SCHEMA_VERSION

def main(argv=None) -> int:
    ap = argparse.ArgumentParser("feature_extractor")
    ap.add_argument("clips_root", type=Path, default=Path("files/data/clips"))
    ap.add_argument("--out", type=Path, default=Path("files/data/features"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    pose = PoseExtractor(Path("models/yolov8s-pose.pt"))
    ball = BallDetector(Path("models/yolov8s.pt"))

    matches = [d for d in sorted(args.clips_root.iterdir()) if d.is_dir()]
    if args.only: matches = [d for d in matches if args.only in d.name]
    for m in matches:
        out_dir = args.out / m.name
        out_dir.mkdir(parents=True, exist_ok=True)
        index = []
        for clip in sorted(m.glob("p_*.mp4")):
            out_npz = out_dir / (clip.stem + ".npz")
            if out_npz.exists() and not args.force:
                index.append(clip.stem); continue
            try:
                process_clip(clip, out_npz, pose=pose, ball=ball)
                index.append(clip.stem)
                print(f"  {m.name}/{clip.stem} ok")
            except Exception as e:
                print(f"  {m.name}/{clip.stem} FAIL: {e}")
        (out_dir / "index.json").write_text(json.dumps({
            "schema": SCHEMA_VERSION, "match": m.name, "clips": index
        }, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
