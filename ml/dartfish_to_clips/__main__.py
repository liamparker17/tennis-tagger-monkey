import argparse, sys
from pathlib import Path
from .extract import process_match

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("dartfish_to_clips")
    ap.add_argument("pairs_root", type=Path, help="files/data/training_pairs")
    ap.add_argument("--out", type=Path, default=Path("files/data/clips"))
    ap.add_argument("--only", default=None, help="substring filter on match folder name")
    args = ap.parse_args(argv)

    matches = [d for d in sorted(args.pairs_root.iterdir()) if d.is_dir()]
    if args.only:
        matches = [d for d in matches if args.only in d.name]
    total = 0
    for d in matches:
        vids = list(d.glob("*.mp4")); csvs = list(d.glob("*.csv"))
        if not vids or not csvs:
            print(f"skip {d.name}: missing video or csv"); continue
        n = process_match(vids[0], csvs[0], args.out, match_name=d.name)
        print(f"{d.name}: {n} clips"); total += n
    print(f"TOTAL: {total}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
