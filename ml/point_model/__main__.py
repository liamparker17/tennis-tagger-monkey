import argparse, json, sys
from pathlib import Path
from .train import train, TrainConfig
from .eval import evaluate
from .eval_dartfish import evaluate as evaluate_dartfish, _print_human

try:
    from ml._telemetry import init_telemetry
    init_telemetry("point_model")
except Exception:
    pass

def main(argv=None) -> int:
    ap = argparse.ArgumentParser("point_model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--clips", type=Path, default=Path("files/data/clips"))
    pt.add_argument("--features", type=Path, default=Path("files/data/features"))
    pt.add_argument("--out", type=Path, default=Path("files/models/point_model/run0"))
    pt.add_argument("--epochs", type=int, default=20)
    pt.add_argument("--batch-size", type=int, default=8)
    pt.add_argument("--resume", type=Path, default=None,
                    help="Path to a last.pt/best.pt to resume from")

    pe = sub.add_parser("eval")
    pe.add_argument("--ckpt", type=Path, required=True)
    pe.add_argument("--clips", type=Path, default=Path("files/data/clips"))
    pe.add_argument("--features", type=Path, default=Path("files/data/features"))

    pd = sub.add_parser("eval-dartfish",
                        help="row-level eval against labels.json: stroke/outcome/won-by/n_shots")
    pd.add_argument("--ckpt", type=Path, required=True)
    pd.add_argument("--clips", type=Path, default=Path("files/data/clips"))
    pd.add_argument("--features", type=Path, default=Path("files/data/features"))
    pd.add_argument("--out", type=Path, default=None,
                    help="optional path for full per-clip dump")

    args = ap.parse_args(argv)
    if args.cmd == "train":
        r = train(TrainConfig(args.clips, args.features, args.out,
                              epochs=args.epochs, batch_size=args.batch_size,
                              resume=args.resume))
        print(json.dumps(r["best_val"]))
    elif args.cmd == "eval":
        r = evaluate(args.ckpt, args.clips, args.features)
        print(json.dumps(r, indent=2))
    elif args.cmd == "eval-dartfish":
        s = evaluate_dartfish(args.ckpt, args.clips, args.features, args.out)
        _print_human(s)
    return 0

if __name__ == "__main__": sys.exit(main())
