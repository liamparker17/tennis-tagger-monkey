from __future__ import annotations
import argparse, json, sys, tempfile
from dataclasses import asdict
from pathlib import Path
from .predict import Predictor

try:
    from ml._telemetry import init_telemetry
    init_telemetry("inference_server")
except Exception:
    pass


def _to_dict(pred):
    return {
        "contact_frames": pred.contact_frames,
        "bounce_frames": pred.bounce_frames,
        "strokes": [asdict(s) for s in pred.strokes],
        "outcome": pred.outcome,
        "outcome_prob": pred.outcome_prob,
        "low_confidence": pred.low_confidence,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("inference_server")
    ap.add_argument("--ckpt", type=Path, required=True)
    args = ap.parse_args(argv)

    pred = Predictor(args.ckpt)
    tmp = Path(tempfile.mkdtemp())

    for line in sys.stdin:
        line = line.strip()
        if not line: continue
        req = None
        try:
            req = json.loads(line)
            method = req.get("method"); rid = req.get("id")
            if method == "predict_point":
                clip = Path(req["params"]["clip_path"])
                fp = pred.predict_clip(clip, tmp)
                resp = {"jsonrpc":"2.0","id":rid,"result": _to_dict(fp)}
            elif method == "ping":
                resp = {"jsonrpc":"2.0","id":rid,"result":"pong"}
            elif method == "shutdown":
                print(json.dumps({"jsonrpc":"2.0","id":rid,"result":"bye"}), flush=True)
                return 0
            else:
                resp = {"jsonrpc":"2.0","id":rid,"error":{"code":-32601,"message":"no such method"}}
        except Exception as e:
            resp = {"jsonrpc":"2.0","id":req.get("id") if req else None,
                    "error":{"code":-32000,"message":str(e)}}
        print(json.dumps(resp), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
