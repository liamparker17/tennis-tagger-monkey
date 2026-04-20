import json, subprocess, sys
from pathlib import Path
import pytest

CKPT = Path("files/models/point_model/run0/best.pt")
CLIP = Path("files/data/clips/_smoke/p_0001.mp4")


@pytest.mark.skipif(not CKPT.exists() or not CLIP.exists(), reason="needs trained ckpt + clip")
def test_predict_point(tmp_path):
    p = subprocess.Popen(
        [sys.executable, "-m", "ml.inference_server.server", "--ckpt", str(CKPT)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    req = json.dumps({"jsonrpc":"2.0","id":1,"method":"predict_point",
                       "params":{"clip_path": str(CLIP), "point_index": 0}}) + "\n"
    p.stdin.write(req.encode()); p.stdin.flush()
    line = p.stdout.readline()
    p.stdin.close(); p.wait(timeout=30)
    resp = json.loads(line)
    assert "result" in resp and "strokes" in resp["result"]
