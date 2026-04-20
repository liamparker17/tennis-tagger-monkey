# Per-Frame Feature Extractor (+ Contact-Frame Labeler) — Plan 2

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Two parallel workstreams that share Plan 1's clip output:

- **2A — Feature extractor** (Python). For each clip, cache pose, court homography, audio mel-spec, and sparse ball detections to one `.npz`.
- **2B — Contact-frame labeler** (browser tool). Tiny stdlib HTML+JS app for hand-labeling per-shot contact frames into `contact_labels.json`. Runs against the same `files/data/clips/` tree.

Both consume Plan 1 output. Neither blocks the other. Both feed Plan 3.

**Tech stack:** Python 3.11, ultralytics YOLOv8s-pose, librosa (new dep), numpy, OpenCV; vanilla HTML/JS served by `python -m http.server`.

---

## File layout

```
ml/feature_extractor/
  __init__.py
  schema.py        # FeatureSet dataclass + SCHEMA_VERSION
  frames.py        # ffmpeg frame iterator
  pose.py          # YOLOv8s-pose wrapper, near/far slot assignment
  court.py         # call existing court analyzer once per clip
  audio.py         # librosa log-mel
  ball.py          # sparse YOLO-ball wrapper
  process.py       # per-clip orchestrator, .npz writer
  inspect.py       # CLI: dump shapes/stats for one clip
  __main__.py      # CLI: walk files/data/clips, write features

ml/contact_labeler/
  index.html       # single-page UI (no framework)
  app.js
  server.py        # tiny stdlib HTTP server: GET clips, POST labels
  README.md        # how to run

ml/tests/
  test_feat_schema.py
  test_feat_pose.py
  test_feat_audio.py
  test_feat_process.py
  test_contact_server.py
```

Outputs:
```
files/data/features/<match>/p_0001.npz
files/data/features/<match>/index.json
files/data/clips/<match>/contact_labels.json
```

---

## 2A — Feature extractor

### Constants

```python
SCHEMA_VERSION = 1
FPS = 30                 # resample to 30
POSE_MODEL = "models/yolov8s-pose.pt"
BALL_MODEL = "models/yolov8s.pt"
AUDIO_SR = 22050
AUDIO_N_MELS = 64
AUDIO_HOP = 512
```

### Task 2A.1 — `FeatureSet` schema + npz I/O

**Files:** create `ml/feature_extractor/schema.py`, `ml/tests/test_feat_schema.py`.

- [ ] Test:

```python
import numpy as np
from pathlib import Path
from ml.feature_extractor.schema import FeatureSet, save_npz, load_npz, SCHEMA_VERSION

def test_roundtrip(tmp_path: Path):
    fs = FeatureSet(
        schema=SCHEMA_VERSION, fps=30,
        pose_px=np.zeros((10, 2, 17, 2), np.float32),
        pose_conf=np.zeros((10, 2, 17), np.float32),
        pose_court=np.zeros((10, 2, 17, 2), np.float32),
        pose_valid=np.zeros((10, 2), np.bool_),
        court_h=np.eye(3, dtype=np.float32),
        ball=np.zeros((10, 3), np.float32),
        audio_mel=np.zeros((64, 200), np.float32),
        clip_meta=np.zeros((4,), np.float32),
    )
    p = tmp_path / "c.npz"; save_npz(p, fs)
    fs2 = load_npz(p)
    assert fs2.schema == SCHEMA_VERSION
    assert fs2.pose_px.shape == (10, 2, 17, 2)
```

- [ ] Implement:

```python
# ml/feature_extractor/schema.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

SCHEMA_VERSION = 1

@dataclass
class FeatureSet:
    schema: int
    fps: int
    pose_px: np.ndarray      # (T, 2, 17, 2) float32 — keypoint pixels
    pose_conf: np.ndarray    # (T, 2, 17)    float32
    pose_court: np.ndarray   # (T, 2, 17, 2) float32 — keypoint in court coords
    pose_valid: np.ndarray   # (T, 2)        bool
    court_h: np.ndarray      # (3, 3)        float32 — homography image→court
    ball: np.ndarray         # (T, 3)        float32 — (x, y, conf), 0 where missing
    audio_mel: np.ndarray    # (n_mels, F)   float32 — log-mel
    clip_meta: np.ndarray    # (4,)          float32 — [duration_s, w, h, fps_native]

def save_npz(path: Path, fs: FeatureSet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in asdict(fs).items() if isinstance(v, np.ndarray)},
                        schema=np.int32(fs.schema), fps=np.int32(fs.fps))

def load_npz(path: Path) -> FeatureSet:
    z = np.load(path)
    return FeatureSet(
        schema=int(z["schema"]), fps=int(z["fps"]),
        pose_px=z["pose_px"], pose_conf=z["pose_conf"],
        pose_court=z["pose_court"], pose_valid=z["pose_valid"],
        court_h=z["court_h"], ball=z["ball"],
        audio_mel=z["audio_mel"], clip_meta=z["clip_meta"],
    )
```

- [ ] Test passes. Commit: `feat(plan2): FeatureSet schema + npz I/O`.

---

### Task 2A.2 — Frame iterator

**Files:** create `ml/feature_extractor/frames.py`. Reuse existing reader if `internal/video` has a Python equivalent; otherwise call ffmpeg.

- [ ] Implement:

```python
# ml/feature_extractor/frames.py
from __future__ import annotations
import subprocess
from pathlib import Path
import numpy as np

def probe_wh(path: Path) -> tuple[int, int, float]:
    r = subprocess.run([
        "ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=width,height,r_frame_rate",
        "-of","default=nw=1:nk=1", str(path)
    ], capture_output=True, text=True, check=True)
    w, h, rate = r.stdout.strip().splitlines()
    num, den = rate.split("/")
    fps = float(num) / float(den) if float(den) != 0 else float(num)
    return int(w), int(h), fps

def iter_frames(path: Path, fps: int = 30):
    w, h, _ = probe_wh(path)
    proc = subprocess.Popen([
        "ffmpeg","-nostdin","-loglevel","error",
        "-i", str(path),
        "-vf", f"fps={fps}",
        "-f","rawvideo","-pix_fmt","bgr24","-"
    ], stdout=subprocess.PIPE)
    bytes_per_frame = w * h * 3
    while True:
        buf = proc.stdout.read(bytes_per_frame)
        if not buf or len(buf) < bytes_per_frame: break
        yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.stdout.close(); proc.wait()
```

No test — covered by 2A.6 integration. Commit: `feat(plan2): ffmpeg frame iterator`.

---

### Task 2A.3 — Pose extractor

**Files:** create `ml/feature_extractor/pose.py`, `ml/tests/test_feat_pose.py`.

Slot semantics: `[:, 0]=near-side`, `[:, 1]=far-side`. Assignment: per frame, take the two highest-confidence person detections, sort by mean keypoint y descending (larger y = closer to bottom of frame = near-side).

- [ ] Test (skip if model file missing):

```python
import pytest, numpy as np
from pathlib import Path
from ml.feature_extractor.pose import PoseExtractor

POSE = Path("models/yolov8s-pose.pt")
SAMPLE = Path("testdata/sample_a.mp4")

@pytest.mark.skipif(not POSE.exists() or not SAMPLE.exists(), reason="needs model + sample")
def test_pose_shape():
    ext = PoseExtractor(POSE)
    frame = np.zeros((720, 1280, 3), np.uint8)
    px, conf = ext.extract(frame)
    assert px.shape == (2, 17, 2) and conf.shape == (2, 17)
```

- [ ] Implement:

```python
# ml/feature_extractor/pose.py
from __future__ import annotations
from pathlib import Path
import numpy as np

class PoseExtractor:
    def __init__(self, weights: Path):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))

    def extract(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # returns (2,17,2) px and (2,17) conf; zeros where slot empty
        r = self.model.predict(frame_bgr, verbose=False)[0]
        px = np.zeros((2, 17, 2), np.float32)
        conf = np.zeros((2, 17), np.float32)
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return px, conf
        kp_xy = r.keypoints.xy.cpu().numpy()         # (N,17,2)
        kp_c  = r.keypoints.conf.cpu().numpy()        # (N,17)
        det_c = r.boxes.conf.cpu().numpy()            # (N,)
        order = np.argsort(-det_c)[:2]
        sel_xy = kp_xy[order]; sel_c = kp_c[order]
        # near-side = larger mean y
        ys = sel_xy[..., 1].mean(axis=1)
        if len(ys) == 2 and ys[0] < ys[1]:
            sel_xy = sel_xy[::-1]; sel_c = sel_c[::-1]
        for i in range(len(sel_xy)):
            px[i] = sel_xy[i]; conf[i] = sel_c[i]
        return px, conf
```

- [ ] Test passes/skips. Commit: `feat(plan2): pose extractor with near/far slot assignment`.

---

### Task 2A.4 — Court extractor

**Files:** create `ml/feature_extractor/court.py`. Wrap existing `ml/analyzer.py` court detection — call once on the middle frame of the clip, return 3×3 homography (image → court). Fallback to identity when detection fails (record in `pose_valid` later by checking `det(H) != 0` is unreliable; just trust caller).

- [ ] Implement:

```python
# ml/feature_extractor/court.py
from __future__ import annotations
import numpy as np

def estimate_homography(middle_frame_bgr: np.ndarray) -> np.ndarray:
    try:
        from ml.analyzer import detect_court_homography  # existing entry point
    except Exception:
        return np.eye(3, dtype=np.float32)
    try:
        H = detect_court_homography(middle_frame_bgr)
        if H is None: return np.eye(3, dtype=np.float32)
        return H.astype(np.float32)
    except Exception:
        return np.eye(3, dtype=np.float32)

def project_to_court(px: np.ndarray, H: np.ndarray) -> np.ndarray:
    # px: (..., 2). Returns same shape projected through H.
    flat = px.reshape(-1, 2)
    ones = np.ones((flat.shape[0], 1), np.float32)
    homo = np.concatenate([flat, ones], axis=1)
    proj = (H @ homo.T).T
    proj = proj[:, :2] / np.clip(proj[:, 2:3], 1e-6, None)
    return proj.reshape(px.shape)
```

If `ml/analyzer.py` doesn't expose `detect_court_homography`, add a thin wrapper there that calls the existing entry point. Commit: `feat(plan2): court homography extractor`.

---

### Task 2A.5 — Audio extractor

**Files:** create `ml/feature_extractor/audio.py`, `ml/tests/test_feat_audio.py`.

Add `librosa` to `requirements.txt` if not present.

- [ ] Test:

```python
import shutil, pytest, numpy as np
from pathlib import Path
from ml.feature_extractor.audio import extract_log_mel

SAMPLE = Path("testdata/sample_a.mp4")

@pytest.mark.skipif(not SAMPLE.exists() or shutil.which("ffmpeg") is None, reason="needs sample+ffmpeg")
def test_audio_shape(tmp_path):
    mel = extract_log_mel(SAMPLE, tmp_dir=tmp_path)
    assert mel.ndim == 2 and mel.shape[0] == 64 and mel.shape[1] > 0
    assert mel.dtype == np.float32
```

- [ ] Implement:

```python
# ml/feature_extractor/audio.py
from __future__ import annotations
import subprocess, tempfile
from pathlib import Path
import numpy as np

SR = 22050; N_MELS = 64; HOP = 512

def extract_log_mel(video: Path, tmp_dir: Path | None = None) -> np.ndarray:
    import librosa
    tmp = Path(tempfile.mkdtemp() if tmp_dir is None else tmp_dir)
    wav = tmp / (video.stem + ".wav")
    subprocess.run([
        "ffmpeg","-nostdin","-y","-loglevel","error",
        "-i", str(video), "-ac","1","-ar", str(SR), str(wav)
    ], check=True)
    y, _ = librosa.load(wav, sr=SR, mono=True)
    if y.size == 0: return np.zeros((N_MELS, 1), np.float32)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_mels=N_MELS, hop_length=HOP)
    return librosa.power_to_db(mel + 1e-10).astype(np.float32)
```

- [ ] Test passes/skips. Commit: `feat(plan2): log-mel audio extractor`.

---

### Task 2A.6 — Ball detector wrapper

**Files:** create `ml/feature_extractor/ball.py`. Wrap existing YOLO ball detection (`models/yolov8s.pt`). Return `(T, 3)` array of `(x, y, conf)` — zeros where no detection.

- [ ] Implement:

```python
# ml/feature_extractor/ball.py
from __future__ import annotations
from pathlib import Path
import numpy as np

class BallDetector:
    def __init__(self, weights: Path, ball_class_id: int = 32):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
        self.cls_id = ball_class_id

    def detect(self, frame_bgr: np.ndarray) -> tuple[float, float, float]:
        r = self.model.predict(frame_bgr, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0: return (0.0, 0.0, 0.0)
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        mask = cls == self.cls_id
        if not mask.any(): return (0.0, 0.0, 0.0)
        i = np.argmax(np.where(mask, conf, -1))
        x = (xyxy[i, 0] + xyxy[i, 2]) / 2; y = (xyxy[i, 1] + xyxy[i, 3]) / 2
        return (float(x), float(y), float(conf[i]))
```

If the project's YOLO is the unified 3-class model (per memory `project_tracknet_batching_attempts.md`), set `ball_class_id` to whatever index that model uses. Confirm by reading `ml/detector.py` first. Commit: `feat(plan2): ball detector wrapper`.

---

### Task 2A.7 — Per-clip orchestrator

**Files:** create `ml/feature_extractor/process.py`, `ml/tests/test_feat_process.py`.

- [ ] Test (heavy — gated):

```python
import os, pytest, json
from pathlib import Path
from ml.feature_extractor.process import process_clip

CLIP = Path("files/data/clips/_smoke/p_0001.mp4")

@pytest.mark.skipif(not CLIP.exists(), reason="needs Plan 1 smoke output")
def test_process_clip(tmp_path):
    out = tmp_path / "p.npz"
    process_clip(CLIP, out, fps=30)
    assert out.exists()
```

- [ ] Implement:

```python
# ml/feature_extractor/process.py
from __future__ import annotations
from pathlib import Path
import numpy as np
from .schema import FeatureSet, save_npz, SCHEMA_VERSION
from .frames import iter_frames, probe_wh
from .pose import PoseExtractor
from .court import estimate_homography, project_to_court
from .audio import extract_log_mel
from .ball import BallDetector

POSE_MODEL = Path("models/yolov8s-pose.pt")
BALL_MODEL = Path("models/yolov8s.pt")

def process_clip(clip: Path, out_npz: Path, fps: int = 30,
                 pose: PoseExtractor | None = None,
                 ball: BallDetector | None = None) -> None:
    pose = pose or PoseExtractor(POSE_MODEL)
    ball = ball or BallDetector(BALL_MODEL)

    w, h, native_fps = probe_wh(clip)
    frames = list(iter_frames(clip, fps=fps))
    T = len(frames)
    pose_px = np.zeros((T, 2, 17, 2), np.float32)
    pose_conf = np.zeros((T, 2, 17), np.float32)
    ball_arr = np.zeros((T, 3), np.float32)
    for t, fr in enumerate(frames):
        pose_px[t], pose_conf[t] = pose.extract(fr)
        ball_arr[t] = ball.detect(fr)
    H = estimate_homography(frames[T // 2]) if T else np.eye(3, np.float32)
    pose_court = project_to_court(pose_px, H)
    pose_valid = pose_conf.max(axis=-1) > 0.2  # (T, 2)
    mel = extract_log_mel(clip)
    duration_s = T / float(fps)
    fs = FeatureSet(
        schema=SCHEMA_VERSION, fps=fps,
        pose_px=pose_px, pose_conf=pose_conf,
        pose_court=pose_court, pose_valid=pose_valid,
        court_h=H.astype(np.float32), ball=ball_arr, audio_mel=mel,
        clip_meta=np.array([duration_s, w, h, native_fps], np.float32),
    )
    save_npz(out_npz, fs)
```

- [ ] Test passes/skips. Commit: `feat(plan2): per-clip feature orchestrator`.

---

### Task 2A.8 — CLI: walk clips → features

**Files:** create `ml/feature_extractor/__main__.py`, `ml/feature_extractor/__init__.py`.

- [ ] Implement:

```python
# ml/feature_extractor/__main__.py
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
```

- [ ] Smoke run: `python -m ml.feature_extractor files/data/clips --only "Marcos Giron"`.
- [ ] Commit: `feat(plan2): feature extractor CLI`.

---

### Task 2A.9 — Inspect tool

**Files:** create `ml/feature_extractor/inspect.py`.

- [ ] Implement:

```python
# ml/feature_extractor/inspect.py
import sys
from pathlib import Path
from .schema import load_npz

def main():
    fs = load_npz(Path(sys.argv[1]))
    print(f"schema={fs.schema} fps={fs.fps}")
    print(f"pose_px={fs.pose_px.shape} pose_conf mean={fs.pose_conf.mean():.3f}")
    print(f"pose_valid per-slot frac={fs.pose_valid.mean(axis=0)}")
    print(f"ball detections={int((fs.ball[:,2] > 0).sum())}/{len(fs.ball)}")
    print(f"court_h det={float(__import__('numpy').linalg.det(fs.court_h)):.3f}")
    print(f"audio_mel={fs.audio_mel.shape}")
    print(f"meta duration_s/w/h/native_fps={fs.clip_meta.tolist()}")

if __name__ == "__main__": main()
```

Smoke: `python -m ml.feature_extractor.inspect files/data/features/<match>/p_0001.npz`. Commit: `feat(plan2): feature inspector`.

---

## 2B — Contact-frame labeler

**Output schema** (matches Plan 3's `strong_contact_frames` integration point):

```json
{
  "schema": 1,
  "match": "<match name>",
  "clips": {
    "p_0001": [
      {"frame": 12, "hitter": 0},
      {"frame": 47, "hitter": 1}
    ]
  }
}
```

`hitter`: 0 = near-side player, 1 = far-side. `frame` is the 30-fps frame index inside the clip (matches feature extractor's resampled timeline).

### Task 2B.1 — Stdlib HTTP server

**Files:** create `ml/contact_labeler/server.py`, `ml/contact_labeler/README.md`, `ml/tests/test_contact_server.py`.

- [ ] Test:

```python
import json, threading, http.client, time
from pathlib import Path
from ml.contact_labeler.server import build_server

def test_get_clips_and_post_labels(tmp_path):
    (tmp_path / "match_a").mkdir()
    (tmp_path / "match_a" / "p_0001.mp4").write_bytes(b"")
    (tmp_path / "match_a" / "p_0002.mp4").write_bytes(b"")
    srv = build_server(("127.0.0.1", 0), clips_root=tmp_path)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True); t.start()
    try:
        c = http.client.HTTPConnection("127.0.0.1", port)
        c.request("GET", "/api/matches"); r = c.getresponse()
        assert r.status == 200
        data = json.loads(r.read())
        assert "match_a" in [m["name"] for m in data["matches"]]

        body = json.dumps({"clip": "p_0001", "events": [{"frame": 10, "hitter": 0}]})
        c.request("POST", "/api/labels/match_a", body=body,
                  headers={"Content-Type": "application/json"})
        assert c.getresponse().status == 200
        saved = json.loads((tmp_path / "match_a" / "contact_labels.json").read_text())
        assert saved["clips"]["p_0001"][0]["frame"] == 10
    finally:
        srv.shutdown()
```

- [ ] Implement:

```python
# ml/contact_labeler/server.py
from __future__ import annotations
import json, mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCHEMA_VERSION = 1
STATIC_DIR = Path(__file__).parent

def _load_or_init(path: Path, match_name: str) -> dict:
    if path.exists():
        try: return json.loads(path.read_text())
        except Exception: pass
    return {"schema": SCHEMA_VERSION, "match": match_name, "clips": {}}

def build_server(address, clips_root: Path):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a, **k): pass

        def _send_json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)

        def do_GET(self):
            if self.path == "/" or self.path == "/index.html":
                return self._send_static(STATIC_DIR / "index.html")
            if self.path == "/app.js":
                return self._send_static(STATIC_DIR / "app.js")
            if self.path == "/api/matches":
                matches = []
                for d in sorted(clips_root.iterdir()):
                    if not d.is_dir(): continue
                    clips = sorted(p.stem for p in d.glob("p_*.mp4"))
                    labels = _load_or_init(d / "contact_labels.json", d.name)
                    matches.append({
                        "name": d.name, "clips": clips,
                        "labeled": list(labels["clips"].keys()),
                    })
                return self._send_json(200, {"matches": matches})
            if self.path.startswith("/clip/"):
                _, _, match, name = self.path.split("/", 3)
                p = clips_root / match / (name)
                return self._send_static(p)
            self.send_error(404)

        def do_POST(self):
            if not self.path.startswith("/api/labels/"):
                return self.send_error(404)
            match = self.path.rsplit("/", 1)[-1]
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            clip = body["clip"]; events = body["events"]
            label_path = clips_root / match / "contact_labels.json"
            doc = _load_or_init(label_path, match)
            doc["clips"][clip] = events
            label_path.write_text(json.dumps(doc, indent=2))
            return self._send_json(200, {"ok": True})

        def _send_static(self, p: Path):
            if not p.exists(): return self.send_error(404)
            ctype, _ = mimetypes.guess_type(p.name)
            data = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)

    return ThreadingHTTPServer(address, H)

def main():
    import argparse
    ap = argparse.ArgumentParser("contact_labeler")
    ap.add_argument("--clips", type=Path, default=Path("files/data/clips"))
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    srv = build_server(("127.0.0.1", args.port), clips_root=args.clips)
    print(f"http://127.0.0.1:{args.port}")
    srv.serve_forever()

if __name__ == "__main__": main()
```

- [ ] Test passes. Commit: `feat(plan2): contact labeler stdlib server`.

---

### Task 2B.2 — Single-page UI

**Files:** create `ml/contact_labeler/index.html`, `ml/contact_labeler/app.js`.

Behavior:
- Sidebar: matches → clips. Clip is "labeled" if it has any events in `contact_labels.json`.
- Main: `<video>` element, `<canvas>` overlay (optional, leave empty for now).
- Keyboard: `,` step back 1 frame, `.` step forward 1 frame, `Space` play/pause, `n` mark contact for near-side, `f` mark contact for far-side, `u` undo last event, `s` save (POST), `]` next clip.
- Frame index = `Math.round(video.currentTime * 30)`.

- [ ] `index.html`:

```html
<!doctype html>
<html><head><meta charset="utf-8"><title>Contact Labeler</title>
<style>
  body{margin:0;font-family:sans-serif;display:grid;grid-template-columns:240px 1fr;height:100vh}
  #side{overflow:auto;border-right:1px solid #ccc;padding:8px;font-size:13px}
  #side .m{font-weight:bold;margin-top:8px}
  #side .c{padding:2px 6px;cursor:pointer}
  #side .c.done{color:green}
  #side .c.active{background:#def}
  #main{display:flex;flex-direction:column}
  video{width:100%;background:#000;flex:1;min-height:0}
  #bar{padding:6px;border-top:1px solid #ccc;font-family:monospace;font-size:12px}
  #events{padding:6px;border-top:1px solid #ccc;max-height:120px;overflow:auto;font-family:monospace;font-size:12px}
</style></head>
<body>
  <div id="side"></div>
  <div id="main">
    <video id="v" controls></video>
    <div id="bar"></div>
    <div id="events"></div>
  </div>
  <script src="/app.js"></script>
</body></html>
```

- [ ] `app.js`:

```javascript
const FPS = 30;
let state = { matches: [], match: null, clip: null, events: [] };

async function loadMatches() {
  const r = await fetch("/api/matches");
  state.matches = (await r.json()).matches;
  renderSide();
}

function renderSide() {
  const el = document.getElementById("side");
  el.innerHTML = "";
  for (const m of state.matches) {
    const h = document.createElement("div"); h.className = "m"; h.textContent = m.name; el.appendChild(h);
    for (const c of m.clips) {
      const d = document.createElement("div"); d.className = "c";
      if (m.labeled.includes(c)) d.classList.add("done");
      if (state.match === m.name && state.clip === c) d.classList.add("active");
      d.textContent = c;
      d.onclick = () => loadClip(m.name, c);
      el.appendChild(d);
    }
  }
}

function loadClip(match, clip) {
  state.match = match; state.clip = clip; state.events = [];
  document.getElementById("v").src = `/clip/${match}/${clip}.mp4`;
  renderSide(); renderEvents();
}

function frame() { return Math.round(document.getElementById("v").currentTime * FPS); }
function step(d) { document.getElementById("v").currentTime = Math.max(0, frame()/FPS + d/FPS); }

function mark(hitter) {
  state.events.push({ frame: frame(), hitter });
  renderEvents();
}

function undo() { state.events.pop(); renderEvents(); }

async function save() {
  if (!state.match || !state.clip) return;
  await fetch(`/api/labels/${state.match}`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ clip: state.clip, events: state.events }),
  });
  await loadMatches();
}

function nextClip() {
  const m = state.matches.find(x => x.name === state.match);
  if (!m) return;
  const i = m.clips.indexOf(state.clip);
  if (i >= 0 && i + 1 < m.clips.length) loadClip(m.name, m.clips[i+1]);
}

function renderEvents() {
  document.getElementById("bar").textContent =
    `${state.match || "-"} / ${state.clip || "-"}  frame=${frame()}  events=${state.events.length}`;
  document.getElementById("events").innerHTML =
    state.events.map(e => `f=${e.frame} hitter=${e.hitter}`).join("<br>");
}

document.addEventListener("keydown", (e) => {
  const v = document.getElementById("v");
  if (e.key === ",") { v.pause(); step(-1); }
  else if (e.key === ".") { v.pause(); step(1); }
  else if (e.key === " ") { e.preventDefault(); v.paused ? v.play() : v.pause(); }
  else if (e.key === "n") mark(0);
  else if (e.key === "f") mark(1);
  else if (e.key === "u") undo();
  else if (e.key === "s") save();
  else if (e.key === "]") nextClip();
});
document.getElementById("v").addEventListener("timeupdate", renderEvents);

loadMatches();
```

- [ ] `README.md`:

```markdown
# Contact Labeler

Run:
    python -m ml.contact_labeler.server --clips files/data/clips

Open http://127.0.0.1:8765

Keys: `,` `.` step frame · Space play/pause · `n` near-side hit · `f` far-side hit
      `u` undo · `s` save · `]` next clip
```

- [ ] Manual smoke: launch server, open page, label one clip, refresh page, confirm clip shows green and `contact_labels.json` is on disk.
- [ ] Commit: `feat(plan2): contact labeler UI`.

---

## Done when

**2A:**
- `python -m ml.feature_extractor files/data/clips` runs over all matches; `.npz` files appear under `files/data/features/<match>/`; reruns with no `--force` are no-ops on existing files.
- `python -m ml.feature_extractor.inspect <path>.npz` prints non-degenerate shapes/stats.
- `pytest ml/tests/test_feat_*.py -v` is green (skipped tests OK where samples/models missing).

**2B:**
- `python -m ml.contact_labeler.server` serves the page, lists matches and clips, supports keyboard labeling, persists to `<match>/contact_labels.json`.
- `pytest ml/tests/test_contact_server.py -v` is green.

Both feed Plan 3 — `contact_labels.json` plugs into `build_targets(strong_contact_frames=...)`; `.npz` files are loaded by Plan 3's `Dataset`.
