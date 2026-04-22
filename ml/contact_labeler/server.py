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


def _find_setup_json(match_dir_name: str) -> dict | None:
    """Try to locate the match's preflight setup.json (near/far player names,
    side_swaps). Looked up under files/data/training_pairs/<match>/ which is
    where the clipper stages the source video."""
    candidates = [
        Path("files/data/training_pairs") / match_dir_name,
    ]
    for root in candidates:
        if not root.is_dir(): continue
        for p in root.glob("*.setup.json"):
            try: return json.loads(p.read_text())
            except Exception: continue
    return None


def _near_is_player_a_at_time(setup: dict | None, player_a: str,
                              global_time_s: float, fps: float) -> bool:
    """Near-court side flips on each `side_swaps` frame. Return True if, at
    global_time_s of the full match, the near-court player is player_a."""
    if not setup: return True  # unknown — default to near=P1
    near_name = str(setup.get("players", {}).get("near", "")).strip().lower()
    pa = player_a.strip().lower()
    near_is_pa_initial = (near_name == pa) if near_name else True
    swaps = list(setup.get("side_swaps", []))
    swaps_before = sum(1 for f in swaps if f / max(fps, 1.0) <= global_time_s)
    return near_is_pa_initial if (swaps_before % 2 == 0) else not near_is_pa_initial


def _clip_info(clips_root: Path, match: str, clip: str) -> dict:
    """Return everything the labeler UI needs to show a clip confidently:
    existing events, the two player names, which is near/far at this clip's
    time, Dartfish stroke hints."""
    out: dict = {"clip": clip, "events": [], "player_a": "", "player_b": "",
                 "near_name": "", "far_name": "", "near_is_p1": True,
                 "stroke_types": [], "stroke_count_lo": 0, "stroke_count_hi": 0}
    match_dir = clips_root / match
    # Existing events
    cl = match_dir / "contact_labels.json"
    if cl.exists():
        try:
            doc = json.loads(cl.read_text())
            out["events"] = list(doc.get("clips", {}).get(clip, []))
        except Exception: pass
    # Dartfish point metadata
    lbl = match_dir / "labels.json"
    fps = 30.0
    if lbl.exists():
        try:
            doc = json.loads(lbl.read_text())
            for p in doc.get("points", []):
                if Path(p.get("clip", "")).stem != clip: continue
                out["player_a"] = p.get("player_a", "")
                out["player_b"] = p.get("player_b", "")
                out["stroke_types"] = list(p.get("stroke_types", []))
                out["stroke_count_lo"] = int(p.get("stroke_count_lo", 0))
                out["stroke_count_hi"] = int(p.get("stroke_count_hi", 0))
                clip_start = float(p.get("clip_start_s", 0.0))
                setup = _find_setup_json(match)
                out["near_is_p1"] = _near_is_player_a_at_time(
                    setup, out["player_a"], clip_start, fps)
                if setup:
                    out["near_name"] = setup.get("players", {}).get("near", "")
                    out["far_name"] = setup.get("players", {}).get("far", "")
                break
        except Exception: pass
    return out

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
            if self.path.startswith("/api/clip_info/"):
                _, _, _, match, clip = self.path.split("/", 4)
                return self._send_json(200, _clip_info(clips_root, match, clip))
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
