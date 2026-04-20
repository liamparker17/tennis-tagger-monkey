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
