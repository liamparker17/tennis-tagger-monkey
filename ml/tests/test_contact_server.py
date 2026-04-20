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
