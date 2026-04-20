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
