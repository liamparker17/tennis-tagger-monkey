from __future__ import annotations

import subprocess
from pathlib import Path

PAD_BEFORE_S = 1.0
PAD_AFTER_S = 1.0


def cut_clip(src: Path, start_s: float, duration_s: float, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ss",
        f"{max(0.0, start_s):.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-b:a",
        "96k",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr}")


def window_seconds(start_ms: int, duration_ms: int) -> tuple[float, float]:
    s = max(0.0, start_ms / 1000.0 - PAD_BEFORE_S)
    d = duration_ms / 1000.0 + PAD_BEFORE_S + PAD_AFTER_S
    return s, d
