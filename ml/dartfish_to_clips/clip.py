from __future__ import annotations

import subprocess
from pathlib import Path

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

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
        "-ss",
        f"{max(0.0, start_s):.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration_s:.3f}",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(out),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {r.stderr}")


def window_seconds(start_ms: int, duration_ms: int) -> tuple[float, float]:
    s = max(0.0, start_ms / 1000.0 - PAD_BEFORE_S)
    d = duration_ms / 1000.0 + PAD_BEFORE_S + PAD_AFTER_S
    return s, d
