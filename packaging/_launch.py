"""Launcher wrapper for Tennis Tagger.

The desktop shortcut runs `pythonw.exe _launch.py`, which detaches from any
console. That means anything written to stderr — including uncaught import
errors before the Tk mainloop starts — vanishes by default, making silent
failures look like "the shortcut does nothing."

This wrapper redirects stdout/stderr to ``launcher.log`` next to the install
and execs ``tagger_ui.py``. Any future top-level crash leaves a traceback
on disk instead of disappearing.
"""

from __future__ import annotations

import os
import sys
import traceback
from datetime import datetime, timezone


def _open_log() -> object:
    app_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(app_dir, "launcher.log")
    log = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
    log.write(f"\n--- launch {datetime.now(timezone.utc).isoformat()} ---\n")
    return log


def main() -> int:
    log = _open_log()
    sys.stdout = log
    sys.stderr = log

    app_dir = os.path.dirname(os.path.abspath(__file__))
    target = os.path.join(app_dir, "tagger_ui.py")

    try:
        with open(target, encoding="utf-8") as f:
            source = f.read()
        sys.argv[0] = target
        exec(
            compile(source, target, "exec"),
            {"__name__": "__main__", "__file__": target},
        )
        return 0
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 0
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
