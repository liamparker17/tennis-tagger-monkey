"""Centralised crash reporting via Sentry.

Idempotent — every entry point can call init_telemetry() at import time;
only the first call actually configures Sentry. If no DSN is configured
(env var TENNIS_TAGGER_SENTRY_DSN unset and no sentry.dsn file next to
the bundle), Sentry is a no-op and the app behaves exactly as before.

End users can opt out by setting TENNIS_TAGGER_TELEMETRY=off.

DSN resolution order (first hit wins):
  1. TENNIS_TAGGER_SENTRY_DSN env var
  2. <bundle-root>/sentry.dsn (written by build_bundle.ps1 -SentryDsn)
  3. <repo-root>/sentry.dsn (for source runs only)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Optional

# Module-level guard so multiple init() calls are cheap no-ops.
_initialised = False


def _resolve_dsn() -> Optional[str]:
    dsn = os.environ.get("TENNIS_TAGGER_SENTRY_DSN", "").strip()
    if dsn:
        return dsn
    # Walk up from this file looking for sentry.dsn — covers both bundled
    # installs (file sits next to ml/) and source runs (file at repo root).
    here = Path(__file__).resolve().parent
    for parent in [here, *here.parents]:
        candidate = parent / "sentry.dsn"
        if candidate.is_file():
            text = candidate.read_text(encoding="utf-8").strip()
            if text:
                return text
    return None


def _scrub_event(event, hint):
    """Strip likely-PII before send: usernames in paths, video filenames.

    Match videos contain players' names in the filename (e.g. 'Christopher
    Eubanks vs Jesper de-Jong 2025 Wimbledon R1.mp4'). Stack traces include
    the path of the user's home dir. We don't need any of that to debug
    crashes — keep stack frames but mask user identity.
    """
    home = os.path.expanduser("~")
    user = os.path.basename(home)

    def _scrub_str(s: str) -> str:
        if not isinstance(s, str): return s
        out = s
        if home and home in out: out = out.replace(home, "<HOME>")
        if user and len(user) > 2 and user in out:
            out = out.replace(user, "<USER>")
        return out

    def _walk(o):
        if isinstance(o, dict):
            return {k: _walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_walk(v) for v in o]
        if isinstance(o, str):
            return _scrub_str(o)
        return o

    return _walk(event)


def init_telemetry(component: str, release: Optional[str] = None) -> bool:
    """Initialise Sentry once per process. Returns True if Sentry is active.

    `component` is the entry-point name (e.g. "tagger_ui", "preflight",
    "bridge_server", "point_model") so we can filter crashes by which
    process died.
    """
    global _initialised
    if _initialised:
        return True

    if os.environ.get("TENNIS_TAGGER_TELEMETRY", "").lower() == "off":
        return False

    dsn = _resolve_dsn()
    if not dsn:
        return False

    try:
        import sentry_sdk
    except ImportError:
        # Sentry SDK not installed — silently skip. Lets the source-tree
        # run work without bundling sentry-sdk in dev environments.
        return False

    if release is None:
        release = os.environ.get("TENNIS_TAGGER_VERSION", "dev")

    try:
        sentry_sdk.init(
            dsn=dsn,
            release=f"tennis-tagger@{release}",
            environment=os.environ.get("TENNIS_TAGGER_ENV", "production"),
            send_default_pii=False,
            traces_sample_rate=0.0,  # crashes only, no perf overhead
            before_send=_scrub_event,
            attach_stacktrace=True,
            max_breadcrumbs=50,
        )
        sentry_sdk.set_tag("component", component)
        sentry_sdk.set_tag("python_version", sys.version.split()[0])
        sentry_sdk.set_tag("platform", sys.platform)
    except Exception:
        # Never let telemetry init kill the app.
        return False

    _initialised = True
    return True


def capture_exception(exc: BaseException) -> None:
    """Manual exception capture for handled but noteworthy errors.
    No-op if Sentry isn't initialised."""
    if not _initialised:
        return
    try:
        import sentry_sdk
        sentry_sdk.capture_exception(exc)
    except Exception:
        pass
