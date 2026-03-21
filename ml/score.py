"""
Score Tracker — OCR-based scoreboard reading.

Transplanted from files/analysis/score_tracker.py.
Lazy-loads EasyOCR to avoid heavy init when scoring is unused.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ScoreTracker:
    """Read the on-screen scoreboard via OCR.

    EasyOCR is loaded lazily on first call to :meth:`read_score`.
    """

    def __init__(self, device: str = "cpu"):
        """
        Args:
            device: ``"cpu"`` or ``"cuda"``.  Passed to EasyOCR's
                    ``gpu`` flag (``True`` when ``device != "cpu"``).
        """
        self._reader: Optional[object] = None
        self._init_attempted: bool = False
        self._gpu = device != "cpu"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_score(self, frame: np.ndarray) -> Optional[dict]:
        """Attempt to read the score from a single frame.

        The scoreboard is assumed to be in the **top-right 30% x 15%**
        of the frame.

        Args:
            frame: BGR image.

        Returns:
            ``{"raw_text": str, "numbers": list[int]}`` when any text
            is detected, otherwise ``None``.
        """
        if self._reader is None and not self._init_attempted:
            self._init_reader()

        h, w = frame.shape[:2]
        x1 = int(w * 0.70)
        y1 = 0
        x2 = w
        y2 = int(h * 0.15)

        region = frame[y1:y2, x1:x2]
        if region.size == 0:
            return None

        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        try:
            results = self._reader.readtext(gray)  # type: ignore[union-attr]
        except Exception:
            logger.debug("OCR failed", exc_info=True)
            return None

        if not results:
            return None

        raw = " ".join(r[1] for r in results)
        numbers = [int(c) for c in raw if c.isdigit()]

        return {"raw_text": raw, "numbers": numbers}

    # ------------------------------------------------------------------
    # Lazy init
    # ------------------------------------------------------------------

    def _init_reader(self) -> None:
        self._init_attempted = True
        try:
            import easyocr

            self._reader = easyocr.Reader(["en"], gpu=self._gpu)
            logger.info("EasyOCR reader initialised (gpu=%s)", self._gpu)
        except ImportError:
            logger.error("easyocr is not installed — score tracking disabled")
        except Exception:
            logger.exception("Failed to initialise EasyOCR")
