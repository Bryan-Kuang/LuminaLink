from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SceneChangeConfig:
    """Configuration for histogram-based scene change detection."""

    hist_correlation_threshold: float
    warmup_frames: int


class HistogramSceneChangeDetector:
    """Detect scene changes using HSV histogram correlation."""

    def __init__(self, cfg: SceneChangeConfig):
        self._cfg = cfg
        self._prev_hist: np.ndarray | None = None
        self._seen = 0

    def is_scene_change(self, frame_bgr: np.ndarray) -> bool:
        """Return True when the current frame likely starts a new scene."""

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        self._seen += 1
        if self._prev_hist is None:
            self._prev_hist = hist
            return False

        corr = float(cv2.compareHist(self._prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_CORREL))
        self._prev_hist = hist

        if self._seen <= self._cfg.warmup_frames:
            return False
        return corr < self._cfg.hist_correlation_threshold

