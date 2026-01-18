from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from luminalink.types import DetectedObject


class ObjectDetector(ABC):
    """Abstract object detector interface."""

    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> list[DetectedObject]:
        """Return detected objects for a single frame."""


class NoopObjectDetector(ObjectDetector):
    """A lightweight detector that returns no objects (for scaffolding)."""

    def detect(self, frame_bgr: np.ndarray) -> list[DetectedObject]:
        """Return an empty list."""

        return []

