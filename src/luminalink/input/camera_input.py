from __future__ import annotations

import logging
import time
from collections.abc import Iterator

import cv2

from ..types import VideoFrame

logger = logging.getLogger(__name__)


class CameraInput:
    """Capture video frames from a camera device in real-time."""

    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._fps = fps
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._start_time: float | None = None

    @property
    def is_opened(self) -> bool:
        """Check if camera is currently opened."""
        return self._cap is not None and self._cap.isOpened()

    def open(self) -> None:
        """Open the camera device and configure capture parameters."""

        self._cap = cv2.VideoCapture(self._camera_index)
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open camera device: {self._camera_index}")

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap.set(cv2.CAP_PROP_FPS, self._fps)

        actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self._cap.get(cv2.CAP_PROP_FPS))

        logger.info(f"Camera opened: {actual_width}x{actual_height} @ {actual_fps} FPS")

        if self._start_time is None:
            self._start_time = time.time()

    def reconnect(self) -> bool:
        """Try to reconnect to the camera after a disconnection.

        Returns:
            True if reconnected successfully, False otherwise.
        """
        # Release old capture if it exists
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

        try:
            self._cap = cv2.VideoCapture(self._camera_index)
            if not self._cap.isOpened():
                self._cap = None
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)

            logger.info(f"Camera reconnected: device {self._camera_index}")
            return True
        except Exception as e:
            logger.error(f"Camera reconnect failed: {e}")
            self._cap = None
            return False

    def frames(self) -> Iterator[VideoFrame]:
        """Yield decoded frames with system clock timestamps."""

        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Camera not opened. Call open() first.")

        while self._cap is not None:
            ok, frame = self._cap.read()
            if not ok:
                break

            pts_ms = int((time.time() - self._start_time) * 1000)
            yield VideoFrame(pts_ms=pts_ms, frame_index=self._frame_index, image_bgr=frame)
            self._frame_index += 1

    def close(self) -> None:
        """Release camera resources."""

        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info("Camera closed")
