from __future__ import annotations

from collections.abc import Iterator

import cv2

from luminalink.types import VideoFrame


class VideoFileInput:
    """Decode video frames from a local file using OpenCV."""

    def __init__(self, video_path: str):
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        self._frame_index = 0

    def frames(self) -> Iterator[VideoFrame]:
        """Yield decoded frames with OpenCV-provided timestamps."""

        while True:
            ok, frame = self._cap.read()
            if not ok:
                break
            pts_ms = int(self._cap.get(cv2.CAP_PROP_POS_MSEC) or 0)
            yield VideoFrame(pts_ms=pts_ms, frame_index=self._frame_index, image_bgr=frame)
            self._frame_index += 1

    def close(self) -> None:
        """Release underlying decoder resources."""

        self._cap.release()

