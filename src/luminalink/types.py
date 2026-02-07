"""
LuminaLink Core Types

Unified data structures for video processing and camera input.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class VideoFrame:
    """
    Unified video frame data structure.

    This class supports both file-based video processing and real-time camera input.
    It provides multiple naming conventions for compatibility:
    - image_bgr / frame: BGR image data (NumPy array)
    - timestamp (seconds) / pts_ms (milliseconds): Time information
    - frame_index / frame_number: Frame sequence number
    """

    # Primary fields (camera input convention)
    pts_ms: int                     # Presentation timestamp in milliseconds
    frame_index: int                # Sequential frame number (0-indexed)
    image_bgr: np.ndarray          # BGR image data (OpenCV format)

    # Optional fields (video processor convention)
    is_keyframe: bool = False       # Whether this is a keyframe
    is_scene_change: bool = False   # Whether this is a scene change point

    @property
    def timestamp(self) -> float:
        """Timestamp in seconds (for video processor compatibility)"""
        return self.pts_ms / 1000.0

    @property
    def frame(self) -> np.ndarray:
        """Alias for image_bgr (for video processor compatibility)"""
        return self.image_bgr

    @property
    def frame_number(self) -> int:
        """Alias for frame_index (for video processor compatibility)"""
        return self.frame_index

    @classmethod
    def from_video_processor(cls, frame: np.ndarray, timestamp: float,
                            frame_number: int, is_keyframe: bool = False,
                            is_scene_change: bool = False) -> VideoFrame:
        """
        Create VideoFrame from video processor convention.

        Args:
            frame: BGR image data
            timestamp: Timestamp in seconds
            frame_number: Frame number
            is_keyframe: Whether this is a keyframe
            is_scene_change: Whether this is a scene change point

        Returns:
            VideoFrame instance
        """
        return cls(
            pts_ms=int(timestamp * 1000),
            frame_index=frame_number,
            image_bgr=frame,
            is_keyframe=is_keyframe,
            is_scene_change=is_scene_change
        )

    @classmethod
    def from_camera_input(cls, pts_ms: int, frame_index: int,
                         image_bgr: np.ndarray) -> VideoFrame:
        """
        Create VideoFrame from camera input convention.

        Args:
            pts_ms: Presentation timestamp in milliseconds
            frame_index: Sequential frame number
            image_bgr: BGR image data

        Returns:
            VideoFrame instance
        """
        return cls(
            pts_ms=pts_ms,
            frame_index=frame_index,
            image_bgr=image_bgr
        )
