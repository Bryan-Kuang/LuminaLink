"""
Video Processing Module

Handles video stream reading, frame extraction, keyframe detection and scene change detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Generator, Optional, Tuple, List
from dataclasses import dataclass
import logging
from concurrent.futures import ThreadPoolExecutor

from .config import get_config
from .luminalink.types import VideoFrame

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    """Video information"""
    path: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration: float  # seconds


class VideoProcessor:
    """Video Processor"""
    
    def __init__(self, video_path: str):
        """
        Initialize video processor
        
        Args:
            video_path: Video file path or stream URL
        """
        self.video_path = video_path
        self.config = get_config().video
        self.cap: Optional[cv2.VideoCapture] = None
        self.video_info: Optional[VideoInfo] = None
        self._previous_frame: Optional[np.ndarray] = None
        
    def open(self) -> bool:
        """Open video"""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            logger.error(f"Failed to open video: {self.video_path}")
            return False
        
        # Get video information
        self.video_info = VideoInfo(
            path=self.video_path,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=self.cap.get(cv2.CAP_PROP_FPS),
            total_frames=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            duration=0.0
        )
        
        if self.video_info.fps > 0:
            self.video_info.duration = self.video_info.total_frames / self.video_info.fps
        
        logger.info(f"Video opened: {self.video_info}")
        return True
    
    def close(self):
        """Close video"""
        if self.cap:
            self.cap.release()
            self.cap = None
        self._previous_frame = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def get_info(self) -> Optional[VideoInfo]:
        """Get video information"""
        return self.video_info
    
    def seek(self, timestamp: float) -> bool:
        """
        Seek to specified time
        
        Args:
            timestamp: Timestamp (seconds)
        """
        if not self.cap:
            return False
        
        # Convert to milliseconds
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        return True
    
    def read_frame(self) -> Optional[VideoFrame]:
        """Read next frame"""
        if not self.cap:
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        
        # Detect scene change
        is_scene_change = self._detect_scene_change(frame)
        
        # Save current frame for next comparison
        self._previous_frame = frame.copy()

        return VideoFrame.from_video_processor(
            frame=frame,
            timestamp=timestamp,
            frame_number=frame_number,
            is_scene_change=is_scene_change
        )
    
    def _detect_scene_change(self, current_frame: np.ndarray) -> bool:
        """
        Detect scene change
        
        Uses histogram comparison to detect scene changes
        """
        if self._previous_frame is None:
            return True  # First frame is considered scene change
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(self._previous_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate histograms
        prev_hist = cv2.calcHist([prev_gray], [0], None, [256], [0, 256])
        curr_hist = cv2.calcHist([curr_gray], [0], None, [256], [0, 256])
        
        # Normalize
        cv2.normalize(prev_hist, prev_hist)
        cv2.normalize(curr_hist, curr_hist)
        
        # Compare histograms
        diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        
        # If correlation is below threshold, consider it a scene change
        return diff < (1 - self.config.scene_change_threshold)
    
    def extract_keyframes(
        self, 
        interval: Optional[float] = None,
        max_frames: Optional[int] = None
    ) -> Generator[VideoFrame, None, None]:
        """
        Extract keyframes
        
        Args:
            interval: Keyframe interval (seconds), uses config value if not specified
            max_frames: Maximum number of frames
        
        Yields:
            VideoFrame objects
        """
        if not self.cap:
            return
        
        interval = interval or self.config.keyframe_interval
        fps = self.video_info.fps if self.video_info else 30.0
        frame_interval = int(fps * interval)
        
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Extract at interval or scene change
            is_interval_frame = frame_count % frame_interval == 0
            
            # Detect scene change
            is_scene_change = self._detect_scene_change(frame)
            self._previous_frame = frame.copy()
            
            if is_interval_frame or is_scene_change:
                if max_frames and extracted_count >= max_frames:
                    break
                
                timestamp = frame_count / fps

                yield VideoFrame.from_video_processor(
                    frame=frame,
                    timestamp=timestamp,
                    frame_number=frame_count,
                    is_keyframe=is_interval_frame,
                    is_scene_change=is_scene_change
                )
                
                extracted_count += 1
            
            frame_count += 1
    
    def get_frame_at(self, timestamp: float) -> Optional[VideoFrame]:
        """
        Get frame at specified timestamp
        
        Args:
            timestamp: Timestamp (seconds)
        
        Returns:
            VideoFrame object
        """
        if not self.cap:
            return None
        
        # Save current position
        current_pos = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Seek to specified position
        self.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        
        ret, frame = self.cap.read()
        
        # Restore position
        self.cap.set(cv2.CAP_PROP_POS_MSEC, current_pos)
        
        if not ret:
            return None

        frame_number = int(timestamp * (self.video_info.fps if self.video_info else 30.0))

        return VideoFrame.from_video_processor(
            frame=frame,
            timestamp=timestamp,
            frame_number=frame_number
        )
    
    def get_frames_in_range(
        self, 
        start_time: float, 
        end_time: float,
        sample_count: int = 3
    ) -> List[VideoFrame]:
        """
        Get frames from time range
        
        Args:
            start_time: Start time (seconds)
            end_time: End time (seconds)
            sample_count: Number of frames to sample
        
        Returns:
            List of VideoFrame objects
        """
        frames = []
        duration = end_time - start_time
        
        if duration <= 0 or sample_count <= 0:
            return frames
        
        step = duration / sample_count
        
        for i in range(sample_count):
            timestamp = start_time + step * i + step / 2
            frame = self.get_frame_at(timestamp)
            if frame:
                frames.append(frame)
        
        return frames
    
    def resize_frame(
        self, 
        frame: np.ndarray, 
        max_width: Optional[int] = None,
        max_height: Optional[int] = None
    ) -> np.ndarray:
        """
        Resize frame
        
        Args:
            frame: Original frame
            max_width: Maximum width
            max_height: Maximum height
        
        Returns:
            Resized frame
        """
        height, width = frame.shape[:2]
        
        max_width = max_width or self.config.preview_width
        max_height = max_height or self.config.preview_height
        
        # Calculate scale ratio
        scale = min(max_width / width, max_height / height)
        
        if scale >= 1:
            return frame
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)


class FrameBuffer:
    """Frame buffer for caching recent frames"""
    
    def __init__(self, max_size: int = 30):
        self.max_size = max_size
        self._frames: List[VideoFrame] = []
    
    def add(self, frame: VideoFrame):
        """Add frame to buffer"""
        self._frames.append(frame)
        if len(self._frames) > self.max_size:
            self._frames.pop(0)
    
    def get_recent(self, count: int = 5) -> List[VideoFrame]:
        """Get recent frames"""
        return self._frames[-count:]
    
    def get_by_timestamp(self, timestamp: float) -> Optional[VideoFrame]:
        """Get frame by timestamp"""
        for frame in reversed(self._frames):
            if abs(frame.timestamp - timestamp) < 0.1:
                return frame
        return None
    
    def clear(self):
        """Clear buffer"""
        self._frames.clear()
