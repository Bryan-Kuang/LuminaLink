"""
Audio Detection Module

Detects dialogue and silence segments in video to determine when to insert narration
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging
import tempfile
import os

logger = logging.getLogger(__name__)


@dataclass
class AudioSegment:
    """Audio segment"""
    start_time: float    # Start time (seconds)
    end_time: float      # End time (seconds)
    has_speech: bool     # Contains speech
    volume_db: float     # Average volume (dB)
    
    @property
    def duration(self) -> float:
        """Segment duration"""
        return self.end_time - self.start_time


@dataclass
class SilenceWindow:
    """Silence window - suitable time period for inserting narration"""
    start_time: float
    end_time: float
    confidence: float  # Confidence (0-1)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class AudioDetector:
    """Audio Detector"""
    
    def __init__(
        self,
        silence_threshold_db: float = -40.0,
        min_silence_duration: float = 1.0,
        min_narration_gap: float = 3.0
    ):
        """
        Initialize audio detector
        
        Args:
            silence_threshold_db: Silence threshold (dB)
            min_silence_duration: Minimum silence duration (seconds)
            min_narration_gap: Minimum gap between narrations (seconds)
        """
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration = min_silence_duration
        self.min_narration_gap = min_narration_gap
        
        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: int = 22050
    
    def load_audio_from_video(self, video_path: str) -> bool:
        """
        Extract audio from video file
        
        Args:
            video_path: Video file path
        
        Returns:
            Whether loading was successful
        """
        try:
            from moviepy import VideoFileClip
            import librosa
            
            logger.info(f"Extracting audio from video: {video_path}")
            
            # Create temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # Extract audio
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(
                    tmp_path, 
                    fps=self._sample_rate,
                    logger=None
                )
                video.close()
                
                # Load audio data
                self._audio_data, self._sample_rate = librosa.load(
                    tmp_path, 
                    sr=self._sample_rate,
                    mono=True
                )
                
                logger.info(f"Audio loaded successfully, duration: {len(self._audio_data) / self._sample_rate:.2f}s")
                return True
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            return False
    
    def load_audio_file(self, audio_path: str) -> bool:
        """
        Load audio file directly
        
        Args:
            audio_path: Audio file path
        """
        try:
            import librosa
            
            self._audio_data, self._sample_rate = librosa.load(
                audio_path,
                sr=self._sample_rate,
                mono=True
            )
            return True
            
        except Exception as e:
            logger.error(f"Audio loading failed: {e}")
            return False
    
    def analyze_audio(
        self,
        window_size: float = 0.5
    ) -> List[AudioSegment]:
        """
        Analyze audio to identify speech and silence segments
        
        Args:
            window_size: Analysis window size (seconds)
        
        Returns:
            List of AudioSegment
        """
        if self._audio_data is None:
            logger.error("Audio data not loaded")
            return []
        
        segments = []
        samples_per_window = int(self._sample_rate * window_size)
        total_samples = len(self._audio_data)
        
        for i in range(0, total_samples, samples_per_window):
            window = self._audio_data[i:i + samples_per_window]
            
            if len(window) == 0:
                continue
            
            # Calculate RMS energy
            rms = np.sqrt(np.mean(window ** 2))
            
            # Convert to dB
            if rms > 0:
                volume_db = 20 * np.log10(rms)
            else:
                volume_db = -100
            
            start_time = i / self._sample_rate
            end_time = min((i + samples_per_window) / self._sample_rate, 
                          total_samples / self._sample_rate)
            
            # Determine if there is speech
            has_speech = volume_db > self.silence_threshold_db
            
            segments.append(AudioSegment(
                start_time=start_time,
                end_time=end_time,
                has_speech=has_speech,
                volume_db=volume_db
            ))
        
        logger.info(f"Audio analysis complete, {len(segments)} segments")
        return segments
    
    def find_silence_windows(
        self,
        segments: Optional[List[AudioSegment]] = None
    ) -> List[SilenceWindow]:
        """
        Find silence windows suitable for inserting narration
        
        Args:
            segments: Audio segment list, analyzes audio if None
        
        Returns:
            List of SilenceWindow
        """
        if segments is None:
            segments = self.analyze_audio()
        
        if not segments:
            return []
        
        windows = []
        silence_start: Optional[float] = None
        
        for segment in segments:
            if not segment.has_speech:
                # Start silence
                if silence_start is None:
                    silence_start = segment.start_time
            else:
                # Speech starts, end silence
                if silence_start is not None:
                    silence_end = segment.start_time
                    duration = silence_end - silence_start
                    
                    if duration >= self.min_silence_duration:
                        # Calculate confidence (based on silence duration)
                        confidence = min(duration / 5.0, 1.0)
                        
                        windows.append(SilenceWindow(
                            start_time=silence_start,
                            end_time=silence_end,
                            confidence=confidence
                        ))
                    
                    silence_start = None
        
        # Handle trailing silence
        if silence_start is not None:
            silence_end = segments[-1].end_time
            duration = silence_end - silence_start
            
            if duration >= self.min_silence_duration:
                windows.append(SilenceWindow(
                    start_time=silence_start,
                    end_time=silence_end,
                    confidence=min(duration / 5.0, 1.0)
                ))
        
        logger.info(f"Found {len(windows)} silence windows")
        return windows
    
    def get_narration_slots(
        self,
        silence_windows: Optional[List[SilenceWindow]] = None
    ) -> List[Tuple[float, float]]:
        """
        Get narration time slots
        
        Args:
            silence_windows: Silence window list
        
        Returns:
            List of (start_time, end_time) tuples
        """
        if silence_windows is None:
            silence_windows = self.find_silence_windows()
        
        slots = []
        last_end = 0.0
        
        for window in silence_windows:
            # Ensure minimum gap from last narration
            if window.start_time - last_end >= self.min_narration_gap:
                # Leave margin at start and end of silence window
                margin = 0.3
                start = window.start_time + margin
                end = window.end_time - margin
                
                if end - start >= 1.0:  # At least 1 second
                    slots.append((start, end))
                    last_end = end
        
        logger.info(f"Generated {len(slots)} narration slots")
        return slots
    
    def is_silence_at(self, timestamp: float) -> bool:
        """
        Check if specified timestamp is silence
        
        Args:
            timestamp: Timestamp (seconds)
        """
        if self._audio_data is None:
            return False
        
        # Get audio samples around timestamp
        sample_idx = int(timestamp * self._sample_rate)
        window_size = int(0.5 * self._sample_rate)  # 0.5 second window
        
        start_idx = max(0, sample_idx - window_size // 2)
        end_idx = min(len(self._audio_data), sample_idx + window_size // 2)
        
        window = self._audio_data[start_idx:end_idx]
        
        if len(window) == 0:
            return True
        
        # Calculate volume
        rms = np.sqrt(np.mean(window ** 2))
        volume_db = 20 * np.log10(rms) if rms > 0 else -100
        
        return volume_db < self.silence_threshold_db


class RealtimeAudioDetector:
    """Realtime audio detector for live audio streams"""
    
    def __init__(
        self,
        silence_threshold_db: float = -40.0,
        buffer_duration: float = 2.0,
        sample_rate: int = 22050
    ):
        """
        Initialize realtime audio detector
        
        Args:
            silence_threshold_db: Silence threshold (dB)
            buffer_duration: Audio buffer duration (seconds)
            sample_rate: Sample rate
        """
        self.silence_threshold_db = silence_threshold_db
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate
        
        buffer_size = int(sample_rate * buffer_duration)
        self._buffer = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_pos = 0
        self._is_silence = True
    
    def feed_audio(self, audio_chunk: np.ndarray):
        """
        Feed audio data to buffer
        
        Args:
            audio_chunk: Audio data chunk
        """
        chunk_size = len(audio_chunk)
        buffer_size = len(self._buffer)
        
        if chunk_size >= buffer_size:
            # If chunk is larger than buffer, take last part
            self._buffer = audio_chunk[-buffer_size:].copy()
            self._buffer_pos = 0
        else:
            # Add to buffer, overwriting old data cyclically
            end_pos = self._buffer_pos + chunk_size
            
            if end_pos <= buffer_size:
                self._buffer[self._buffer_pos:end_pos] = audio_chunk
            else:
                first_part = buffer_size - self._buffer_pos
                self._buffer[self._buffer_pos:] = audio_chunk[:first_part]
                self._buffer[:end_pos - buffer_size] = audio_chunk[first_part:]
            
            self._buffer_pos = end_pos % buffer_size
        
        # Update silence state
        self._update_silence_state()
    
    def _update_silence_state(self):
        """Update silence state"""
        # Calculate current volume
        rms = np.sqrt(np.mean(self._buffer ** 2))
        volume_db = 20 * np.log10(rms) if rms > 0 else -100
        
        self._is_silence = volume_db < self.silence_threshold_db
    
    def is_current_silence(self) -> bool:
        """Check if current audio is silence"""
        return self._is_silence
    
    def get_current_volume_db(self) -> float:
        """Get current volume (dB)"""
        rms = np.sqrt(np.mean(self._buffer ** 2))
        return 20 * np.log10(rms) if rms > 0 else -100
    
    def reset(self):
        """Reset detector"""
        self._buffer.fill(0)
        self._buffer_pos = 0
        self._is_silence = True
