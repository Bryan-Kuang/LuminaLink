"""
Camera Realtime Controller Module

Orchestrates real-time camera narration pipeline:
Camera Input → Scene Analysis → Narration → TTS Playback
"""

import asyncio
import logging
import time
from threading import Thread, Event
from queue import Queue, Empty
from typing import Optional, Callable
from dataclasses import dataclass
import numpy as np

from .config import get_config
from .luminalink.input import CameraInput
from .audio_detector import RealtimeAudioDetector
from .audio_input import AudioInputStream
from .character_recognizer import CharacterRecognizer
from .scene_analyzer import SceneAnalyzer
from .narrator import Narrator
from .tts_engine import TTSManager, AudioPlayer

logger = logging.getLogger(__name__)


@dataclass
class PlaybackState:
    """Playback state for camera mode"""
    is_playing: bool = False
    is_paused: bool = False
    current_narration: Optional[str] = None


class CameraRealtimeController:
    """
    Orchestrates real-time camera narration pipeline.

    Thread Model:
    - Main thread: GUI event loop
    - Camera thread: Frame capture
    - Audio thread: Microphone capture (handled by AudioInputStream)
    - Analysis thread: Scene analysis
    - Narration thread: TTS generation and playback
    """

    def __init__(
        self,
        camera_index: int = 0,
        characters_config: Optional[str] = None,
        camera_width: int = 1280,
        camera_height: int = 720,
        camera_fps: int = 30,
        mic_device_index: Optional[int] = None,
        cooldown: float = 5.0,
    ):
        """
        Initialize camera realtime controller.

        Args:
            camera_index: Camera device index (default: 0)
            characters_config: Path to character configuration JSON (optional)
            camera_width: Camera resolution width (default: 1280)
            camera_height: Camera resolution height (default: 720)
            camera_fps: Camera frame rate (default: 30)
        """
        self.camera_index = camera_index
        self.config = get_config()

        # Components
        self.camera_input = CameraInput(
            camera_index=camera_index,
            width=camera_width,
            height=camera_height,
            fps=camera_fps
        )
        self.audio_detector = RealtimeAudioDetector(
            silence_threshold_db=self.config.narration.silence_threshold
        )
        try:
            self.audio_stream = AudioInputStream(
                detector=self.audio_detector,
                sample_rate=22050,
                device_id=mic_device_index
            )
        except (ImportError, Exception) as e:
            logger.warning(f"Audio input unavailable: {e}")
            logger.warning("Continuing without dialogue detection")
            self.audio_stream = None
        self.scene_analyzer = SceneAnalyzer()
        self.narrator = Narrator(cooldown=cooldown)
        self.tts_manager = TTSManager()
        self.audio_player = AudioPlayer()

        # Character recognition (optional)
        self.character_recognizer = CharacterRecognizer()
        if characters_config:
            try:
                self.character_recognizer.load_config(characters_config)
                logger.info(f"Loaded character config: {characters_config}")
            except Exception as e:
                logger.error(f"Failed to load character config: {e}")

        # State
        self.state = PlaybackState()

        # Thread control
        self._stop_event = Event()
        self._pause_event = Event()
        self._pause_event.set()  # Not paused by default

        # Queues
        self._analysis_queue: Queue = Queue(maxsize=10)
        self._narration_queue: Queue = Queue(maxsize=10)

        # Callbacks for GUI
        self._on_frame_callback: Optional[Callable[[np.ndarray], None]] = None
        self._on_narration_callback: Optional[Callable[[str], None]] = None
        self._on_status_callback: Optional[Callable[[str], None]] = None

        # Threads
        self._threads = []

        # Statistics
        self._frame_count = 0
        self._narration_count = 0

    def set_on_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Set frame callback (called when new frame is captured)"""
        self._on_frame_callback = callback

    def set_on_narration_callback(self, callback: Callable[[str], None]):
        """Set narration callback (called when narration is generated)"""
        self._on_narration_callback = callback

    def set_on_status_callback(self, callback: Callable[[str], None]):
        """Set status callback (called when status changes)"""
        self._on_status_callback = callback

    def start(self):
        """Start all worker threads"""
        try:
            # Open camera
            self.camera_input.open()
            logger.info(f"Camera opened: index={self.camera_index}")

            # Start audio stream
            if self.audio_stream:
                try:
                    self.audio_stream.start()
                    logger.info("Audio stream started")
                except Exception as e:
                    logger.warning(f"Failed to start audio stream: {e}")
                    logger.warning("Continuing without audio detection")

            # Start worker threads
            threads = [
                Thread(target=self._camera_worker, daemon=True, name="CameraThread"),
                Thread(target=self._analysis_worker, daemon=True, name="AnalysisThread"),
                Thread(target=self._narration_worker, daemon=True, name="NarrationThread")
            ]

            for t in threads:
                t.start()
                self._threads.append(t)

            self.state.is_playing = True
            self._notify_status("Running")
            logger.info("Camera realtime controller started")

        except Exception as e:
            logger.error(f"Failed to start controller: {e}")
            self.stop()
            raise

    def pause(self):
        """Pause narration"""
        self._pause_event.clear()
        self.state.is_paused = True
        self._notify_status("Paused")
        logger.info("Controller paused")

    def resume(self):
        """Resume narration"""
        self._pause_event.set()
        self.state.is_paused = False
        self._notify_status("Running")
        logger.info("Controller resumed")

    def stop(self):
        """Stop all threads and release resources"""
        logger.info("Stopping controller...")
        self._stop_event.set()

        # Wait for threads to finish
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=2.0)

        # Cleanup resources
        try:
            self.camera_input.close()
        except Exception as e:
            logger.error(f"Error closing camera: {e}")

        if self.audio_stream:
            try:
                self.audio_stream.stop()
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")

        try:
            self.audio_player.stop()
        except Exception as e:
            logger.error(f"Error stopping audio player: {e}")

        self.state.is_playing = False
        self._notify_status("Stopped")
        logger.info("Controller stopped")

    def _camera_worker(self):
        """Camera capture thread with auto-reconnect on disconnection."""
        last_analysis_time = 0.0
        analysis_interval = self.config.video.keyframe_interval  # Default 1.0 second
        max_reconnect_attempts = 10
        base_reconnect_delay = 1.0  # seconds

        while not self._stop_event.is_set():
            try:
                for video_frame in self.camera_input.frames():
                    if self._stop_event.is_set():
                        return

                    # Wait if paused
                    self._pause_event.wait()

                    self._frame_count += 1

                    # Send to GUI for display (non-blocking)
                    if self._on_frame_callback:
                        try:
                            self._on_frame_callback(video_frame.image_bgr)
                        except Exception as e:
                            logger.error(f"Frame callback error: {e}")

                    # Send to analysis queue at intervals
                    current_time = video_frame.timestamp
                    if current_time - last_analysis_time >= analysis_interval:
                        try:
                            self._analysis_queue.put_nowait(video_frame)
                            last_analysis_time = current_time
                        except:
                            logger.debug("Analysis queue full, skipping frame")

            except Exception as e:
                logger.error(f"Camera worker error: {e}")

            # If we get here, the frame iterator exited (camera disconnected)
            if self._stop_event.is_set():
                break

            # Attempt reconnection with exponential backoff
            logger.warning("Camera disconnected, attempting to reconnect...")
            self._notify_status("Reconnecting")

            for attempt in range(1, max_reconnect_attempts + 1):
                if self._stop_event.is_set():
                    break

                delay = min(base_reconnect_delay * attempt, 10.0)
                logger.info(f"Reconnect attempt {attempt}/{max_reconnect_attempts} "
                            f"in {delay:.0f}s...")
                self._stop_event.wait(delay)

                if self._stop_event.is_set():
                    break

                if self.camera_input.reconnect():
                    logger.info("Camera reconnected successfully")
                    self._notify_status("Running")
                    break
            else:
                # All attempts failed
                logger.error(f"Failed to reconnect after {max_reconnect_attempts} attempts")
                self._notify_status("Camera Lost")
                break

        logger.info("Camera worker stopped")

    def _analysis_worker(self):
        """Scene analysis worker thread"""
        # Create a persistent event loop for this thread to avoid
        # "Event loop is closed" errors from repeated asyncio.run() calls
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        consecutive_failures = 0

        try:
            while not self._stop_event.is_set():
                try:
                    video_frame = self._analysis_queue.get(timeout=1.0)
                except Empty:
                    continue

                # Wait if paused
                self._pause_event.wait()

                try:
                    timestamp = video_frame.timestamp

                    # Check if should narrate (minimum interval)
                    if not self.narrator.should_narrate(timestamp):
                        logger.debug(f"Skipping narration: too soon (interval)")
                        continue

                    # Check if silence (no dialogue)
                    if not self.audio_detector.is_silence_long_enough(min_duration=1.5):
                        logger.debug("Skipping narration: insufficient silence (need 1.5s)")
                        continue

                    # Back off if API keeps failing
                    if consecutive_failures >= 3:
                        backoff = min(30, 5 * consecutive_failures)
                        logger.warning(
                            f"API failed {consecutive_failures} times, "
                            f"backing off {backoff}s"
                        )
                        self._stop_event.wait(backoff)
                        if self._stop_event.is_set():
                            break

                    # Recognize characters (optional)
                    characters = []
                    try:
                        if hasattr(self.character_recognizer, 'is_enabled') and \
                           self.character_recognizer.is_enabled():
                            characters = self.character_recognizer.get_characters_in_frame(
                                video_frame.image_bgr,
                                timestamp=timestamp
                            )
                    except Exception as e:
                        logger.warning(f"Character recognition error: {e}")

                    # Analyze scene (async AI call on persistent loop)
                    try:
                        analysis = loop.run_until_complete(
                            self.scene_analyzer.analyze_frame_async(
                                video_frame.image_bgr,
                                characters_in_frame=characters,
                                timestamp=timestamp
                            )
                        )
                    except Exception as e:
                        logger.error(f"Scene analysis error: {e}")
                        consecutive_failures += 1
                        continue

                    # Check if analysis was a fallback (API failure)
                    if analysis.confidence == 0.0:
                        consecutive_failures += 1
                        continue

                    consecutive_failures = 0

                    # Generate narration
                    try:
                        narration = self.narrator.generate_narration(
                            analysis,
                            slot=(timestamp, timestamp + 5.0),
                            characters_in_frame=characters
                        )

                        if narration:
                            self._narration_queue.put(narration)
                            logger.info(f"Narration generated: {narration.text[:50]}...")
                    except Exception as e:
                        logger.error(f"Narration generation error: {e}")

                except Exception as e:
                    logger.error(f"Analysis worker error: {e}")
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            logger.info("Analysis worker stopped")

    def _narration_worker(self):
        """TTS generation and playback worker thread"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            while not self._stop_event.is_set():
                try:
                    narration = self._narration_queue.get(timeout=1.0)
                except Empty:
                    continue

                # Wait if paused
                self._pause_event.wait()

                try:
                    # Notify GUI
                    if self._on_narration_callback:
                        try:
                            self._on_narration_callback(narration.text)
                        except Exception as e:
                            logger.error(f"Narration callback error: {e}")

                    self.state.current_narration = narration.text

                    # Synthesize TTS audio
                    try:
                        result = loop.run_until_complete(
                            self.tts_manager.synthesize(narration.text)
                        )
                    except Exception as e:
                        logger.error(f"TTS synthesis error: {e}")
                        continue

                    if not result.success:
                        logger.error(f"TTS synthesis failed: {result.error}")
                        continue

                    # Play audio — blocks until playback completes
                    try:
                        loop.run_until_complete(
                            self.audio_player.play(result.audio_path, block=True)
                        )
                        self._narration_count += 1
                    except Exception as e:
                        logger.error(f"Audio playback error: {e}")

                    # Clean up temp file
                    try:
                        import os as _os
                        if _os.path.exists(result.audio_path):
                            _os.remove(result.audio_path)
                    except Exception:
                        pass

                    self.state.current_narration = None

                except Exception as e:
                    logger.error(f"Narration worker error: {e}")
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            logger.info("Narration worker stopped")

    def _notify_status(self, status: str):
        """Notify status change via callback"""
        if self._on_status_callback:
            try:
                self._on_status_callback(status)
            except Exception as e:
                logger.error(f"Status callback error: {e}")

    def get_statistics(self) -> dict:
        """Get runtime statistics"""
        return {
            "frame_count": self._frame_count,
            "narration_count": self._narration_count,
            "is_playing": self.state.is_playing,
            "is_paused": self.state.is_paused,
        }
