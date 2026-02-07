"""
Realtime Playback Module

Plays video in realtime with synchronized narration
"""

import asyncio
import cv2
import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass
import logging
import time
from threading import Thread, Event
from queue import Queue, Empty

from .config import get_config
from .video_processor import VideoProcessor, VideoFrame
from .audio_detector import RealtimeAudioDetector
from .character_recognizer import CharacterRecognizer, CharacterTracker
from .scene_analyzer import SceneAnalyzer
from .narrator import Narrator, Narration
from .tts_engine import TTSManager, AudioPlayer

logger = logging.getLogger(__name__)


@dataclass
class PlaybackState:
    """Playback state"""
    is_playing: bool = False
    is_paused: bool = False
    current_time: float = 0.0
    duration: float = 0.0
    current_narration: Optional[str] = None


class RealtimePlayer:
    """Realtime Player"""
    
    def __init__(
        self,
        video_path: str,
        character_recognizer: Optional[CharacterRecognizer] = None
    ):
        """
        Initialize realtime player
        
        Args:
            video_path: Video path
            character_recognizer: Character recognizer
        """
        self.video_path = video_path
        self.config = get_config()
        
        # Components
        self.video_processor = VideoProcessor(video_path)
        self.character_recognizer = character_recognizer or CharacterRecognizer()
        self.character_tracker = CharacterTracker(self.character_recognizer)
        self.scene_analyzer = SceneAnalyzer()
        self.narrator = Narrator()
        self.tts_manager = TTSManager()
        self.audio_player = AudioPlayer()
        self.audio_detector = RealtimeAudioDetector()
        
        # State
        self.state = PlaybackState()
        
        # Thread control
        self._stop_event = Event()
        self._pause_event = Event()
        self._pause_event.set()  # Not paused by default
        
        # Task queues
        self._analysis_queue: Queue = Queue(maxsize=10)
        self._narration_queue: Queue = Queue(maxsize=10)
        
        # Callbacks
        self._on_frame_callback: Optional[Callable[[np.ndarray], None]] = None
        self._on_narration_callback: Optional[Callable[[str], None]] = None
    
    def set_on_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """Set frame callback"""
        self._on_frame_callback = callback
    
    def set_on_narration_callback(self, callback: Callable[[str], None]):
        """Set narration callback"""
        self._on_narration_callback = callback
    
    def start(self):
        """Start playback"""
        if not self.video_processor.open():
            raise RuntimeError("Failed to open video")
        
        video_info = self.video_processor.get_info()
        if video_info:
            self.state.duration = video_info.duration
        
        self.state.is_playing = True
        self._stop_event.clear()
        
        # Start worker threads
        Thread(target=self._analysis_worker, daemon=True).start()
        Thread(target=self._narration_worker, daemon=True).start()
        
        # Main playback loop
        self._playback_loop()
    
    def stop(self):
        """Stop playback"""
        self._stop_event.set()
        self.state.is_playing = False
        self.video_processor.close()
        self.audio_player.stop()
    
    def pause(self):
        """Pause"""
        self._pause_event.clear()
        self.state.is_paused = True
    
    def resume(self):
        """Resume"""
        self._pause_event.set()
        self.state.is_paused = False
    
    def seek(self, timestamp: float):
        """Seek to timestamp"""
        self.video_processor.seek(timestamp)
        self.state.current_time = timestamp
        self.scene_analyzer.clear_context()
    
    def _playback_loop(self):
        """Main playback loop"""
        video_info = self.video_processor.get_info()
        fps = video_info.fps if video_info else 30.0
        frame_duration = 1.0 / fps
        
        last_analysis_time = 0.0
        analysis_interval = self.config.video.keyframe_interval
        
        while not self._stop_event.is_set():
            # Check pause
            self._pause_event.wait()
            
            # Read frame
            frame = self.video_processor.read_frame()
            if frame is None:
                logger.info("Video playback complete")
                break
            
            self.state.current_time = frame.timestamp
            
            # Display frame
            if self._on_frame_callback:
                self._on_frame_callback(frame.frame)
            
            # Periodic scene analysis
            if frame.timestamp - last_analysis_time >= analysis_interval:
                try:
                    self._analysis_queue.put_nowait((frame, frame.timestamp))
                    last_analysis_time = frame.timestamp
                except:
                    pass  # Queue full, skip
            
            # Control frame rate
            time.sleep(frame_duration)
        
        self.state.is_playing = False
    
    def _analysis_worker(self):
        """Analysis worker thread"""
        while not self._stop_event.is_set():
            try:
                frame, timestamp = self._analysis_queue.get(timeout=1.0)
            except Empty:
                continue
            
            try:
                # Recognize characters
                characters = self.character_recognizer.get_characters_in_frame(
                    frame.frame,
                    timestamp=timestamp
                )
                
                # Check if should narrate
                if not self.narrator.should_narrate(timestamp):
                    continue
                
                # Check if silence
                if not self.audio_detector.is_current_silence():
                    continue
                
                # Analyze scene
                analysis = asyncio.run(
                    self.scene_analyzer.analyze_frame_async(
                        frame.frame,
                        characters_in_frame=characters,
                        timestamp=timestamp
                    )
                )
                
                # Generate narration
                estimated_duration = 5.0  # Estimated narration duration
                narration = self.narrator.generate_narration(
                    analysis,
                    slot=(timestamp, timestamp + estimated_duration),
                    characters_in_frame=characters
                )
                
                if narration:
                    try:
                        self._narration_queue.put_nowait(narration)
                    except:
                        pass
                        
            except Exception as e:
                logger.error(f"Analysis failed: {e}")
    
    def _narration_worker(self):
        """Narration worker thread"""
        while not self._stop_event.is_set():
            try:
                narration = self._narration_queue.get(timeout=1.0)
            except Empty:
                continue
            
            try:
                self.state.current_narration = narration.text
                
                # Callback
                if self._on_narration_callback:
                    self._on_narration_callback(narration.text)
                
                # Synthesize and play speech
                result = asyncio.run(self.tts_manager.synthesize(narration.text))
                
                if result.success:
                    asyncio.run(self.audio_player.play(result.audio_path))
                
                self.state.current_narration = None
                
            except Exception as e:
                logger.error(f"Narration playback failed: {e}")


class PreviewWindow:
    """Preview window"""
    
    def __init__(self, title: str = "LuminaLink Preview"):
        self.title = title
        self.window_created = False
        self._current_narration: Optional[str] = None
    
    def show_frame(self, frame: np.ndarray):
        """Display frame"""
        if not self.window_created:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            self.window_created = True
        
        display_frame = frame.copy()
        
        # Add narration subtitle
        if self._current_narration:
            self._add_subtitle(display_frame, self._current_narration)
        
        cv2.imshow(self.title, display_frame)
        cv2.waitKey(1)
    
    def set_narration(self, text: str):
        """Set current narration"""
        self._current_narration = text
    
    def clear_narration(self):
        """Clear narration"""
        self._current_narration = None
    
    def _add_subtitle(self, frame: np.ndarray, text: str):
        """Add subtitle"""
        height, width = frame.shape[:2]
        
        # Subtitle background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # Calculate text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, height - 60),
            (width, height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        x = (width - text_width) // 2
        y = height - 20
        
        cv2.putText(
            frame,
            text,
            (x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )
    
    def close(self):
        """Close window"""
        if self.window_created:
            cv2.destroyWindow(self.title)
            self.window_created = False


def run_realtime_player(
    video_path: str,
    characters_config: Optional[str] = None,
    show_preview: bool = True
):
    """
    Run realtime player
    
    Args:
        video_path: Video path
        characters_config: Character config
        show_preview: Show preview
    """
    from rich.console import Console
    console = Console()
    
    # Initialize character recognition
    recognizer = CharacterRecognizer()
    if characters_config:
        import json
        with open(characters_config) as f:
            data = json.load(f)
        for char in data.get("characters", []):
            recognizer.add_character(
                name=char["name"],
                aliases=char.get("aliases", [])
            )
    
    # Create player
    player = RealtimePlayer(video_path, recognizer)
    
    # Setup preview
    preview = PreviewWindow() if show_preview else None
    
    if preview:
        player.set_on_frame_callback(preview.show_frame)
        player.set_on_narration_callback(lambda t: (
            preview.set_narration(t),
            console.print(f"[green]Narration:[/green] {t}")
        ))
    
    console.print("[bold blue]Starting realtime playback...[/bold blue]")
    console.print("Press Q to exit\n")
    
    try:
        player.start()
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()
        if preview:
            preview.close()
        console.print("\n[yellow]Playback ended[/yellow]")
