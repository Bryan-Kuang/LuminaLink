"""
Text-to-Speech Module

Converts narration text to speech, supports multiple TTS engines
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Optional, Callable, Awaitable
from dataclasses import dataclass
import logging
import subprocess

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """TTS Result"""
    audio_path: str           # Audio file path
    duration: float           # Audio duration (seconds)
    text: str                 # Original text
    success: bool = True
    error: Optional[str] = None


class TTSEngine:
    """TTS Engine Base Class"""
    
    def __init__(self):
        self.config = get_config().tts
        self._cache_dir = get_config().paths.cache_dir / "tts"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        """
        Synthesize speech
        
        Args:
            text: Text to synthesize
            output_path: Output path, uses temp file if None
        
        Returns:
            TTSResult object
        """
        raise NotImplementedError
    
    def synthesize_sync(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        """Synchronous speech synthesis"""
        return asyncio.run(self.synthesize(text, output_path))
    
    def _get_temp_path(self) -> str:
        """Get temporary file path"""
        fd, path = tempfile.mkstemp(suffix=".mp3", dir=self._cache_dir)
        os.close(fd)
        return path
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration"""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}")
            # Estimate duration (about 5 Chinese chars per second)
            return 0.0


class EdgeTTSEngine(TTSEngine):
    """Microsoft Edge TTS Engine (free, high quality)"""
    
    async def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        try:
            import edge_tts
            
            output_path = output_path or self._get_temp_path()
            
            # Create communicate object
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.config.voice,
                rate=self._format_rate(self.config.speed)
            )
            
            # Save to file
            await communicate.save(output_path)
            
            # Get duration
            duration = self._get_audio_duration(output_path)
            
            return TTSResult(
                audio_path=output_path,
                duration=duration,
                text=text,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Edge TTS synthesis failed: {e}")
            return TTSResult(
                audio_path="",
                duration=0.0,
                text=text,
                success=False,
                error=str(e)
            )
    
    def _format_rate(self, speed: float) -> str:
        """Format speech rate"""
        if speed == 1.0:
            return "+0%"
        elif speed > 1.0:
            return f"+{int((speed - 1) * 100)}%"
        else:
            return f"-{int((1 - speed) * 100)}%"
    
    @staticmethod
    async def list_voices(language: str = "zh") -> list:
        """List available voices"""
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            return [v for v in voices if v["Locale"].startswith(language)]
        except Exception as e:
            logger.error(f"Failed to get voice list: {e}")
            return []


class GTTSEngine(TTSEngine):
    """Google TTS Engine"""
    
    async def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        try:
            from gtts import gTTS
            
            output_path = output_path or self._get_temp_path()
            
            # Use thread pool for sync operation
            await asyncio.to_thread(self._synthesize_sync, text, output_path)
            
            duration = self._get_audio_duration(output_path)
            
            return TTSResult(
                audio_path=output_path,
                duration=duration,
                text=text,
                success=True
            )
            
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            return TTSResult(
                audio_path="",
                duration=0.0,
                text=text,
                success=False,
                error=str(e)
            )
    
    def _synthesize_sync(self, text: str, output_path: str):
        from gtts import gTTS
        tts = gTTS(text=text, lang="zh-cn", slow=False)
        tts.save(output_path)


class Pyttsx3Engine(TTSEngine):
    """pyttsx3 Local TTS Engine"""
    
    def __init__(self):
        super().__init__()
        self._engine = None
    
    def _get_engine(self):
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', int(150 * self.config.speed))
        return self._engine
    
    async def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        try:
            output_path = output_path or self._get_temp_path()
            
            await asyncio.to_thread(self._synthesize_sync, text, output_path)
            
            duration = self._get_audio_duration(output_path)
            
            return TTSResult(
                audio_path=output_path,
                duration=duration,
                text=text,
                success=True
            )
            
        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return TTSResult(
                audio_path="",
                duration=0.0,
                text=text,
                success=False,
                error=str(e)
            )
    
    def _synthesize_sync(self, text: str, output_path: str):
        engine = self._get_engine()
        engine.save_to_file(text, output_path)
        engine.runAndWait()


class TTSManager:
    """TTS Manager"""
    
    def __init__(self, engine_type: Optional[str] = None):
        """
        Initialize TTS Manager
        
        Args:
            engine_type: Engine type (edge, gtts, pyttsx3)
        """
        config = get_config().tts
        engine_type = engine_type or config.engine
        
        if engine_type == "edge":
            self.engine = EdgeTTSEngine()
        elif engine_type == "gtts":
            self.engine = GTTSEngine()
        elif engine_type == "pyttsx3":
            self.engine = Pyttsx3Engine()
        else:
            logger.warning(f"Unknown TTS engine: {engine_type}, using Edge TTS")
            self.engine = EdgeTTSEngine()
        
        logger.info(f"TTS engine initialized: {engine_type}")
        
        # Audio playback callback
        self._play_callback: Optional[Callable[[str], Awaitable[None]]] = None
    
    def set_play_callback(self, callback: Callable[[str], Awaitable[None]]):
        """Set audio playback callback"""
        self._play_callback = callback
    
    async def speak(self, text: str) -> TTSResult:
        """
        Synthesize and play speech
        
        Args:
            text: Text to play
        
        Returns:
            TTSResult object
        """
        result = await self.engine.synthesize(text)
        
        if result.success and self._play_callback:
            await self._play_callback(result.audio_path)
        
        return result
    
    def speak_sync(self, text: str) -> TTSResult:
        """Synchronous synthesis and playback"""
        return asyncio.run(self.speak(text))
    
    async def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        """Synthesize speech (without playback)"""
        return await self.engine.synthesize(text, output_path)


class AudioPlayer:
    """Audio Player"""
    
    def __init__(self):
        self._playing = False
        self._current_process: Optional[subprocess.Popen] = None
    
    async def play(self, audio_path: str, block: bool = True):
        """
        Play audio
        
        Args:
            audio_path: Audio file path
            block: Whether to block until playback completes
        """
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return
        
        self._playing = True
        
        try:
            # macOS
            if os.name == "posix" and os.uname().sysname == "Darwin":
                cmd = ["afplay", audio_path]
            # Linux
            elif os.name == "posix":
                cmd = ["aplay", audio_path]
            # Windows
            else:
                cmd = ["powershell", "-c", f'(New-Object Media.SoundPlayer "{audio_path}").PlaySync()']
            
            if block:
                await asyncio.to_thread(subprocess.run, cmd, check=True)
            else:
                self._current_process = subprocess.Popen(cmd)
                
        except Exception as e:
            logger.error(f"Audio playback failed: {e}")
        finally:
            self._playing = False
    
    def stop(self):
        """Stop playback"""
        if self._current_process:
            self._current_process.terminate()
            self._current_process = None
        self._playing = False
    
    @property
    def is_playing(self) -> bool:
        """Check if playing"""
        return self._playing


class NarrationPlayer:
    """Narration Player - pre-generates and plays narration"""
    
    def __init__(self, tts_manager: TTSManager):
        """
        Initialize narration player
        
        Args:
            tts_manager: TTS manager
        """
        self.tts_manager = tts_manager
        self.audio_player = AudioPlayer()
        
        # Pre-generated audio cache
        self._cache: Dict[str, TTSResult] = {}
    
    async def prepare_narration(self, text: str, narration_id: str):
        """
        Pre-generate narration audio
        
        Args:
            text: Narration text
            narration_id: Narration ID
        """
        if narration_id in self._cache:
            return
        
        result = await self.tts_manager.synthesize(text)
        
        if result.success:
            self._cache[narration_id] = result
            logger.debug(f"Pre-generated narration: {narration_id}")
    
    async def play_narration(self, narration_id: str):
        """
        Play pre-generated narration
        
        Args:
            narration_id: Narration ID
        """
        if narration_id not in self._cache:
            logger.warning(f"Narration not found in cache: {narration_id}")
            return
        
        result = self._cache[narration_id]
        await self.audio_player.play(result.audio_path)
    
    def get_narration_duration(self, narration_id: str) -> float:
        """Get pre-generated narration duration"""
        if narration_id in self._cache:
            return self._cache[narration_id].duration
        return 0.0
    
    def clear_cache(self):
        """Clear cache"""
        # Delete audio files
        for result in self._cache.values():
            if os.path.exists(result.audio_path):
                try:
                    os.remove(result.audio_path)
                except:
                    pass
        
        self._cache.clear()
        logger.info("Narration cache cleared")


# For backward compatibility
from typing import Dict
