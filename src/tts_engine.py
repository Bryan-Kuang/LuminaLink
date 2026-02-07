"""
语音合成模块

将讲解文本转换为语音，支持多种 TTS 引擎
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
    """TTS 结果"""
    audio_path: str           # 音频文件路径
    duration: float           # 音频时长（秒）
    text: str                 # 原始文本
    success: bool = True
    error: Optional[str] = None


class TTSEngine:
    """TTS 引擎基类"""
    
    def __init__(self):
        self.config = get_config().tts
        self._cache_dir = get_config().paths.cache_dir / "tts"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
    
    async def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        """
        合成语音
        
        Args:
            text: 要合成的文本
            output_path: 输出路径，如果为 None 则使用临时文件
        
        Returns:
            TTSResult 对象
        """
        raise NotImplementedError
    
    def synthesize_sync(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        """同步合成语音"""
        return asyncio.run(self.synthesize(text, output_path))
    
    def _get_temp_path(self) -> str:
        """获取临时文件路径"""
        fd, path = tempfile.mkstemp(suffix=".mp3", dir=self._cache_dir)
        os.close(fd)
        return path
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """获取音频时长"""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0
        except Exception as e:
            logger.warning(f"无法获取音频时长: {e}")
            # 估算时长（每秒约5个中文字）
            return 0.0


class EdgeTTSEngine(TTSEngine):
    """微软 Edge TTS 引擎（免费，高质量）"""
    
    async def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        try:
            import edge_tts
            
            output_path = output_path or self._get_temp_path()
            
            # 创建通信对象
            communicate = edge_tts.Communicate(
                text=text,
                voice=self.config.voice,
                rate=self._format_rate(self.config.speed)
            )
            
            # 保存到文件
            await communicate.save(output_path)
            
            # 获取时长
            duration = self._get_audio_duration(output_path)
            
            return TTSResult(
                audio_path=output_path,
                duration=duration,
                text=text,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Edge TTS 合成失败: {e}")
            return TTSResult(
                audio_path="",
                duration=0.0,
                text=text,
                success=False,
                error=str(e)
            )
    
    def _format_rate(self, speed: float) -> str:
        """格式化语速"""
        if speed == 1.0:
            return "+0%"
        elif speed > 1.0:
            return f"+{int((speed - 1) * 100)}%"
        else:
            return f"-{int((1 - speed) * 100)}%"
    
    @staticmethod
    async def list_voices(language: str = "zh") -> list:
        """列出可用的语音"""
        try:
            import edge_tts
            voices = await edge_tts.list_voices()
            return [v for v in voices if v["Locale"].startswith(language)]
        except Exception as e:
            logger.error(f"获取语音列表失败: {e}")
            return []


class GTTSEngine(TTSEngine):
    """Google TTS 引擎"""
    
    async def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        try:
            from gtts import gTTS
            
            output_path = output_path or self._get_temp_path()
            
            # 使用线程池执行同步操作
            await asyncio.to_thread(self._synthesize_sync, text, output_path)
            
            duration = self._get_audio_duration(output_path)
            
            return TTSResult(
                audio_path=output_path,
                duration=duration,
                text=text,
                success=True
            )
            
        except Exception as e:
            logger.error(f"gTTS 合成失败: {e}")
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
    """pyttsx3 本地 TTS 引擎"""
    
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
            logger.error(f"pyttsx3 合成失败: {e}")
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
    """TTS 管理器"""
    
    def __init__(self, engine_type: Optional[str] = None):
        """
        初始化 TTS 管理器
        
        Args:
            engine_type: 引擎类型 (edge, gtts, pyttsx3)
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
            logger.warning(f"未知的 TTS 引擎: {engine_type}，使用 Edge TTS")
            self.engine = EdgeTTSEngine()
        
        logger.info(f"TTS 引擎已初始化: {engine_type}")
        
        # 音频播放回调
        self._play_callback: Optional[Callable[[str], Awaitable[None]]] = None
    
    def set_play_callback(self, callback: Callable[[str], Awaitable[None]]):
        """设置音频播放回调"""
        self._play_callback = callback
    
    async def speak(self, text: str) -> TTSResult:
        """
        合成并播放语音
        
        Args:
            text: 要播放的文本
        
        Returns:
            TTSResult 对象
        """
        result = await self.engine.synthesize(text)
        
        if result.success and self._play_callback:
            await self._play_callback(result.audio_path)
        
        return result
    
    def speak_sync(self, text: str) -> TTSResult:
        """同步合成并播放"""
        return asyncio.run(self.speak(text))
    
    async def synthesize(self, text: str, output_path: Optional[str] = None) -> TTSResult:
        """合成语音（不播放）"""
        return await self.engine.synthesize(text, output_path)


class AudioPlayer:
    """音频播放器"""
    
    def __init__(self):
        self._playing = False
        self._current_process: Optional[subprocess.Popen] = None
    
    async def play(self, audio_path: str, block: bool = True):
        """
        播放音频
        
        Args:
            audio_path: 音频文件路径
            block: 是否阻塞直到播放完成
        """
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
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
                cmd = ["powershell", "-c", f"(New-Object Media.SoundPlayer '{audio_path}').PlaySync()"]
            
            if block:
                await asyncio.to_thread(subprocess.run, cmd, capture_output=True)
            else:
                self._current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
        except Exception as e:
            logger.error(f"播放音频失败: {e}")
        finally:
            self._playing = False
    
    def stop(self):
        """停止播放"""
        if self._current_process:
            self._current_process.terminate()
            self._current_process = None
        self._playing = False
    
    def is_playing(self) -> bool:
        """是否正在播放"""
        return self._playing


class NarrationPlayer:
    """讲解音频播放器"""
    
    def __init__(self, tts_manager: Optional[TTSManager] = None):
        self.tts = tts_manager or TTSManager()
        self.player = AudioPlayer()
        
        # 预生成的音频缓存
        self._audio_cache: dict = {}
    
    async def prepare_narration(self, text: str, narration_id: str) -> Optional[str]:
        """
        预生成讲解音频
        
        Args:
            text: 讲解文本
            narration_id: 讲解 ID
        
        Returns:
            音频文件路径
        """
        if narration_id in self._audio_cache:
            return self._audio_cache[narration_id]
        
        result = await self.tts.synthesize(text)
        
        if result.success:
            self._audio_cache[narration_id] = result.audio_path
            return result.audio_path
        
        return None
    
    async def play_narration(self, text: str, block: bool = True) -> bool:
        """
        播放讲解
        
        Args:
            text: 讲解文本
            block: 是否阻塞
        
        Returns:
            是否成功
        """
        result = await self.tts.synthesize(text)
        
        if result.success:
            await self.player.play(result.audio_path, block)
            return True
        
        return False
    
    async def play_prepared(self, narration_id: str, block: bool = True) -> bool:
        """
        播放预生成的讲解
        
        Args:
            narration_id: 讲解 ID
            block: 是否阻塞
        
        Returns:
            是否成功
        """
        if narration_id not in self._audio_cache:
            return False
        
        await self.player.play(self._audio_cache[narration_id], block)
        return True
    
    def stop(self):
        """停止播放"""
        self.player.stop()
    
    def clear_cache(self):
        """清除音频缓存"""
        for path in self._audio_cache.values():
            try:
                os.remove(path)
            except:
                pass
        self._audio_cache.clear()
