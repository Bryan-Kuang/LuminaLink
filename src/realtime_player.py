"""
实时播放模块

实时播放视频并同步插入讲解
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
    """播放状态"""
    is_playing: bool = False
    is_paused: bool = False
    current_time: float = 0.0
    duration: float = 0.0
    current_narration: Optional[str] = None


class RealtimePlayer:
    """实时播放器"""
    
    def __init__(
        self,
        video_path: str,
        character_recognizer: Optional[CharacterRecognizer] = None
    ):
        """
        初始化实时播放器
        
        Args:
            video_path: 视频路径
            character_recognizer: 角色识别器
        """
        self.video_path = video_path
        self.config = get_config()
        
        # 组件
        self.video_processor = VideoProcessor(video_path)
        self.character_recognizer = character_recognizer or CharacterRecognizer()
        self.character_tracker = CharacterTracker(self.character_recognizer)
        self.scene_analyzer = SceneAnalyzer()
        self.narrator = Narrator()
        self.tts_manager = TTSManager()
        self.audio_player = AudioPlayer()
        self.audio_detector = RealtimeAudioDetector()
        
        # 状态
        self.state = PlaybackState()
        
        # 线程控制
        self._stop_event = Event()
        self._pause_event = Event()
        self._pause_event.set()  # 默认不暂停
        
        # 任务队列
        self._analysis_queue: Queue = Queue(maxsize=10)
        self._narration_queue: Queue = Queue(maxsize=10)
        
        # 回调
        self._on_frame_callback: Optional[Callable[[np.ndarray], None]] = None
        self._on_narration_callback: Optional[Callable[[str], None]] = None
    
    def set_on_frame_callback(self, callback: Callable[[np.ndarray], None]):
        """设置帧回调"""
        self._on_frame_callback = callback
    
    def set_on_narration_callback(self, callback: Callable[[str], None]):
        """设置讲解回调"""
        self._on_narration_callback = callback
    
    def start(self):
        """开始播放"""
        if not self.video_processor.open():
            raise RuntimeError("无法打开视频")
        
        video_info = self.video_processor.get_info()
        if video_info:
            self.state.duration = video_info.duration
        
        self.state.is_playing = True
        self._stop_event.clear()
        
        # 启动工作线程
        Thread(target=self._analysis_worker, daemon=True).start()
        Thread(target=self._narration_worker, daemon=True).start()
        
        # 主播放循环
        self._playback_loop()
    
    def stop(self):
        """停止播放"""
        self._stop_event.set()
        self.state.is_playing = False
        self.video_processor.close()
        self.audio_player.stop()
    
    def pause(self):
        """暂停"""
        self._pause_event.clear()
        self.state.is_paused = True
    
    def resume(self):
        """继续"""
        self._pause_event.set()
        self.state.is_paused = False
    
    def seek(self, timestamp: float):
        """跳转"""
        self.video_processor.seek(timestamp)
        self.state.current_time = timestamp
        self.scene_analyzer.clear_context()
    
    def _playback_loop(self):
        """主播放循环"""
        video_info = self.video_processor.get_info()
        fps = video_info.fps if video_info else 30.0
        frame_duration = 1.0 / fps
        
        last_analysis_time = 0.0
        analysis_interval = self.config.video.keyframe_interval
        
        while not self._stop_event.is_set():
            # 检查暂停
            self._pause_event.wait()
            
            # 读取帧
            frame = self.video_processor.read_frame()
            if frame is None:
                logger.info("视频播放完成")
                break
            
            self.state.current_time = frame.timestamp
            
            # 显示帧
            if self._on_frame_callback:
                self._on_frame_callback(frame.frame)
            
            # 定期进行场景分析
            if frame.timestamp - last_analysis_time >= analysis_interval:
                try:
                    self._analysis_queue.put_nowait((frame, frame.timestamp))
                    last_analysis_time = frame.timestamp
                except:
                    pass  # 队列满，跳过
            
            # 控制帧率
            time.sleep(frame_duration)
        
        self.state.is_playing = False
    
    def _analysis_worker(self):
        """分析工作线程"""
        while not self._stop_event.is_set():
            try:
                frame, timestamp = self._analysis_queue.get(timeout=1.0)
            except Empty:
                continue
            
            try:
                # 识别角色
                characters = self.character_recognizer.get_characters_in_frame(
                    frame.frame,
                    timestamp=timestamp
                )
                
                # 检查是否应该讲解
                if not self.narrator.should_narrate(timestamp):
                    continue
                
                # 检查是否静音
                if not self.audio_detector.is_current_silence():
                    continue
                
                # 分析场景
                analysis = asyncio.run(
                    self.scene_analyzer.analyze_frame_async(
                        frame.frame,
                        characters_in_frame=characters,
                        timestamp=timestamp
                    )
                )
                
                # 生成讲解
                estimated_duration = 5.0  # 估计的讲解时长
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
                logger.error(f"分析失败: {e}")
    
    def _narration_worker(self):
        """讲解工作线程"""
        while not self._stop_event.is_set():
            try:
                narration = self._narration_queue.get(timeout=1.0)
            except Empty:
                continue
            
            try:
                self.state.current_narration = narration.text
                
                # 回调
                if self._on_narration_callback:
                    self._on_narration_callback(narration.text)
                
                # 合成并播放语音
                result = asyncio.run(self.tts_manager.synthesize(narration.text))
                
                if result.success:
                    asyncio.run(self.audio_player.play(result.audio_path))
                
                self.state.current_narration = None
                
            except Exception as e:
                logger.error(f"讲解播放失败: {e}")


class PreviewWindow:
    """预览窗口"""
    
    def __init__(self, title: str = "LuminaLink Preview"):
        self.title = title
        self.window_created = False
        self._current_narration: Optional[str] = None
    
    def show_frame(self, frame: np.ndarray):
        """显示帧"""
        if not self.window_created:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            self.window_created = True
        
        display_frame = frame.copy()
        
        # 添加讲解字幕
        if self._current_narration:
            self._add_subtitle(display_frame, self._current_narration)
        
        cv2.imshow(self.title, display_frame)
        cv2.waitKey(1)
    
    def set_narration(self, text: str):
        """设置当前讲解"""
        self._current_narration = text
    
    def clear_narration(self):
        """清除讲解"""
        self._current_narration = None
    
    def _add_subtitle(self, frame: np.ndarray, text: str):
        """添加字幕"""
        height, width = frame.shape[:2]
        
        # 字幕背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        
        # 计算文本大小
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 绘制半透明背景
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (0, height - 60),
            (width, height),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # 绘制文本
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
        """关闭窗口"""
        if self.window_created:
            cv2.destroyWindow(self.title)
            self.window_created = False


def run_realtime_player(
    video_path: str,
    characters_config: Optional[str] = None,
    show_preview: bool = True
):
    """
    运行实时播放器
    
    Args:
        video_path: 视频路径
        characters_config: 角色配置
        show_preview: 是否显示预览
    """
    from rich.console import Console
    console = Console()
    
    # 初始化角色识别
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
    
    # 创建播放器
    player = RealtimePlayer(video_path, recognizer)
    
    # 设置预览
    preview = PreviewWindow() if show_preview else None
    
    if preview:
        player.set_on_frame_callback(preview.show_frame)
        player.set_on_narration_callback(lambda t: (
            preview.set_narration(t),
            console.print(f"[green]讲解:[/green] {t}")
        ))
    
    console.print("[bold blue]开始实时播放...[/bold blue]")
    console.print("按 Q 退出\n")
    
    try:
        player.start()
    except KeyboardInterrupt:
        pass
    finally:
        player.stop()
        if preview:
            preview.close()
        console.print("\n[yellow]播放结束[/yellow]")
