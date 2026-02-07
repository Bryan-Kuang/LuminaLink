"""
音频检测模块

检测视频中的对话、静音片段，用于确定何时插入讲解
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
    """音频片段"""
    start_time: float    # 开始时间（秒）
    end_time: float      # 结束时间（秒）
    has_speech: bool     # 是否包含语音
    volume_db: float     # 平均音量（dB）
    
    @property
    def duration(self) -> float:
        """片段时长"""
        return self.end_time - self.start_time


@dataclass
class SilenceWindow:
    """静音窗口 - 适合插入讲解的时间段"""
    start_time: float
    end_time: float
    confidence: float  # 置信度 (0-1)
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class AudioDetector:
    """音频检测器"""
    
    def __init__(
        self,
        silence_threshold_db: float = -40.0,
        min_silence_duration: float = 1.0,
        min_narration_gap: float = 3.0
    ):
        """
        初始化音频检测器
        
        Args:
            silence_threshold_db: 静音阈值（dB）
            min_silence_duration: 最小静音时长（秒）
            min_narration_gap: 讲解之间的最小间隔（秒）
        """
        self.silence_threshold_db = silence_threshold_db
        self.min_silence_duration = min_silence_duration
        self.min_narration_gap = min_narration_gap
        
        self._audio_data: Optional[np.ndarray] = None
        self._sample_rate: int = 22050
    
    def load_audio_from_video(self, video_path: str) -> bool:
        """
        从视频文件中提取音频
        
        Args:
            video_path: 视频文件路径
        
        Returns:
            是否成功加载
        """
        try:
            from moviepy import VideoFileClip
            import librosa
            
            logger.info(f"正在从视频提取音频: {video_path}")
            
            # 创建临时音频文件
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            
            try:
                # 提取音频
                video = VideoFileClip(video_path)
                video.audio.write_audiofile(
                    tmp_path, 
                    fps=self._sample_rate,
                    logger=None
                )
                video.close()
                
                # 加载音频数据
                self._audio_data, self._sample_rate = librosa.load(
                    tmp_path, 
                    sr=self._sample_rate,
                    mono=True
                )
                
                logger.info(f"音频加载成功，时长: {len(self._audio_data) / self._sample_rate:.2f}秒")
                return True
                
            finally:
                # 清理临时文件
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                    
        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            return False
    
    def load_audio_file(self, audio_path: str) -> bool:
        """
        直接加载音频文件
        
        Args:
            audio_path: 音频文件路径
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
            logger.error(f"音频加载失败: {e}")
            return False
    
    def analyze_audio(
        self,
        window_size: float = 0.5
    ) -> List[AudioSegment]:
        """
        分析音频，识别语音和静音片段
        
        Args:
            window_size: 分析窗口大小（秒）
        
        Returns:
            AudioSegment 列表
        """
        if self._audio_data is None:
            logger.error("音频数据未加载")
            return []
        
        segments = []
        samples_per_window = int(self._sample_rate * window_size)
        total_samples = len(self._audio_data)
        
        for i in range(0, total_samples, samples_per_window):
            window = self._audio_data[i:i + samples_per_window]
            
            if len(window) == 0:
                continue
            
            # 计算RMS能量
            rms = np.sqrt(np.mean(window ** 2))
            
            # 转换为dB
            if rms > 0:
                volume_db = 20 * np.log10(rms)
            else:
                volume_db = -100
            
            start_time = i / self._sample_rate
            end_time = min((i + samples_per_window) / self._sample_rate, 
                          total_samples / self._sample_rate)
            
            # 判断是否有语音
            has_speech = volume_db > self.silence_threshold_db
            
            segments.append(AudioSegment(
                start_time=start_time,
                end_time=end_time,
                has_speech=has_speech,
                volume_db=volume_db
            ))
        
        logger.info(f"音频分析完成，共 {len(segments)} 个片段")
        return segments
    
    def find_silence_windows(
        self,
        segments: Optional[List[AudioSegment]] = None
    ) -> List[SilenceWindow]:
        """
        找出适合插入讲解的静音窗口
        
        Args:
            segments: 音频片段列表，如果为 None 则先分析音频
        
        Returns:
            SilenceWindow 列表
        """
        if segments is None:
            segments = self.analyze_audio()
        
        if not segments:
            return []
        
        windows = []
        silence_start: Optional[float] = None
        
        for segment in segments:
            if not segment.has_speech:
                # 开始静音
                if silence_start is None:
                    silence_start = segment.start_time
            else:
                # 语音开始，结束静音
                if silence_start is not None:
                    silence_end = segment.start_time
                    duration = silence_end - silence_start
                    
                    if duration >= self.min_silence_duration:
                        # 计算置信度（基于静音时长）
                        confidence = min(duration / 5.0, 1.0)
                        
                        windows.append(SilenceWindow(
                            start_time=silence_start,
                            end_time=silence_end,
                            confidence=confidence
                        ))
                    
                    silence_start = None
        
        # 处理末尾的静音
        if silence_start is not None:
            silence_end = segments[-1].end_time
            duration = silence_end - silence_start
            
            if duration >= self.min_silence_duration:
                windows.append(SilenceWindow(
                    start_time=silence_start,
                    end_time=silence_end,
                    confidence=min(duration / 5.0, 1.0)
                ))
        
        logger.info(f"找到 {len(windows)} 个静音窗口")
        return windows
    
    def get_narration_slots(
        self,
        windows: Optional[List[SilenceWindow]] = None,
        max_narration_duration: float = 10.0
    ) -> List[Tuple[float, float]]:
        """
        获取讲解时间槽
        
        返回适合插入讲解的时间段，考虑讲解间隔和最大时长
        
        Args:
            windows: 静音窗口列表
            max_narration_duration: 最大讲解时长
        
        Returns:
            (start_time, end_time) 元组列表
        """
        if windows is None:
            windows = self.find_silence_windows()
        
        slots = []
        last_narration_end = 0.0
        
        for window in windows:
            # 检查与上次讲解的间隔
            if window.start_time < last_narration_end + self.min_narration_gap:
                continue
            
            # 计算讲解时长
            available_duration = window.duration
            narration_duration = min(available_duration * 0.8, max_narration_duration)
            
            if narration_duration >= 1.0:  # 至少1秒
                slot_start = window.start_time + 0.3  # 留一点缓冲
                slot_end = slot_start + narration_duration
                
                slots.append((slot_start, slot_end))
                last_narration_end = slot_end
        
        logger.info(f"生成 {len(slots)} 个讲解时间槽")
        return slots
    
    def is_silence_at(self, timestamp: float, window_size: float = 0.5) -> bool:
        """
        检查指定时间是否为静音
        
        Args:
            timestamp: 时间戳（秒）
            window_size: 检测窗口大小
        
        Returns:
            是否为静音
        """
        if self._audio_data is None:
            return True
        
        start_sample = int(timestamp * self._sample_rate)
        end_sample = int((timestamp + window_size) * self._sample_rate)
        
        if start_sample >= len(self._audio_data):
            return True
        
        window = self._audio_data[start_sample:end_sample]
        
        if len(window) == 0:
            return True
        
        rms = np.sqrt(np.mean(window ** 2))
        volume_db = 20 * np.log10(rms) if rms > 0 else -100
        
        return volume_db <= self.silence_threshold_db
    
    def get_audio_duration(self) -> float:
        """获取音频总时长"""
        if self._audio_data is None:
            return 0.0
        return len(self._audio_data) / self._sample_rate


class RealtimeAudioDetector:
    """实时音频检测器"""
    
    def __init__(self, silence_threshold_db: float = -40.0):
        self.silence_threshold_db = silence_threshold_db
        self._buffer: List[float] = []
        self._buffer_duration: float = 1.0  # 1秒缓冲
    
    def add_samples(self, samples: np.ndarray, sample_rate: int):
        """添加音频样本"""
        # 计算RMS
        rms = np.sqrt(np.mean(samples ** 2))
        volume_db = 20 * np.log10(rms) if rms > 0 else -100
        
        self._buffer.append(volume_db)
        
        # 保持缓冲区大小
        max_size = int(self._buffer_duration * sample_rate / len(samples))
        if len(self._buffer) > max_size:
            self._buffer = self._buffer[-max_size:]
    
    def is_current_silence(self) -> bool:
        """当前是否为静音"""
        if not self._buffer:
            return True
        
        avg_volume = np.mean(self._buffer[-10:])  # 最近10个窗口
        return avg_volume <= self.silence_threshold_db
    
    def get_silence_duration(self) -> float:
        """获取当前静音持续时间"""
        if not self._buffer:
            return 0.0
        
        silence_count = 0
        for db in reversed(self._buffer):
            if db <= self.silence_threshold_db:
                silence_count += 1
            else:
                break
        
        return silence_count * (self._buffer_duration / len(self._buffer))
