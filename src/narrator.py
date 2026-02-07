"""
讲解生成模块

根据场景分析结果生成自然语言讲解，控制讲解节奏和风格
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import logging
from enum import Enum

from .scene_analyzer import SceneAnalysis
from .config import get_config

logger = logging.getLogger(__name__)


class NarrationStyle(Enum):
    """讲解风格"""
    CONCISE = "简洁"       # 简洁明了
    DETAILED = "详细"      # 详细描述
    DRAMATIC = "电影解说"  # 电影解说风格
    NEUTRAL = "中性"       # 中性客观


@dataclass
class Narration:
    """讲解内容"""
    text: str                    # 讲解文本
    start_time: float            # 开始时间
    end_time: float              # 结束时间
    priority: int = 1            # 优先级 (1-5, 5最高)
    style: NarrationStyle = NarrationStyle.CONCISE
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class NarrationQueue:
    """讲解队列"""
    narrations: List[Narration] = field(default_factory=list)
    current_index: int = 0
    
    def add(self, narration: Narration):
        """添加讲解"""
        self.narrations.append(narration)
        # 按开始时间排序
        self.narrations.sort(key=lambda x: x.start_time)
    
    def get_current(self, timestamp: float) -> Optional[Narration]:
        """获取当前应播放的讲解"""
        for narration in self.narrations:
            if narration.start_time <= timestamp <= narration.end_time:
                return narration
        return None
    
    def get_next(self, timestamp: float) -> Optional[Narration]:
        """获取下一个讲解"""
        for narration in self.narrations:
            if narration.start_time > timestamp:
                return narration
        return None
    
    def clear(self):
        """清空队列"""
        self.narrations.clear()
        self.current_index = 0


class Narrator:
    """讲解生成器"""
    
    def __init__(self, style: NarrationStyle = NarrationStyle.CONCISE):
        """
        初始化讲解生成器
        
        Args:
            style: 讲解风格
        """
        self.config = get_config().narration
        self.style = style
        
        # 讲解队列
        self.queue = NarrationQueue()
        
        # 上一次讲解结束时间
        self._last_narration_end: float = 0.0
        
        # 讲解历史
        self._history: List[Narration] = []
        
        # 重复内容检测
        self._recent_descriptions: List[str] = []
    
    def set_style(self, style: NarrationStyle):
        """设置讲解风格"""
        self.style = style
        logger.info(f"讲解风格已设置为: {style.value}")
    
    def should_narrate(self, timestamp: float) -> bool:
        """
        判断是否应该在此时间点进行讲解
        
        Args:
            timestamp: 当前时间戳
        
        Returns:
            是否应该讲解
        """
        # 检查与上次讲解的间隔
        if timestamp - self._last_narration_end < self.config.interval:
            return False
        
        return True
    
    def generate_narration(
        self,
        scene_analysis: SceneAnalysis,
        slot: Tuple[float, float],
        characters_in_frame: Optional[List[str]] = None
    ) -> Optional[Narration]:
        """
        生成讲解内容
        
        Args:
            scene_analysis: 场景分析结果
            slot: 讲解时间槽 (start, end)
            characters_in_frame: 画面中的角色
        
        Returns:
            Narration 对象，如果不需要讲解则返回 None
        """
        start_time, end_time = slot
        
        # 检查是否应该讲解
        if not self.should_narrate(start_time):
            return None
        
        # 获取描述文本
        text = scene_analysis.description
        
        if not text or text == "[场景分析暂时不可用]":
            return None
        
        # 过滤掉AI拒绝回复
        refusal_keywords = ["对不起", "抱歉", "无法", "不能", "sorry", "cannot", "can't"]
        if any(keyword in text.lower() for keyword in refusal_keywords):
            logger.debug(f"跳过拒绝回复: {text[:30]}...")
            return None
        
        # 检查是否与最近的描述重复
        if self._is_duplicate(text):
            logger.debug("跳过重复内容")
            return None
        
        # 根据风格调整文本
        text = self._adjust_for_style(text)
        
        # 确保文本长度合适
        text = self._trim_text(text, end_time - start_time)
        
        narration = Narration(
            text=text,
            start_time=start_time,
            end_time=end_time,
            priority=self._calculate_priority(scene_analysis),
            style=self.style
        )
        
        # 更新状态
        self._last_narration_end = end_time
        self._recent_descriptions.append(text)
        if len(self._recent_descriptions) > 10:
            self._recent_descriptions.pop(0)
        
        # 添加到队列和历史
        self.queue.add(narration)
        self._history.append(narration)
        
        logger.info(f"生成讲解 [{start_time:.1f}s - {end_time:.1f}s]: {text[:50]}...")
        
        return narration
    
    def _is_duplicate(self, text: str) -> bool:
        """检查是否与最近的描述重复"""
        if not self._recent_descriptions:
            return False
        
        # 简单的重复检测：检查关键词重叠
        text_words = set(text)
        
        for recent in self._recent_descriptions[-3:]:
            recent_words = set(recent)
            overlap = len(text_words & recent_words) / max(len(text_words), 1)
            if overlap > 0.7:  # 70% 重叠认为是重复
                return True
        
        return False
    
    def _adjust_for_style(self, text: str) -> str:
        """根据风格调整文本"""
        if self.style == NarrationStyle.CONCISE:
            # 移除冗余词汇
            text = text.replace("正在", "")
            text = text.replace("正", "")
            text = text.replace("似乎", "")
        
        elif self.style == NarrationStyle.DRAMATIC:
            # 添加戏剧性元素（可选）
            pass
        
        return text.strip()
    
    def _trim_text(self, text: str, available_duration: float) -> str:
        """
        根据可用时间裁剪文本
        
        假设平均语速为每秒6-8个中文字（专业配音可以达到更快）
        """
        chars_per_second = 7  # 专业配音语速
        max_chars = int(available_duration * chars_per_second)
        max_chars = max(max_chars, 15)  # 至少保留15个字，确保描述完整
        max_chars = min(max_chars, self.config.max_length)
        
        if len(text) <= max_chars:
            return text
        
        # 尝试在标点处截断
        punctuations = ["。", "，", "；", "！", "？", "、"]
        
        for i in range(max_chars - 1, max_chars // 2, -1):
            if text[i] in punctuations:
                return text[:i + 1]
        
        # 否则直接截断
        return text[:max_chars - 1] + "。"
    
    def _calculate_priority(self, scene_analysis: SceneAnalysis) -> int:
        """计算讲解优先级"""
        priority = 1
        
        # 场景切换时优先级更高
        if scene_analysis.confidence > 0.8:
            priority += 1
        
        # 有动作描述时优先级更高
        if scene_analysis.actions:
            priority += 1
        
        # 有情感变化时优先级更高
        if scene_analysis.emotions:
            priority += 1
        
        return min(priority, 5)
    
    def format_with_characters(
        self,
        text: str,
        characters: List[str]
    ) -> str:
        """
        格式化文本，确保角色名称正确使用
        
        Args:
            text: 原始文本
            characters: 已知角色列表
        
        Returns:
            格式化后的文本
        """
        # 替换泛指代词为角色名称
        replacements = [
            ("一个男人", characters[0] if characters else "男子"),
            ("一个女人", characters[1] if len(characters) > 1 else "女子"),
            ("那个男人", characters[0] if characters else "他"),
            ("那个女人", characters[1] if len(characters) > 1 else "她"),
            ("一名男子", characters[0] if characters else "男子"),
            ("一名女子", characters[1] if len(characters) > 1 else "女子"),
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        return text
    
    def get_narration_at(self, timestamp: float) -> Optional[Narration]:
        """获取指定时间的讲解"""
        return self.queue.get_current(timestamp)
    
    def get_next_narration(self, timestamp: float) -> Optional[Narration]:
        """获取下一个讲解"""
        return self.queue.get_next(timestamp)
    
    def get_history(self, limit: int = 10) -> List[Narration]:
        """获取讲解历史"""
        return self._history[-limit:]
    
    def clear(self):
        """清除所有状态"""
        self.queue.clear()
        self._history.clear()
        self._recent_descriptions.clear()
        self._last_narration_end = 0.0
    
    def export_subtitles(self, output_path: str, format: str = "srt"):
        """
        导出讲解为字幕文件
        
        Args:
            output_path: 输出路径
            format: 字幕格式 (srt, vtt)
        """
        if format == "srt":
            self._export_srt(output_path)
        elif format == "vtt":
            self._export_vtt(output_path)
        else:
            raise ValueError(f"不支持的字幕格式: {format}")
    
    def _export_srt(self, output_path: str):
        """导出为 SRT 格式"""
        lines = []
        
        for i, narration in enumerate(self._history, 1):
            start = self._format_time_srt(narration.start_time)
            end = self._format_time_srt(narration.end_time)
            
            lines.append(str(i))
            lines.append(f"{start} --> {end}")
            lines.append(narration.text)
            lines.append("")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        logger.info(f"已导出 SRT 字幕: {output_path}")
    
    def _export_vtt(self, output_path: str):
        """导出为 WebVTT 格式"""
        lines = ["WEBVTT", ""]
        
        for i, narration in enumerate(self._history, 1):
            start = self._format_time_vtt(narration.start_time)
            end = self._format_time_vtt(narration.end_time)
            
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(narration.text)
            lines.append("")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        
        logger.info(f"已导出 VTT 字幕: {output_path}")
    
    @staticmethod
    def _format_time_srt(seconds: float) -> str:
        """格式化为 SRT 时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @staticmethod
    def _format_time_vtt(seconds: float) -> str:
        """格式化为 VTT 时间格式"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
