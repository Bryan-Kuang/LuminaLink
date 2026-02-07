"""
Narration Generation Module

Generates natural language narration based on scene analysis results, controls narration rhythm and style
"""

from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import logging
from enum import Enum

from .scene_analyzer import SceneAnalysis
from .config import get_config

logger = logging.getLogger(__name__)


class NarrationStyle(Enum):
    """Narration style"""
    CONCISE = "concise"       # Brief and clear
    DETAILED = "detailed"      # Detailed description
    DRAMATIC = "cinematic"     # Cinematic narration style
    NEUTRAL = "neutral"        # Neutral and objective


@dataclass
class Narration:
    """Narration content"""
    text: str                    # Narration text
    start_time: float            # Start time
    end_time: float              # End time
    priority: int = 1            # Priority (1-5, 5 is highest)
    style: NarrationStyle = NarrationStyle.CONCISE
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


@dataclass
class NarrationQueue:
    """Narration queue"""
    narrations: List[Narration] = field(default_factory=list)
    current_index: int = 0
    
    def add(self, narration: Narration):
        """Add narration"""
        self.narrations.append(narration)
        # Sort by start time
        self.narrations.sort(key=lambda x: x.start_time)
    
    def get_current(self, timestamp: float) -> Optional[Narration]:
        """Get narration that should be playing at current time"""
        for narration in self.narrations:
            if narration.start_time <= timestamp <= narration.end_time:
                return narration
        return None
    
    def get_next(self, timestamp: float) -> Optional[Narration]:
        """Get next narration"""
        for narration in self.narrations:
            if narration.start_time > timestamp:
                return narration
        return None
    
    def clear(self):
        """Clear queue"""
        self.narrations.clear()
        self.current_index = 0


class Narrator:
    """Narration Generator"""
    
    def __init__(self, style: NarrationStyle = NarrationStyle.CONCISE):
        """
        Initialize narration generator
        
        Args:
            style: Narration style
        """
        self.config = get_config().narration
        self.style = style
        
        # Narration queue
        self.queue = NarrationQueue()
        
        # Last narration end time
        self._last_narration_end: float = 0.0
        
        # Narration history
        self._history: List[Narration] = []
        
        # Duplicate content detection
        self._recent_descriptions: List[str] = []
    
    def set_style(self, style: NarrationStyle):
        """Set narration style"""
        self.style = style
        logger.info(f"Narration style set to: {style.value}")
    
    def should_narrate(self, timestamp: float) -> bool:
        """
        Determine if narration should occur at this timestamp
        
        Args:
            timestamp: Current timestamp
        
        Returns:
            Whether to narrate
        """
        # Check interval from last narration
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
        Generate narration content
        
        Args:
            scene_analysis: Scene analysis result
            slot: Narration time slot (start, end)
            characters_in_frame: Characters in frame
        
        Returns:
            Narration object, None if no narration needed
        """
        start_time, end_time = slot
        
        # Check if should narrate
        if not self.should_narrate(start_time):
            return None
        
        # Get description text
        text = scene_analysis.description
        
        if not text or text == "[Scene analysis temporarily unavailable]":
            return None
        
        # Filter AI refusal responses (English and Chinese)
        refusal_keywords = [
            # English
            "sorry", "cannot", "can't", "unable", "i'm not able", "i cannot",
            "i can't", "i am unable", "i'm unable", "apologize", "apologies"
        ]
        if any(keyword in text.lower() for keyword in refusal_keywords):
            logger.debug(f"Skipping refusal response: {text[:30]}...")
            return None
        
        # Check if duplicate with recent descriptions
        if self._is_duplicate(text):
            logger.debug("Skipping duplicate content")
            return None
        
        # Adjust text based on style
        text = self._adjust_for_style(text)
        
        # Ensure text length is appropriate
        text = self._trim_text(text, end_time - start_time)
        
        narration = Narration(
            text=text,
            start_time=start_time,
            end_time=end_time,
            priority=self._calculate_priority(scene_analysis),
            style=self.style
        )
        
        # Update state
        self._last_narration_end = end_time
        self._recent_descriptions.append(text)
        if len(self._recent_descriptions) > 10:
            self._recent_descriptions.pop(0)
        
        # Add to queue and history
        self.queue.add(narration)
        self._history.append(narration)
        
        logger.info(f"Generated narration [{start_time:.1f}s - {end_time:.1f}s]: {text[:50]}...")
        
        return narration
    
    def _is_duplicate(self, text: str) -> bool:
        """Check if duplicate with recent descriptions"""
        if not self._recent_descriptions:
            return False
        
        # Simple duplicate detection: check word overlap
        text_words = set(text.lower().split())

        for recent in self._recent_descriptions[-3:]:
            recent_words = set(recent.lower().split())
            overlap = len(text_words & recent_words) / max(len(text_words), 1)
            if overlap > 0.7:  # 70% word overlap considered duplicate
                return True
        
        return False
    
    def _adjust_for_style(self, text: str) -> str:
        """Adjust text based on style"""
        if self.style == NarrationStyle.CONCISE:
            # Remove redundant words
            pass
        
        elif self.style == NarrationStyle.DRAMATIC:
            # Add dramatic elements (optional)
            pass
        
        return text.strip()
    
    def _trim_text(self, text: str, available_duration: float) -> str:
        """
        Trim text based on available time
        
        Speaking rate: ~150 words per minute = 2.5 words/sec
        Average word length: ~5 characters, so ~12-15 chars/sec for English
        For Chinese: ~6-8 chars/sec
        """
        # Detect if text is primarily English or Chinese
        ascii_count = sum(1 for c in text if ord(c) < 128)
        is_english = ascii_count / max(len(text), 1) > 0.5
        
        if is_english:
            chars_per_second = 15  # English speaking rate
        else:
            chars_per_second = 7   # Chinese speaking rate
        
        max_chars = int(available_duration * chars_per_second)
        max_chars = max(max_chars, 40)  # Keep at least 40 chars for meaningful description
        max_chars = min(max_chars, self.config.max_length)
        
        if len(text) <= max_chars:
            return text
        
        # Try to truncate at punctuation
        punctuations = [".", ",", ";", "!", "?", ":", "-"]
        
        for i in range(max_chars - 1, max_chars // 2, -1):
            if i < len(text) and text[i] in punctuations:
                return text[:i + 1]
        
        # Otherwise truncate directly
        return text[:max_chars - 1] + "."
    
    def _calculate_priority(self, scene_analysis: SceneAnalysis) -> int:
        """Calculate narration priority"""
        priority = 1
        
        # Higher priority on scene change
        if scene_analysis.confidence > 0.8:
            priority += 1
        
        # Higher priority with action description
        if scene_analysis.actions:
            priority += 1
        
        # Higher priority with emotional change
        if scene_analysis.emotions:
            priority += 1
        
        return min(priority, 5)
    
    def format_with_characters(
        self,
        text: str,
        characters: List[str]
    ) -> str:
        """
        Format text to ensure correct use of character names
        
        Args:
            text: Original text
            characters: Known character list
        
        Returns:
            Formatted text
        """
        # Replace generic pronouns with character names
        replacements = [
            ("a man", characters[0] if characters else "the man"),
            ("a woman", characters[1] if len(characters) > 1 else "the woman"),
            ("that man", characters[0] if characters else "he"),
            ("that woman", characters[1] if len(characters) > 1 else "she"),
        ]
        
        for old, new in replacements:
            text = text.replace(old, new)
        
        return text
    
    def get_narration_at(self, timestamp: float) -> Optional[Narration]:
        """Get narration at specified time"""
        return self.queue.get_current(timestamp)
    
    def get_next_narration(self, timestamp: float) -> Optional[Narration]:
        """Get next narration"""
        return self.queue.get_next(timestamp)
    
    def get_history(self, limit: int = 10) -> List[Narration]:
        """Get narration history"""
        return self._history[-limit:]
    
    def clear(self):
        """Clear all state"""
        self.queue.clear()
        self._history.clear()
        self._recent_descriptions.clear()
        self._last_narration_end = 0.0
    
    def export_subtitles(self, output_path: str, format: str = "srt"):
        """
        Export narration as subtitle file
        
        Args:
            output_path: Output path
            format: Subtitle format (srt, vtt)
        """
        if format == "srt":
            self._export_srt(output_path)
        elif format == "vtt":
            self._export_vtt(output_path)
        else:
            raise ValueError(f"Unsupported subtitle format: {format}")
    
    def _export_srt(self, output_path: str):
        """Export as SRT format"""
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
        
        logger.info(f"Exported SRT subtitle: {output_path}")
    
    def _export_vtt(self, output_path: str):
        """Export as WebVTT format"""
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
        
        logger.info(f"Exported VTT subtitle: {output_path}")
    
    @staticmethod
    def _format_time_srt(seconds: float) -> str:
        """Format as SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    @staticmethod
    def _format_time_vtt(seconds: float) -> str:
        """Format as VTT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
