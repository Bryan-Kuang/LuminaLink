"""
Narrator Module

Generates, filters, and tracks real-time narration from scene analysis results.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from .config import get_config
from .scene_analyzer import SceneAnalysis

logger = logging.getLogger(__name__)

_REFUSAL_PATTERNS = re.compile(
    r"i (can'?t|cannot|am unable to|won'?t|will not)"
    r"|i'?m (sorry|unable|not able)"
    r"|(unable|not able) to (assist|help|describe|analyze)"
    r"|(copyrighted|intellectual property|content policy)"
    r"|as an ai"
    r"|i apologize",
    re.IGNORECASE,
)


class NarrationStyle(Enum):
    CONCISE = "concise"
    DETAILED = "detailed"
    CINEMATIC = "cinematic"
    NEUTRAL = "neutral"


@dataclass
class Narration:
    """A single narration entry."""
    text: str
    timestamp: float
    duration: float = 0.0
    style: NarrationStyle = NarrationStyle.CINEMATIC


class Narrator:
    """
    Converts SceneAnalysis results into Narration objects.

    Responsibilities:
    - Cooldown enforcement (min seconds between narrations)
    - Refusal response detection (skip AI refusals)
    - Near-duplicate filtering (word-overlap Jaccard check)
    - History tracking and SRT export
    """

    def __init__(
        self,
        style: NarrationStyle = NarrationStyle.CINEMATIC,
        cooldown: Optional[float] = None,
    ):
        config = get_config()
        self._style = style
        self._cooldown = cooldown if cooldown is not None else config.narration.interval
        self._max_length = config.narration.max_length
        self._history: List[Narration] = []
        self._last_narration_time: float = -999.0

    def should_narrate(self, timestamp: float) -> bool:
        """Return True if enough time has elapsed since the last narration."""
        return (timestamp - self._last_narration_time) >= self._cooldown

    def generate_narration(
        self,
        scene_analysis: SceneAnalysis,
        slot: Tuple[float, float],
        characters_in_frame: Optional[List[str]] = None,
    ) -> Optional[Narration]:
        """
        Convert a SceneAnalysis into a Narration, applying all filters.
        Returns None if the narration should be skipped.
        """
        text = scene_analysis.description.strip()
        timestamp = slot[0]

        if not text:
            return None

        if self._is_refusal_response(text):
            logger.warning(f"Refusal response detected, skipping: {text[:60]}")
            return None

        if self._is_duplicate(text):
            logger.debug("Duplicate narration skipped")
            return None

        if len(text) > self._max_length:
            text = self._truncate(text, self._max_length)

        narration = Narration(
            text=text,
            timestamp=timestamp,
            duration=slot[1] - slot[0],
            style=self._style,
        )

        self._history.append(narration)
        self._last_narration_time = timestamp
        return narration

    def get_history(self) -> List[Narration]:
        """Return a copy of the narration history."""
        return list(self._history)

    def export_subtitles(self, path: str) -> bool:
        """Export narration history as an SRT subtitle file."""
        try:
            with open(Path(path), "w", encoding="utf-8") as f:
                for i, narration in enumerate(self._history, start=1):
                    start = self._seconds_to_srt(narration.timestamp)
                    end = self._seconds_to_srt(
                        narration.timestamp + max(narration.duration, 3.0)
                    )
                    f.write(f"{i}\n{start} --> {end}\n{narration.text}\n\n")
            logger.info(f"Subtitles exported to {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export subtitles: {e}")
            return False

    def _is_refusal_response(self, text: str) -> bool:
        return bool(_REFUSAL_PATTERNS.search(text))

    def _is_duplicate(self, text: str, threshold: float = 0.6) -> bool:
        """Jaccard word-overlap check against the last 3 narrations."""
        if not self._history:
            return False
        words = set(text.lower().split())
        if not words:
            return False
        for recent in self._history[-3:]:
            recent_words = set(recent.text.lower().split())
            if not recent_words:
                continue
            union = words | recent_words
            if union and len(words & recent_words) / len(union) >= threshold:
                return True
        return False

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate at the last sentence boundary before max_len."""
        if len(text) <= max_len:
            return text
        truncated = text[:max_len]
        last_stop = max(
            truncated.rfind("."),
            truncated.rfind("!"),
            truncated.rfind("?"),
        )
        if last_stop > max_len // 2:
            return truncated[: last_stop + 1]
        return truncated.rstrip() + "…"

    @staticmethod
    def _seconds_to_srt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
