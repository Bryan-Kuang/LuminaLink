"""
Narrator Module

Generates, filters, and tracks real-time narration from scene analysis results.
"""

import logging
import re
import time
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
        self._cooldown = max(1.0, cooldown if cooldown is not None else config.narration.interval)
        self._max_length = config.narration.max_length
        self._history: List[Narration] = []
        # Use wall clock (time.time()) so mark_played() can reset it correctly.
        # Start far in the past so the first narration fires immediately.
        self._last_narration_time: float = time.time() - 9999.0

    def should_narrate(self) -> bool:
        """Return True if enough time has elapsed since the last narration ended."""
        elapsed = time.time() - self._last_narration_time
        return elapsed >= self._cooldown

    def mark_played(self) -> None:
        """
        Reset the cooldown clock from RIGHT NOW.

        Call this AFTER audio playback finishes so the cooldown gap is measured
        from the END of the last spoken narration — not from when it was queued.
        This prevents back-to-back narrations when TTS + playback takes longer
        than the cooldown interval.
        """
        self._last_narration_time = time.time()
        logger.debug(f"Cooldown reset after playback (next in ≥{self._cooldown:.0f}s)")

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

        if len(slot) < 2 or slot[1] < slot[0]:
            logger.warning("Invalid slot passed to generate_narration, skipping")
            return None

        if not text:
            return None

        # Skip sentinel/error strings from the AI fallback
        if text.startswith("[") and text.endswith("]"):
            logger.debug("Skipping fallback/error description from scene analyzer")
            return None

        if self._is_refusal_response(text):
            logger.warning(f"Refusal response detected, skipping: {text[:60]}")
            return None

        if len(text) > self._max_length:
            text = self._truncate(text, self._max_length)

        if self._is_duplicate(text):
            logger.debug("Duplicate narration skipped")
            return None

        narration = Narration(
            text=text,
            timestamp=timestamp,
            duration=slot[1] - slot[0],
            style=self._style,
        )

        self._history.append(narration)
        # Set to now so the cooldown gate blocks analysis during TTS+playback.
        # mark_played() will reset this again once audio finishes.
        self._last_narration_time = time.time()
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

    # Common English stopwords — excluded from Jaccard so "he walks to the
    # door" and "the man moves toward the door" don't appear 100% different.
    _STOPWORDS = frozenset({
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to",
        "for", "of", "with", "by", "from", "is", "are", "was", "were",
        "he", "she", "they", "it", "his", "her", "their", "its", "him",
        "as", "into", "onto", "up", "down", "out", "over", "through",
        "while", "toward", "towards", "across", "around", "this", "that",
    })

    def _is_duplicate(self, text: str) -> bool:
        """
        Smart duplicate check:
        - Strips stopwords before Jaccard so paraphrases are caught
        - Checks the last 5 narrations (wider window than before)
        - Uses time-decay: stricter threshold within the last 60 s
        """
        if not self._history:
            return False

        words = {w for w in text.lower().split() if w not in self._STOPWORDS}
        if not words:
            return False

        now = time.time()
        for recent in self._history[-5:]:
            recent_words = {
                w for w in recent.text.lower().split()
                if w not in self._STOPWORDS
            }
            if not recent_words:
                continue

            union = words | recent_words
            overlap = len(words & recent_words) / len(union)

            # How many seconds ago was this narration?
            age = now - recent.timestamp          # recent.timestamp is wall-clock
            # Tighter threshold for recent narrations, looser for older ones:
            #   age   0 s → threshold 0.35
            #   age  60 s → threshold 0.55
            #   age 120 s → threshold 0.70 (effectively off)
            threshold = min(0.35 + age / 200.0, 0.70)

            if overlap >= threshold:
                logger.debug(
                    f"Duplicate skipped (overlap={overlap:.2f} >= "
                    f"threshold={threshold:.2f}, age={age:.0f}s)"
                )
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
