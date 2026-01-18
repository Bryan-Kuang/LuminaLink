from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field


@dataclass(frozen=True)
class VideoFrame:
    """A decoded video frame with an absolute timestamp in milliseconds."""

    pts_ms: int
    frame_index: int
    image_bgr: np.ndarray


@dataclass(frozen=True)
class AudioFrame:
    """A fixed-duration audio frame aligned to the same timebase as video."""

    start_ms: int
    end_ms: int
    pcm_s16le: bytes
    sample_rate_hz: int


@dataclass(frozen=True)
class DetectedObject:
    """An object detection result on a single frame."""

    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


@dataclass(frozen=True)
class CharacterMatch:
    """A character identification result for a detected face."""

    character_id: str
    display_name: str
    confidence: float
    bbox_xyxy: Optional[tuple[int, int, int, int]] = None


@dataclass(frozen=True)
class VisualFacts:
    """Structured facts extracted from a frame for narration generation."""

    pts_ms: int
    scene_change: bool
    objects: list[DetectedObject]
    characters: list[CharacterMatch]


@dataclass(frozen=True)
class DialogueState:
    """Dialogue state used to gate narration output."""

    is_dialogue: bool
    confidence: float
    active_segment_start_ms: Optional[int] = None


class NarrationEvent(BaseModel):
    """A standardized narration event to be consumed by downstream sinks (e.g., TTS)."""

    type: str = Field(default="narration")
    run_id: str
    timestamp_start_ms: int
    timestamp_end_ms: int
    text: str
    style: str
    detail_level: int
    confidence: float
    metadata: dict[str, Any] = Field(default_factory=dict)

