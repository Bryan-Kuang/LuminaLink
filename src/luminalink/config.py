from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class PerformanceConfig(BaseModel):
    """Runtime performance knobs for the streaming pipeline."""

    target_fps: float = Field(default=12.0, ge=1.0)
    max_end_to_end_latency_ms: int = Field(default=300, ge=0)
    narration_min_interval_ms: int = Field(default=1200, ge=0)
    narration_duration_ms: int = Field(default=900, ge=200)


class SceneChangeConfig(BaseModel):
    """Configuration for scene change detection."""

    hist_correlation_threshold: float = Field(default=0.55, ge=-1.0, le=1.0)
    warmup_frames: int = Field(default=5, ge=0)


class VisionConfig(BaseModel):
    """Configuration for vision inference."""

    scene_change: SceneChangeConfig = Field(default_factory=SceneChangeConfig)
    object_detector: str = Field(default="noop")
    face_pipeline: str = Field(default="auto")


class DialogueConfig(BaseModel):
    """Configuration for dialogue detection."""

    enabled: bool = True
    sample_rate_hz: int = Field(default=16000)
    vad_mode: int = Field(default=2, ge=0, le=3)
    frame_ms: int = Field(default=30)
    start_trigger_frames: int = Field(default=3, ge=1)
    end_trigger_frames: int = Field(default=8, ge=1)


class NLGConfig(BaseModel):
    """Configuration for narration text generation."""

    backend: str = Field(default="auto")
    style: str = Field(default="concise")
    detail_level: int = Field(default=2, ge=1, le=5)


class CharacterDBConfig(BaseModel):
    """Configuration for character database storage."""

    path: str = Field(default="data/characters.sqlite")
    match_threshold: float = Field(default=0.75, ge=0.0, le=1.0)


class OutputConfig(BaseModel):
    """Configuration for output sinks."""

    format: str = Field(default="jsonl")


class AppConfig(BaseModel):
    """Root configuration for LuminaLink phase-1 pipeline."""

    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    dialogue: DialogueConfig = Field(default_factory=DialogueConfig)
    nlg: NLGConfig = Field(default_factory=NLGConfig)
    character_db: CharacterDBConfig = Field(default_factory=CharacterDBConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @staticmethod
    def load(path: Optional[Path]) -> "AppConfig":
        """Load configuration from YAML, or return defaults when absent."""

        if path is None:
            return AppConfig()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return AppConfig.model_validate(raw)

