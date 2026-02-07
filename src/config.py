"""
Configuration Module

Manages all system configuration parameters, supports loading from environment variables and config files
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class AIConfig:
    """AI Model Configuration"""
    # Default provider
    provider: Literal["openai", "google", "dashscope", "local"] = "openai"
    
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4-vision-preview"
    openai_base_url: Optional[str] = None
    
    # Google Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-pro-vision"
    
    # Alibaba DashScope
    dashscope_api_key: str = ""
    qwen_model: str = "qwen-vl-max"
    
    # Local model
    local_model_path: str = ""
    
    def __post_init__(self):
        self.provider = os.getenv("AI_PROVIDER", self.provider)
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.openai_model = os.getenv("OPENAI_MODEL", self.openai_model)
        self.openai_base_url = os.getenv("OPENAI_BASE_URL")
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.gemini_model = os.getenv("GEMINI_MODEL", self.gemini_model)
        self.dashscope_api_key = os.getenv("DASHSCOPE_API_KEY", "")
        self.qwen_model = os.getenv("QWEN_MODEL", self.qwen_model)


@dataclass
class TTSConfig:
    """Text-to-Speech Configuration"""
    engine: Literal["edge", "gtts", "pyttsx3"] = "edge"
    voice: str = "zh-CN-XiaoxiaoNeural"
    speed: float = 1.0
    
    def __post_init__(self):
        self.engine = os.getenv("TTS_ENGINE", self.engine)
        self.voice = os.getenv("TTS_VOICE", self.voice)
        self.speed = float(os.getenv("TTS_SPEED", self.speed))


@dataclass
class NarrationConfig:
    """Narration Parameters Configuration"""
    # Minimum narration interval (seconds)
    interval: float = 5.0
    # Silence detection threshold (dB)
    silence_threshold: float = -40.0
    # Maximum characters per narration
    max_length: int = 100
    # Narration style
    style: Literal["concise", "detailed", "cinematic"] = "concise"
    
    def __post_init__(self):
        self.interval = float(os.getenv("NARRATION_INTERVAL", self.interval))
        self.silence_threshold = float(os.getenv("SILENCE_THRESHOLD", self.silence_threshold))
        self.max_length = int(os.getenv("MAX_NARRATION_LENGTH", self.max_length))


@dataclass
class FaceRecognitionConfig:
    """Face Recognition Configuration"""
    # Recognition confidence threshold
    threshold: float = 0.6
    # Detection model
    detection_model: Literal["hog", "cnn"] = "hog"
    # Enable deep learning enhancement
    use_deep_learning: bool = False
    
    def __post_init__(self):
        self.threshold = float(os.getenv("FACE_RECOGNITION_THRESHOLD", self.threshold))
        self.detection_model = os.getenv("FACE_DETECTION_MODEL", self.detection_model)


@dataclass
class VideoConfig:
    """Video Processing Configuration"""
    # Keyframe extraction interval (seconds)
    keyframe_interval: float = 1.0
    # Scene change detection threshold
    scene_change_threshold: float = 0.3
    # Preview window size
    preview_width: int = 800
    preview_height: int = 450
    # Analysis frame rate
    analysis_fps: int = 1
    
    def __post_init__(self):
        self.keyframe_interval = float(os.getenv("KEYFRAME_INTERVAL", self.keyframe_interval))
        self.scene_change_threshold = float(os.getenv("SCENE_CHANGE_THRESHOLD", self.scene_change_threshold))
        self.preview_width = int(os.getenv("PREVIEW_WIDTH", self.preview_width))
        self.preview_height = int(os.getenv("PREVIEW_HEIGHT", self.preview_height))


@dataclass
class PathConfig:
    """Path Configuration"""
    cache_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "cache")
    characters_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "characters")
    models_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "models")
    
    def __post_init__(self):
        cache = os.getenv("CACHE_DIR")
        if cache:
            self.cache_dir = Path(cache)
        
        chars = os.getenv("CHARACTERS_DIR")
        if chars:
            self.characters_dir = Path(chars)
        
        # Ensure directories exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Main Configuration Class"""
    ai: AIConfig = field(default_factory=AIConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    narration: NarrationConfig = field(default_factory=NarrationConfig)
    face_recognition: FaceRecognitionConfig = field(default_factory=FaceRecognitionConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Log level
    log_level: str = "INFO"
    
    def __post_init__(self):
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get configuration instance"""
    return config


def reload_config():
    """Reload configuration"""
    global config
    load_dotenv(override=True)
    config = Config()
    return config
