"""
配置管理模块

管理系统的所有配置参数，支持从环境变量和配置文件加载
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent


@dataclass
class AIConfig:
    """AI 模型配置"""
    # 默认提供商
    provider: Literal["openai", "google", "dashscope", "local"] = "openai"
    
    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4-vision-preview"
    openai_base_url: Optional[str] = None
    
    # Google Gemini
    google_api_key: str = ""
    gemini_model: str = "gemini-pro-vision"
    
    # 阿里云 DashScope
    dashscope_api_key: str = ""
    qwen_model: str = "qwen-vl-max"
    
    # 本地模型
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
    """语音合成配置"""
    engine: Literal["edge", "gtts", "pyttsx3"] = "edge"
    voice: str = "zh-CN-XiaoxiaoNeural"
    speed: float = 1.0
    
    def __post_init__(self):
        self.engine = os.getenv("TTS_ENGINE", self.engine)
        self.voice = os.getenv("TTS_VOICE", self.voice)
        self.speed = float(os.getenv("TTS_SPEED", self.speed))


@dataclass
class NarrationConfig:
    """讲解参数配置"""
    # 最小讲解间隔（秒）
    interval: float = 5.0
    # 静音检测阈值（dB）
    silence_threshold: float = -40.0
    # 每次讲解最大字数
    max_length: int = 100
    # 讲解风格
    style: Literal["简洁", "详细", "电影解说"] = "简洁"
    
    def __post_init__(self):
        self.interval = float(os.getenv("NARRATION_INTERVAL", self.interval))
        self.silence_threshold = float(os.getenv("SILENCE_THRESHOLD", self.silence_threshold))
        self.max_length = int(os.getenv("MAX_NARRATION_LENGTH", self.max_length))


@dataclass
class FaceRecognitionConfig:
    """人脸识别配置"""
    # 识别置信度阈值
    threshold: float = 0.6
    # 检测模型
    detection_model: Literal["hog", "cnn"] = "hog"
    # 是否启用深度学习增强
    use_deep_learning: bool = False
    
    def __post_init__(self):
        self.threshold = float(os.getenv("FACE_RECOGNITION_THRESHOLD", self.threshold))
        self.detection_model = os.getenv("FACE_DETECTION_MODEL", self.detection_model)


@dataclass
class VideoConfig:
    """视频处理配置"""
    # 关键帧提取间隔（秒）
    keyframe_interval: float = 1.0
    # 场景切换检测阈值
    scene_change_threshold: float = 0.3
    # 预览窗口大小
    preview_width: int = 800
    preview_height: int = 450
    # 分析帧率
    analysis_fps: int = 1
    
    def __post_init__(self):
        self.keyframe_interval = float(os.getenv("KEYFRAME_INTERVAL", self.keyframe_interval))
        self.scene_change_threshold = float(os.getenv("SCENE_CHANGE_THRESHOLD", self.scene_change_threshold))
        self.preview_width = int(os.getenv("PREVIEW_WIDTH", self.preview_width))
        self.preview_height = int(os.getenv("PREVIEW_HEIGHT", self.preview_height))


@dataclass
class PathConfig:
    """路径配置"""
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
        
        # 确保目录存在
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """主配置类"""
    ai: AIConfig = field(default_factory=AIConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    narration: NarrationConfig = field(default_factory=NarrationConfig)
    face_recognition: FaceRecognitionConfig = field(default_factory=FaceRecognitionConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # 日志级别
    log_level: str = "INFO"
    
    def __post_init__(self):
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)


# 全局配置实例
config = Config()


def get_config() -> Config:
    """获取配置实例"""
    return config


def reload_config():
    """重新加载配置"""
    global config
    load_dotenv(override=True)
    config = Config()
    return config
