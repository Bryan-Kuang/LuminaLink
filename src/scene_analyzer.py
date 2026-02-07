"""
场景分析模块

使用多模态 AI 模型分析视频帧，理解场景内容、动作和情感
"""

import base64
import cv2
import numpy as np
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import asyncio
import aiohttp

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class SceneAnalysis:
    """场景分析结果"""
    description: str               # 场景描述
    actions: List[str]             # 正在发生的动作
    emotions: Dict[str, str]       # 角色情感 {角色名: 情感}
    setting: str                   # 场景环境
    objects: List[str]             # 重要物品
    timestamp: float               # 时间戳
    confidence: float = 0.0        # 置信度
    raw_response: Optional[str] = None


class SceneAnalyzer:
    """场景分析器"""
    
    def __init__(self, provider: Optional[str] = None):
        """
        初始化场景分析器
        
        Args:
            provider: AI 提供商 (openai, google, dashscope, local)
        """
        self.config = get_config().ai
        self.provider = provider or self.config.provider
        
        # 上下文记忆
        self._context_history: List[str] = []
        self._max_context_length = 5
        
        # 角色信息缓存
        self._known_characters: List[str] = []
    
    def set_characters(self, characters: List[str]):
        """设置已知角色列表"""
        self._known_characters = characters
        logger.info(f"已设置 {len(characters)} 个已知角色")
    
    def _build_prompt(
        self,
        characters_in_frame: List[str],
        previous_context: Optional[str] = None
    ) -> str:
        """
        构建分析提示词
        
        Args:
            characters_in_frame: 画面中的角色
            previous_context: 之前的上下文
        """
        character_info = ""
        if characters_in_frame:
            character_info = f"\n画面中识别到的角色: {', '.join(characters_in_frame)}"
        elif self._known_characters:
            character_info = f"\n这部电影中的主要角色有: {', '.join(self._known_characters)}"
        
        context_info = ""
        if previous_context:
            context_info = f"\n之前的剧情: {previous_context}"
        elif self._context_history:
            context_info = f"\n之前的剧情: {self._context_history[-1]}"
        
        prompt = f"""你是一位专业的电影无障碍解说员，正在为视障人士描述电影画面。
{character_info}
{context_info}

请描述画面中：
- 人物的动作（站、坐、走、说话等）
- 人物的表情和情绪
- 场景环境和重要物品
- 人物之间的互动

要求：
- 直接输出描述，不要任何前缀
- 使用客观的陈述句
- 20-40个中文字
- 例如："男子站在讲台前，神情严肃，手指敲击桌面"

请描述这个画面："""

        return prompt
    
    async def analyze_frame_async(
        self,
        frame: np.ndarray,
        characters_in_frame: Optional[List[str]] = None,
        timestamp: float = 0.0
    ) -> SceneAnalysis:
        """
        异步分析单帧
        
        Args:
            frame: BGR 格式的图像
            characters_in_frame: 画面中的角色名称
            timestamp: 时间戳
        
        Returns:
            SceneAnalysis 对象
        """
        characters = characters_in_frame or []
        
        if self.provider == "openai":
            result = await self._analyze_with_openai(frame, characters)
        elif self.provider == "google":
            result = await self._analyze_with_gemini(frame, characters)
        elif self.provider == "dashscope":
            result = await self._analyze_with_qwen(frame, characters)
        else:
            result = await self._analyze_with_local(frame, characters)
        
        # 更新上下文历史
        if result.description:
            self._context_history.append(result.description)
            if len(self._context_history) > self._max_context_length:
                self._context_history.pop(0)
        
        result.timestamp = timestamp
        return result
    
    def analyze_frame(
        self,
        frame: np.ndarray,
        characters_in_frame: Optional[List[str]] = None,
        timestamp: float = 0.0
    ) -> SceneAnalysis:
        """
        同步分析单帧
        """
        return asyncio.run(self.analyze_frame_async(frame, characters_in_frame, timestamp))
    
    async def _analyze_with_openai(
        self,
        frame: np.ndarray,
        characters: List[str]
    ) -> SceneAnalysis:
        """使用 OpenAI GPT-4V 分析"""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url
            )
            
            # 转换图像为 base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            prompt = self._build_prompt(characters)
            
            response = await client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}",
                                    "detail": "auto"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200
            )
            
            description = response.choices[0].message.content.strip()
            
            return SceneAnalysis(
                description=description,
                actions=[],
                emotions={},
                setting="",
                objects=[],
                timestamp=0.0,
                confidence=0.9,
                raw_response=description
            )
            
        except Exception as e:
            logger.error(f"OpenAI 分析失败: {e}")
            return self._fallback_analysis()
    
    async def _analyze_with_gemini(
        self,
        frame: np.ndarray,
        characters: List[str]
    ) -> SceneAnalysis:
        """使用 Google Gemini 分析"""
        try:
            import google.generativeai as genai
            from PIL import Image
            
            genai.configure(api_key=self.config.google_api_key)
            model = genai.GenerativeModel(self.config.gemini_model)
            
            # 转换为 PIL Image
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            
            prompt = self._build_prompt(characters)
            
            response = await asyncio.to_thread(
                model.generate_content,
                [prompt, pil_image]
            )
            
            description = response.text.strip()
            
            return SceneAnalysis(
                description=description,
                actions=[],
                emotions={},
                setting="",
                objects=[],
                timestamp=0.0,
                confidence=0.9,
                raw_response=description
            )
            
        except Exception as e:
            logger.error(f"Gemini 分析失败: {e}")
            return self._fallback_analysis()
    
    async def _analyze_with_qwen(
        self,
        frame: np.ndarray,
        characters: List[str]
    ) -> SceneAnalysis:
        """使用阿里通义千问 VL 分析"""
        try:
            import dashscope
            from dashscope import MultiModalConversation
            
            dashscope.api_key = self.config.dashscope_api_key
            
            # 转换图像为 base64
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            image_base64 = base64.b64encode(buffer).decode('utf-8')
            
            prompt = self._build_prompt(characters)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": f"data:image/jpeg;base64,{image_base64}"},
                        {"text": prompt}
                    ]
                }
            ]
            
            response = await asyncio.to_thread(
                MultiModalConversation.call,
                model=self.config.qwen_model,
                messages=messages
            )
            
            if response.status_code == 200:
                description = response.output.choices[0].message.content[0]["text"]
            else:
                raise Exception(f"API 错误: {response.code} - {response.message}")
            
            return SceneAnalysis(
                description=description,
                actions=[],
                emotions={},
                setting="",
                objects=[],
                timestamp=0.0,
                confidence=0.9,
                raw_response=description
            )
            
        except Exception as e:
            logger.error(f"通义千问分析失败: {e}")
            return self._fallback_analysis()
    
    async def _analyze_with_local(
        self,
        frame: np.ndarray,
        characters: List[str]
    ) -> SceneAnalysis:
        """使用本地模型分析"""
        # TODO: 实现本地模型支持（如 LLaVA）
        logger.warning("本地模型暂未实现，使用备用分析")
        return self._fallback_analysis()
    
    def _fallback_analysis(self) -> SceneAnalysis:
        """备用分析（当 AI 服务不可用时）"""
        return SceneAnalysis(
            description="[场景分析暂时不可用]",
            actions=[],
            emotions={},
            setting="",
            objects=[],
            timestamp=0.0,
            confidence=0.0
        )
    
    async def analyze_sequence(
        self,
        frames: List[np.ndarray],
        characters_per_frame: Optional[List[List[str]]] = None,
        timestamps: Optional[List[float]] = None
    ) -> List[SceneAnalysis]:
        """
        分析帧序列
        
        Args:
            frames: 帧列表
            characters_per_frame: 每帧的角色列表
            timestamps: 时间戳列表
        
        Returns:
            SceneAnalysis 列表
        """
        results = []
        
        for i, frame in enumerate(frames):
            characters = characters_per_frame[i] if characters_per_frame else None
            timestamp = timestamps[i] if timestamps else i * 1.0
            
            result = await self.analyze_frame_async(frame, characters, timestamp)
            results.append(result)
        
        return results
    
    def get_context_summary(self) -> str:
        """获取上下文摘要"""
        if not self._context_history:
            return ""
        return " -> ".join(self._context_history[-3:])
    
    def clear_context(self):
        """清除上下文历史"""
        self._context_history.clear()


class SceneChangeDetector:
    """场景变化检测器"""
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self._previous_hist: Optional[np.ndarray] = None
    
    def detect_change(self, frame: np.ndarray) -> bool:
        """检测是否发生场景变化"""
        # 计算直方图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        
        if self._previous_hist is None:
            self._previous_hist = hist
            return True
        
        # 比较直方图
        similarity = cv2.compareHist(self._previous_hist, hist, cv2.HISTCMP_CORREL)
        self._previous_hist = hist
        
        return similarity < (1 - self.threshold)
    
    def reset(self):
        """重置检测器"""
        self._previous_hist = None
