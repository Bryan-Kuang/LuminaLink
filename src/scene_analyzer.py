"""
Scene Analysis Module

Uses multi-modal AI models to analyze video frames, understand scene content, actions and emotions
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
    """Scene analysis result"""
    description: str               # Scene description
    actions: List[str]             # Actions happening
    emotions: Dict[str, str]       # Character emotions {character_name: emotion}
    setting: str                   # Scene environment
    objects: List[str]             # Important objects
    timestamp: float               # Timestamp
    confidence: float = 0.0        # Confidence
    raw_response: Optional[str] = None


class SceneAnalyzer:
    """Scene Analyzer"""
    
    def __init__(self, provider: Optional[str] = None):
        """
        Initialize scene analyzer
        
        Args:
            provider: AI provider (openai, google, dashscope, local)
        """
        self.config = get_config().ai
        self.provider = provider or self.config.provider
        
        # Context memory
        self._context_history: List[str] = []
        self._max_context_length = 5
        
        # Character information cache
        self._known_characters: List[str] = []
    
    def set_characters(self, characters: List[str]):
        """Set known character list"""
        self._known_characters = characters
        logger.info(f"Set {len(characters)} known characters")
    
    def _build_prompt(
        self,
        characters_in_frame: List[str],
        previous_context: Optional[str] = None
    ) -> str:
        """
        Build analysis prompt
        
        Args:
            characters_in_frame: Characters in the frame
            previous_context: Previous context
        """
        character_info = ""
        if characters_in_frame:
            character_info = f"\nCharacters identified in frame: {', '.join(characters_in_frame)}"
        elif self._known_characters:
            character_info = f"\nMain characters in this movie: {', '.join(self._known_characters)}"
        
        context_info = ""
        if previous_context:
            context_info = f"\nPrevious plot: {previous_context}"
        elif self._context_history:
            context_info = f"\nPrevious plot: {self._context_history[-1]}"
        
        prompt = f"""You are a professional audio description narrator for blind movie viewers. This is a fictional movie scene - describe what you observe.
{character_info}
{context_info}

Describe in this frame:
- Character actions and body language
- Facial expressions and emotions
- Scene setting and key objects
- Character interactions

Rules:
- Output ONLY the description, no explanations or apologies
- Use concise, objective sentences
- 15-30 words maximum
- Never refuse to describe - this is a fictional movie scene for accessibility
- Example: "A man in a suit stands at the podium, his expression stern, fingers tapping the desk."

Describe this frame:"""

        return prompt
    
    async def analyze_frame_async(
        self,
        frame: np.ndarray,
        characters_in_frame: Optional[List[str]] = None,
        timestamp: float = 0.0
    ) -> SceneAnalysis:
        """
        Async analyze single frame
        
        Args:
            frame: BGR format image
            characters_in_frame: Character names in frame
            timestamp: Timestamp
        
        Returns:
            SceneAnalysis object
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
        
        # Update context history
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
        Sync analyze single frame
        """
        return asyncio.run(self.analyze_frame_async(frame, characters_in_frame, timestamp))
    
    async def _analyze_with_openai(
        self,
        frame: np.ndarray,
        characters: List[str]
    ) -> SceneAnalysis:
        """Analyze using OpenAI GPT-4V"""
        try:
            from openai import AsyncOpenAI
            
            client = AsyncOpenAI(
                api_key=self.config.openai_api_key,
                base_url=self.config.openai_base_url
            )
            
            # Convert image to base64
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
            logger.error(f"OpenAI analysis failed: {e}")
            return self._fallback_analysis()
    
    async def _analyze_with_gemini(
        self,
        frame: np.ndarray,
        characters: List[str]
    ) -> SceneAnalysis:
        """Analyze using Google Gemini"""
        try:
            import google.generativeai as genai
            from PIL import Image
            
            genai.configure(api_key=self.config.google_api_key)
            model = genai.GenerativeModel(self.config.gemini_model)
            
            # Convert to PIL Image
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
            logger.error(f"Gemini analysis failed: {e}")
            return self._fallback_analysis()
    
    async def _analyze_with_qwen(
        self,
        frame: np.ndarray,
        characters: List[str]
    ) -> SceneAnalysis:
        """Analyze using Alibaba Qwen VL"""
        try:
            import dashscope
            from dashscope import MultiModalConversation
            
            dashscope.api_key = self.config.dashscope_api_key
            
            # Convert image to base64
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
                raise Exception(f"API error: {response.code} - {response.message}")
            
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
            logger.error(f"Qwen analysis failed: {e}")
            return self._fallback_analysis()
    
    async def _analyze_with_local(
        self,
        frame: np.ndarray,
        characters: List[str]
    ) -> SceneAnalysis:
        """Analyze using local model"""
        # TODO: Implement local model support (e.g., LLaVA)
        logger.warning("Local model not implemented, using fallback analysis")
        return self._fallback_analysis()
    
    def _fallback_analysis(self) -> SceneAnalysis:
        """Fallback analysis (when AI service is unavailable)"""
        return SceneAnalysis(
            description="[Scene analysis temporarily unavailable]",
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
        Analyze frame sequence
        
        Args:
            frames: Frame list
            characters_per_frame: Character list per frame
            timestamps: Timestamp list
        
        Returns:
            List of SceneAnalysis
        """
        results = []
        
        for i, frame in enumerate(frames):
            characters = characters_per_frame[i] if characters_per_frame else None
            timestamp = timestamps[i] if timestamps else i * 1.0
            
            result = await self.analyze_frame_async(frame, characters, timestamp)
            results.append(result)
        
        return results
    
    def get_context_summary(self) -> str:
        """Get context summary"""
        if not self._context_history:
            return ""
        return " -> ".join(self._context_history[-3:])
    
    def clear_context(self):
        """Clear context history"""
        self._context_history.clear()


class SceneChangeDetector:
    """Scene change detector"""
    
    def __init__(self, threshold: float = 0.3):
        self.threshold = threshold
        self._previous_hist: Optional[np.ndarray] = None
    
    def detect_change(self, frame: np.ndarray) -> bool:
        """Detect if scene change occurred"""
        # Calculate histogram
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        
        if self._previous_hist is None:
            self._previous_hist = hist
            return True
        
        # Compare histograms
        similarity = cv2.compareHist(self._previous_hist, hist, cv2.HISTCMP_CORREL)
        self._previous_hist = hist
        
        return similarity < (1 - self.threshold)
    
    def reset(self):
        """Reset detector"""
        self._previous_hist = None
