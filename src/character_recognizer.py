"""
角色识别模块

负责识别视频中的人物角色，维护角色数据库，将人脸与角色名称关联
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import logging
import pickle
from datetime import datetime

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class Character:
    """角色信息"""
    id: str                          # 唯一标识符
    name: str                        # 角色名称
    aliases: List[str] = field(default_factory=list)  # 别名列表
    description: str = ""            # 角色描述
    face_encodings: List[np.ndarray] = field(default_factory=list)  # 人脸编码列表
    appearance_count: int = 0        # 出现次数
    last_seen: Optional[float] = None  # 最后出现时间
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """转换为字典（不含人脸编码）"""
        return {
            "id": self.id,
            "name": self.name,
            "aliases": self.aliases,
            "description": self.description,
            "appearance_count": self.appearance_count,
            "metadata": self.metadata
        }


@dataclass
class FaceDetection:
    """人脸检测结果"""
    location: Tuple[int, int, int, int]  # (top, right, bottom, left)
    encoding: Optional[np.ndarray] = None
    character_id: Optional[str] = None
    character_name: Optional[str] = None
    confidence: float = 0.0
    
    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """获取边界框 (x, y, width, height)"""
        top, right, bottom, left = self.location
        return (left, top, right - left, bottom - top)


class CharacterRecognizer:
    """角色识别器"""
    
    def __init__(self, characters_dir: Optional[Path] = None):
        """
        初始化角色识别器
        
        Args:
            characters_dir: 角色数据目录
        """
        self.config = get_config().face_recognition
        self.characters_dir = characters_dir or get_config().paths.characters_dir
        
        # 角色数据库
        self.characters: Dict[str, Character] = {}
        
        # 未知人脸缓存（用于自动学习）
        self._unknown_faces: List[Tuple[np.ndarray, float]] = []
        
        # 最近识别结果缓存
        self._recognition_cache: Dict[str, Tuple[str, float]] = {}
        
        # 尝试加载已有角色数据
        self._load_characters()
    
    def _load_characters(self):
        """加载角色数据"""
        db_file = self.characters_dir / "characters.pkl"
        
        if db_file.exists():
            try:
                with open(db_file, "rb") as f:
                    self.characters = pickle.load(f)
                logger.info(f"已加载 {len(self.characters)} 个角色")
            except Exception as e:
                logger.error(f"加载角色数据失败: {e}")
        
        # 同时加载 JSON 配置（用于手动编辑）
        json_file = self.characters_dir / "characters.json"
        if json_file.exists():
            self._load_characters_json(json_file)
    
    def _load_characters_json(self, json_path: Path):
        """从 JSON 文件加载角色配置"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for char_data in data.get("characters", []):
                char_id = char_data.get("id", char_data["name"])
                
                # 如果角色不存在，创建新角色
                if char_id not in self.characters:
                    self.characters[char_id] = Character(
                        id=char_id,
                        name=char_data["name"],
                        aliases=char_data.get("aliases", []),
                        description=char_data.get("description", "")
                    )
                else:
                    # 更新已有角色信息
                    self.characters[char_id].name = char_data["name"]
                    self.characters[char_id].aliases = char_data.get("aliases", [])
                    self.characters[char_id].description = char_data.get("description", "")
                
                # 加载角色图片
                for img_path in char_data.get("face_images", []):
                    full_path = self.characters_dir / img_path
                    if full_path.exists():
                        self._add_face_from_image(char_id, str(full_path))
                        
        except Exception as e:
            logger.error(f"加载角色 JSON 失败: {e}")
    
    def _add_face_from_image(self, character_id: str, image_path: str):
        """从图片添加人脸编码"""
        try:
            import face_recognition
            
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings and character_id in self.characters:
                self.characters[character_id].face_encodings.append(encodings[0])
                logger.debug(f"已添加人脸编码: {character_id}")
                
        except Exception as e:
            logger.error(f"添加人脸失败 {image_path}: {e}")
    
    def save_characters(self):
        """保存角色数据"""
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存二进制数据（包含人脸编码）
        db_file = self.characters_dir / "characters.pkl"
        try:
            with open(db_file, "wb") as f:
                pickle.dump(self.characters, f)
            logger.info(f"角色数据已保存: {len(self.characters)} 个角色")
        except Exception as e:
            logger.error(f"保存角色数据失败: {e}")
        
        # 同时保存 JSON（便于手动编辑）
        json_file = self.characters_dir / "characters.json"
        try:
            data = {
                "characters": [c.to_dict() for c in self.characters.values()],
                "updated_at": datetime.now().isoformat()
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存角色 JSON 失败: {e}")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        检测帧中的人脸
        
        Args:
            frame: BGR 格式的图像
        
        Returns:
            FaceDetection 列表
        """
        try:
            import face_recognition
            
            # BGR 转 RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 检测人脸位置
            face_locations = face_recognition.face_locations(
                rgb_frame,
                model=self.config.detection_model
            )
            
            if not face_locations:
                return []
            
            # 计算人脸编码
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            detections = []
            for location, encoding in zip(face_locations, face_encodings):
                detection = FaceDetection(
                    location=location,
                    encoding=encoding
                )
                detections.append(detection)
            
            return detections
            
        except ImportError:
            logger.warning("face_recognition 库未安装，使用 OpenCV 人脸检测")
            return self._detect_faces_opencv(frame)
        except Exception as e:
            logger.error(f"人脸检测失败: {e}")
            return []
    
    def _detect_faces_opencv(self, frame: np.ndarray) -> List[FaceDetection]:
        """使用 OpenCV 检测人脸（备用方案）"""
        try:
            # 加载预训练的人脸检测器
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            detections = []
            for (x, y, w, h) in faces:
                # 转换为 face_recognition 格式的位置
                location = (y, x + w, y + h, x)
                detections.append(FaceDetection(location=location))
            
            return detections
            
        except Exception as e:
            logger.error(f"OpenCV 人脸检测失败: {e}")
            return []
    
    def recognize_faces(
        self, 
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> List[FaceDetection]:
        """
        识别帧中的人脸并匹配角色
        
        Args:
            frame: BGR 格式的图像
            timestamp: 当前时间戳
        
        Returns:
            带有角色信息的 FaceDetection 列表
        """
        detections = self.detect_faces(frame)
        
        if not detections or not self.characters:
            return detections
        
        try:
            import face_recognition
            
            # 收集所有已知的人脸编码
            known_encodings = []
            known_ids = []
            
            for char_id, character in self.characters.items():
                for encoding in character.face_encodings:
                    known_encodings.append(encoding)
                    known_ids.append(char_id)
            
            if not known_encodings:
                return detections
            
            # 匹配每个检测到的人脸
            for detection in detections:
                if detection.encoding is None:
                    continue
                
                # 计算与已知人脸的距离
                distances = face_recognition.face_distance(
                    known_encodings, 
                    detection.encoding
                )
                
                if len(distances) > 0:
                    min_idx = np.argmin(distances)
                    min_distance = distances[min_idx]
                    
                    # 判断是否匹配
                    threshold = self.config.threshold
                    if min_distance < threshold:
                        char_id = known_ids[min_idx]
                        character = self.characters[char_id]
                        
                        detection.character_id = char_id
                        detection.character_name = character.name
                        detection.confidence = 1 - min_distance
                        
                        # 更新角色出现信息
                        character.appearance_count += 1
                        character.last_seen = timestamp
            
            return detections
            
        except Exception as e:
            logger.error(f"人脸识别失败: {e}")
            return detections
    
    def add_character(
        self,
        name: str,
        face_image: Optional[np.ndarray] = None,
        aliases: Optional[List[str]] = None,
        description: str = ""
    ) -> Character:
        """
        添加新角色
        
        Args:
            name: 角色名称
            face_image: BGR 格式的人脸图像
            aliases: 别名列表
            description: 角色描述
        
        Returns:
            创建的 Character 对象
        """
        char_id = f"char_{len(self.characters) + 1}_{name}"
        
        character = Character(
            id=char_id,
            name=name,
            aliases=aliases or [],
            description=description
        )
        
        if face_image is not None:
            self._add_face_encoding(character, face_image)
        
        self.characters[char_id] = character
        logger.info(f"添加角色: {name} (ID: {char_id})")
        
        return character
    
    def _add_face_encoding(self, character: Character, face_image: np.ndarray):
        """为角色添加人脸编码"""
        try:
            import face_recognition
            
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)
            
            if encodings:
                character.face_encodings.append(encodings[0])
                
        except Exception as e:
            logger.error(f"添加人脸编码失败: {e}")
    
    def add_face_to_character(
        self, 
        character_id: str, 
        face_image: np.ndarray
    ) -> bool:
        """
        为已有角色添加新的人脸图像
        
        Args:
            character_id: 角色 ID
            face_image: BGR 格式的人脸图像
        
        Returns:
            是否成功
        """
        if character_id not in self.characters:
            logger.error(f"角色不存在: {character_id}")
            return False
        
        self._add_face_encoding(self.characters[character_id], face_image)
        return True
    
    def get_character(self, character_id: str) -> Optional[Character]:
        """获取角色信息"""
        return self.characters.get(character_id)
    
    def get_character_by_name(self, name: str) -> Optional[Character]:
        """按名称获取角色"""
        for character in self.characters.values():
            if character.name == name or name in character.aliases:
                return character
        return None
    
    def list_characters(self) -> List[Character]:
        """列出所有角色"""
        return list(self.characters.values())
    
    def get_characters_in_frame(
        self, 
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> List[str]:
        """
        获取帧中出现的角色名称
        
        Args:
            frame: BGR 格式的图像
            timestamp: 时间戳
        
        Returns:
            角色名称列表
        """
        detections = self.recognize_faces(frame, timestamp)
        
        names = []
        for detection in detections:
            if detection.character_name:
                names.append(detection.character_name)
            else:
                # 未识别的人物
                names.append(f"未知人物{len(names) + 1}")
        
        return names
    
    def register_unknown_face(
        self, 
        face_encoding: np.ndarray,
        timestamp: float
    ):
        """
        注册未知人脸（用于后续手动标注）
        
        Args:
            face_encoding: 人脸编码
            timestamp: 出现时间
        """
        self._unknown_faces.append((face_encoding, timestamp))
        
        # 限制缓存大小
        if len(self._unknown_faces) > 100:
            self._unknown_faces = self._unknown_faces[-50:]
    
    def get_unknown_faces_count(self) -> int:
        """获取未知人脸数量"""
        return len(self._unknown_faces)


class CharacterTracker:
    """角色追踪器 - 在视频序列中追踪角色"""
    
    def __init__(self, recognizer: CharacterRecognizer):
        self.recognizer = recognizer
        self._active_characters: Dict[str, float] = {}  # character_id -> last_seen
        self._scene_characters: List[str] = []
    
    def update(self, frame: np.ndarray, timestamp: float):
        """更新追踪状态"""
        detections = self.recognizer.recognize_faces(frame, timestamp)
        
        # 更新活跃角色
        current_characters = []
        for detection in detections:
            if detection.character_id:
                self._active_characters[detection.character_id] = timestamp
                current_characters.append(detection.character_name or detection.character_id)
        
        self._scene_characters = current_characters
        
        # 清理长时间未出现的角色
        timeout = 10.0  # 10秒超时
        self._active_characters = {
            k: v for k, v in self._active_characters.items()
            if timestamp - v < timeout
        }
    
    def get_current_characters(self) -> List[str]:
        """获取当前场景中的角色"""
        return self._scene_characters
    
    def get_active_characters(self) -> List[str]:
        """获取所有活跃角色（包括最近出现的）"""
        return [
            self.recognizer.characters[cid].name
            for cid in self._active_characters.keys()
            if cid in self.recognizer.characters
        ]
