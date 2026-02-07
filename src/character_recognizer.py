"""
Character Recognition Module

Identifies characters in video, maintains character database, associates faces with character names
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
    """Character information"""
    id: str                          # Unique identifier
    name: str                        # Character name
    aliases: List[str] = field(default_factory=list)  # Alias list
    description: str = ""            # Character description
    face_encodings: List[np.ndarray] = field(default_factory=list)  # Face encoding list
    appearance_count: int = 0        # Appearance count
    last_seen: Optional[float] = None  # Last seen timestamp
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary (without face encodings)"""
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
    """Face detection result"""
    location: Tuple[int, int, int, int]  # (top, right, bottom, left)
    encoding: Optional[np.ndarray] = None
    character_id: Optional[str] = None
    character_name: Optional[str] = None
    confidence: float = 0.0
    
    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        """Get bounding box (x, y, width, height)"""
        top, right, bottom, left = self.location
        return (left, top, right - left, bottom - top)


class CharacterRecognizer:
    """Character Recognizer"""
    
    def __init__(self, characters_dir: Optional[Path] = None):
        """
        Initialize character recognizer
        
        Args:
            characters_dir: Character data directory
        """
        self.config = get_config().face_recognition
        self.characters_dir = characters_dir or get_config().paths.characters_dir
        
        # Character database
        self.characters: Dict[str, Character] = {}
        
        # Unknown faces cache (for auto-learning)
        self._unknown_faces: List[Tuple[np.ndarray, float]] = []
        
        # Recent recognition results cache
        self._recognition_cache: Dict[str, Tuple[str, float]] = {}
        
        # Try to load existing character data
        self._load_characters()
    
    def _load_characters(self):
        """Load character data"""
        db_file = self.characters_dir / "characters.pkl"
        
        if db_file.exists():
            try:
                with open(db_file, "rb") as f:
                    self.characters = pickle.load(f)
                logger.info(f"Loaded {len(self.characters)} characters")
            except Exception as e:
                logger.error(f"Failed to load character data: {e}")
        
        # Also load JSON config (for manual editing)
        json_file = self.characters_dir / "characters.json"
        if json_file.exists():
            self._load_characters_json(json_file)
    
    def _load_characters_json(self, json_path: Path):
        """Load character config from JSON file"""
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            for char_data in data.get("characters", []):
                char_id = char_data.get("id", char_data["name"])
                
                # Create new character if not exists
                if char_id not in self.characters:
                    self.characters[char_id] = Character(
                        id=char_id,
                        name=char_data["name"],
                        aliases=char_data.get("aliases", []),
                        description=char_data.get("description", "")
                    )
                else:
                    # Update existing character info
                    self.characters[char_id].name = char_data["name"]
                    self.characters[char_id].aliases = char_data.get("aliases", [])
                    self.characters[char_id].description = char_data.get("description", "")
                
                # Load character images
                for img_path in char_data.get("face_images", []):
                    full_path = self.characters_dir / img_path
                    if full_path.exists():
                        self._add_face_from_image(char_id, str(full_path))
                        
        except Exception as e:
            logger.error(f"Failed to load character JSON: {e}")
    
    def _add_face_from_image(self, character_id: str, image_path: str):
        """Add face encoding from image"""
        try:
            import face_recognition
            
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if encodings and character_id in self.characters:
                self.characters[character_id].face_encodings.append(encodings[0])
                logger.debug(f"Added face encoding: {character_id}")
                
        except Exception as e:
            logger.error(f"Failed to add face {image_path}: {e}")
    
    def save_characters(self):
        """Save character data"""
        self.characters_dir.mkdir(parents=True, exist_ok=True)
        
        # Save binary data (including face encodings)
        db_file = self.characters_dir / "characters.pkl"
        try:
            with open(db_file, "wb") as f:
                pickle.dump(self.characters, f)
            logger.info(f"Character data saved: {len(self.characters)} characters")
        except Exception as e:
            logger.error(f"Failed to save character data: {e}")
        
        # Also save JSON (for manual editing)
        json_file = self.characters_dir / "characters.json"
        try:
            data = {
                "characters": [c.to_dict() for c in self.characters.values()],
                "updated_at": datetime.now().isoformat()
            }
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save character JSON: {e}")
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in frame
        
        Args:
            frame: BGR format image
        
        Returns:
            List of FaceDetection
        """
        try:
            import face_recognition
            
            # BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(
                rgb_frame,
                model=self.config.detection_model
            )
            
            if not face_locations:
                return []
            
            # Calculate face encodings
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
            logger.warning("face_recognition library not installed, using OpenCV face detection")
            return self._detect_faces_opencv(frame)
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def _detect_faces_opencv(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces using OpenCV (fallback)"""
        try:
            # Load pre-trained face detector
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            detections = []
            for (x, y, w, h) in faces:
                # Convert to face_recognition format location
                location = (y, x + w, y + h, x)
                detections.append(FaceDetection(location=location))
            
            return detections
            
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return []
    
    def recognize_faces(
        self, 
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> List[FaceDetection]:
        """
        Recognize faces in frame and match characters
        
        Args:
            frame: BGR format image
            timestamp: Current timestamp
        
        Returns:
            List of FaceDetection with character info
        """
        detections = self.detect_faces(frame)
        
        if not detections or not self.characters:
            return detections
        
        try:
            import face_recognition
            
            # Collect all known face encodings
            known_encodings = []
            known_ids = []
            
            for char_id, character in self.characters.items():
                for encoding in character.face_encodings:
                    known_encodings.append(encoding)
                    known_ids.append(char_id)
            
            if not known_encodings:
                return detections
            
            # Match each detected face
            for detection in detections:
                if detection.encoding is None:
                    continue
                
                # Calculate distance to known faces
                distances = face_recognition.face_distance(
                    known_encodings, 
                    detection.encoding
                )
                
                if len(distances) > 0:
                    min_idx = np.argmin(distances)
                    min_distance = distances[min_idx]
                    
                    # Check if match
                    threshold = self.config.threshold
                    if min_distance < threshold:
                        char_id = known_ids[min_idx]
                        character = self.characters[char_id]
                        
                        detection.character_id = char_id
                        detection.character_name = character.name
                        detection.confidence = 1 - min_distance
                        
                        # Update character appearance info
                        character.appearance_count += 1
                        character.last_seen = timestamp
            
            return detections
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return detections
    
    def add_character(
        self,
        name: str,
        face_image: Optional[np.ndarray] = None,
        aliases: Optional[List[str]] = None,
        description: str = ""
    ) -> Character:
        """
        Add new character
        
        Args:
            name: Character name
            face_image: BGR format face image
            aliases: Alias list
            description: Character description
        
        Returns:
            Created Character object
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
        logger.info(f"Added character: {name} (ID: {char_id})")
        
        return character
    
    def _add_face_encoding(self, character: Character, face_image: np.ndarray):
        """Add face encoding to character"""
        try:
            import face_recognition
            
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_image)
            
            if encodings:
                character.face_encodings.append(encodings[0])
                
        except Exception as e:
            logger.error(f"Failed to add face encoding: {e}")
    
    def add_face_to_character(
        self, 
        character_id: str, 
        face_image: np.ndarray
    ) -> bool:
        """
        Add new face image to existing character
        
        Args:
            character_id: Character ID
            face_image: BGR format face image
        
        Returns:
            Whether successful
        """
        if character_id not in self.characters:
            logger.error(f"Character not found: {character_id}")
            return False
        
        self._add_face_encoding(self.characters[character_id], face_image)
        return True
    
    def get_character(self, character_id: str) -> Optional[Character]:
        """Get character info"""
        return self.characters.get(character_id)
    
    def get_character_by_name(self, name: str) -> Optional[Character]:
        """Get character by name"""
        for character in self.characters.values():
            if character.name == name or name in character.aliases:
                return character
        return None
    
    def list_characters(self) -> List[Character]:
        """List all characters"""
        return list(self.characters.values())
    
    def get_characters_in_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ) -> List[str]:
        """
        Get character names in frame
        
        Args:
            frame: BGR format image
            timestamp: Current timestamp
        
        Returns:
            List of character names
        """
        detections = self.recognize_faces(frame, timestamp)
        
        names = []
        for detection in detections:
            if detection.character_name:
                names.append(detection.character_name)
            else:
                # Unknown character
                names.append(f"Unknown{len(names) + 1}")
        
        return names
    
    def remove_character(self, character_id: str) -> bool:
        """Remove character"""
        if character_id in self.characters:
            del self.characters[character_id]
            logger.info(f"Removed character: {character_id}")
            return True
        return False
    
    def clear_all(self):
        """Clear all character data"""
        self.characters.clear()
        self._unknown_faces.clear()
        self._recognition_cache.clear()
        logger.info("All character data cleared")


class CharacterTracker:
    """Character Tracker - tracks character positions across frames"""
    
    def __init__(self, recognizer: CharacterRecognizer):
        """
        Initialize character tracker
        
        Args:
            recognizer: Character recognizer
        """
        self.recognizer = recognizer
        self._tracks: Dict[str, List[Tuple[float, FaceDetection]]] = {}
        self._max_track_length = 30
    
    def update(
        self,
        frame: np.ndarray,
        timestamp: float
    ) -> List[FaceDetection]:
        """
        Update tracker with new frame
        
        Args:
            frame: BGR format image
            timestamp: Current timestamp
        
        Returns:
            List of FaceDetection with character info
        """
        detections = self.recognizer.recognize_faces(frame, timestamp)
        
        # Update tracks
        for detection in detections:
            char_id = detection.character_id or "unknown"
            
            if char_id not in self._tracks:
                self._tracks[char_id] = []
            
            self._tracks[char_id].append((timestamp, detection))
            
            # Limit track length
            if len(self._tracks[char_id]) > self._max_track_length:
                self._tracks[char_id].pop(0)
        
        return detections
    
    def get_character_history(
        self,
        character_id: str
    ) -> List[Tuple[float, FaceDetection]]:
        """Get character appearance history"""
        return self._tracks.get(character_id, [])
    
    def get_active_characters(
        self,
        timestamp: float,
        window: float = 5.0
    ) -> List[str]:
        """
        Get recently active characters
        
        Args:
            timestamp: Current timestamp
            window: Time window (seconds)
        
        Returns:
            List of character IDs
        """
        active = []
        min_time = timestamp - window
        
        for char_id, track in self._tracks.items():
            for t, _ in reversed(track):
                if t >= min_time:
                    active.append(char_id)
                    break
        
        return active
    
    def clear(self):
        """Clear all tracks"""
        self._tracks.clear()
