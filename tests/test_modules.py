"""
LuminaLink Test Module

Tests all module functionality
"""

import pytest
import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfig:
    """Test configuration module"""
    
    def test_get_config(self):
        """Test getting configuration"""
        from src.config import get_config
        config = get_config()
        
        assert config is not None
        assert hasattr(config, "ai")
        assert hasattr(config, "tts")
        assert hasattr(config, "narration")
    
    def test_config_paths(self):
        """Test path configuration"""
        from src.config import get_config
        config = get_config()
        
        assert config.paths.project_root.exists()
        assert config.paths.cache_dir is not None


class TestVideoProcessor:
    """Test video processor"""
    
    def test_video_processor_import(self):
        """Test import"""
        from src.video_processor import VideoProcessor, VideoFrame
        assert VideoProcessor is not None
        assert VideoFrame is not None
    
    def test_frame_buffer(self):
        """Test frame buffer"""
        from src.video_processor import FrameBuffer
        import numpy as np
        
        buffer = FrameBuffer(max_size=3)
        
        # Add frames
        for i in range(5):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            buffer.add_frame(frame, i * 0.1)
        
        # Check buffer size
        assert len(buffer) == 3
        
        # Get frame
        frame_data = buffer.get_frame(0.2)
        assert frame_data is not None


class TestAudioDetector:
    """Test audio detector"""
    
    def test_audio_detector_import(self):
        """Test import"""
        from src.audio_detector import AudioDetector, AudioSegment, SilenceWindow
        assert AudioDetector is not None
        assert AudioSegment is not None
        assert SilenceWindow is not None
    
    def test_audio_segment_creation(self):
        """Test audio segment creation"""
        from src.audio_detector import AudioSegment
        
        segment = AudioSegment(
            start_time=0.0,
            end_time=1.0,
            has_speech=True,
            volume_db=0.5
        )
        
        assert segment.duration == 1.0

    def test_silence_duration_tracking(self):
        """Test that silence duration is tracked."""
        from src.audio_detector import RealtimeAudioDetector
        import numpy as np

        detector = RealtimeAudioDetector(silence_threshold_db=-20.0)

        # Feed silent audio
        silent_chunk = np.zeros(512, dtype=np.float32)
        detector.feed_audio(silent_chunk)

        assert detector.is_current_silence()
        assert detector.get_silence_duration() >= 0.0

        # Feed loud audio — use a chunk large enough to fill the entire buffer
        sample_rate = detector.sample_rate
        buffer_size = int(sample_rate * detector.buffer_duration)
        loud_chunk = np.ones(buffer_size, dtype=np.float32) * 0.5
        detector.feed_audio(loud_chunk)

        assert not detector.is_current_silence()
        assert detector.get_silence_duration() == 0.0
        assert not detector.is_silence_long_enough(min_duration=1.5)


class TestCharacterRecognizer:
    """Test character recognition"""
    
    def test_recognizer_import(self):
        """Test import"""
        from src.character_recognizer import CharacterRecognizer, Character
        assert CharacterRecognizer is not None
        assert Character is not None
    
    def test_add_character(self):
        """Test adding character"""
        from src.character_recognizer import CharacterRecognizer
        
        recognizer = CharacterRecognizer()
        char = recognizer.add_character(
            name="John",
            aliases=["Johnny"],
            description="The protagonist"
        )
        
        assert char.name == "John"
        assert "Johnny" in char.aliases
        
        # Get character
        result = recognizer.get_character("John")
        assert result is not None
        assert result.name == "John"
        
        # Get by alias
        result = recognizer.get_character("Johnny")
        assert result is not None
        assert result.name == "John"


class TestSceneAnalyzer:
    """Test scene analyzer"""
    
    def test_analyzer_import(self):
        """Test import"""
        from src.scene_analyzer import SceneAnalyzer, SceneAnalysis
        assert SceneAnalyzer is not None
        assert SceneAnalysis is not None
    
    def test_set_characters(self):
        """Test setting characters"""
        from src.scene_analyzer import SceneAnalyzer
        
        analyzer = SceneAnalyzer()
        analyzer.set_characters(["John", "Mary"])
        
        assert len(analyzer.known_characters) == 2


class TestNarrator:
    """Test narration generator"""
    
    def test_narrator_import(self):
        """Test import"""
        from src.narrator import Narrator, Narration, NarrationStyle
        assert Narrator is not None
        assert Narration is not None
        assert NarrationStyle is not None
    
    def test_should_narrate(self):
        """Test narration timing"""
        from src.narrator import Narrator
        
        narrator = Narrator()
        
        # First time should be allowed
        assert narrator.should_narrate(0.0)
        
        # Immediately after should be rejected
        narrator._last_narration_time = 0.0
        assert not narrator.should_narrate(0.5)
        
        # After interval should be allowed
        assert narrator.should_narrate(10.0)
    
    def test_is_refusal_response(self):
        """Test refusal detection"""
        from src.narrator import Narrator
        
        narrator = Narrator()
        
        # Should detect refusal
        assert narrator._is_refusal_response("Sorry, I cannot assist with that request.")
        assert narrator._is_refusal_response("I am unable to describe copyrighted content.")
        
        # Normal narration should pass
        assert not narrator._is_refusal_response("John walks into the room and greets Mary.")


class TestTTSEngine:
    """Test TTS engine"""
    
    def test_tts_import(self):
        """Test import"""
        from src.tts_engine import TTSManager, TTSResult, EdgeTTSEngine
        assert TTSManager is not None
        assert TTSResult is not None
        assert EdgeTTSEngine is not None
    
    def test_tts_result(self):
        """Test TTS result"""
        from src.tts_engine import TTSResult
        
        result = TTSResult(
            audio_path="/tmp/test.mp3",
            duration=2.5,
            text="Hello world",
            success=True
        )
        
        assert result.success
        assert result.duration == 2.5


@pytest.mark.asyncio
async def test_edge_tts_format_rate():
    """Test Edge TTS rate formatting"""
    from src.tts_engine import EdgeTTSEngine
    
    engine = EdgeTTSEngine()
    
    assert engine._format_rate(1.0) == "+0%"
    assert engine._format_rate(1.2) == "+20%"
    assert engine._format_rate(0.8) == "-20%"


def test_all_modules_import():
    """Test all modules import successfully"""
    from src import config
    from src import video_processor
    from src import audio_detector
    from src import character_recognizer
    from src import scene_analyzer
    from src import narrator
    from src import tts_engine
    from src import main
    from src import realtime_player
    
    print("All modules imported successfully!")


if __name__ == "__main__":
    # Run basic tests
    print("Running basic tests...\n")
    
    # Test config
    test_config = TestConfig()
    test_config.test_get_config()
    print("✓ Config test passed")
    
    # Test video processor
    test_vp = TestVideoProcessor()
    test_vp.test_video_processor_import()
    test_vp.test_frame_buffer()
    print("✓ Video processor test passed")
    
    # Test audio detector
    test_ad = TestAudioDetector()
    test_ad.test_audio_detector_import()
    test_ad.test_audio_segment_creation()
    print("✓ Audio detector test passed")
    
    # Test character recognizer
    test_cr = TestCharacterRecognizer()
    test_cr.test_recognizer_import()
    test_cr.test_add_character()
    print("✓ Character recognizer test passed")
    
    # Test scene analyzer
    test_sa = TestSceneAnalyzer()
    test_sa.test_analyzer_import()
    test_sa.test_set_characters()
    print("✓ Scene analyzer test passed")
    
    # Test narrator
    test_nr = TestNarrator()
    test_nr.test_narrator_import()
    test_nr.test_should_narrate()
    test_nr.test_is_refusal_response()
    print("✓ Narrator test passed")
    
    # Test TTS engine
    test_tts = TestTTSEngine()
    test_tts.test_tts_import()
    test_tts.test_tts_result()
    print("✓ TTS engine test passed")
    
    # Test all imports
    test_all_modules_import()
    
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)
