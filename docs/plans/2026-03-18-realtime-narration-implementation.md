# Real-Time Narration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix LuminaLink so blind users get real-time spoken narrations of a film playing on TV, captured via webcam + microphone.

**Architecture:** 4-thread pipeline — frame capture, audio monitoring, scene analysis (GPT-4o), and TTS playback. The missing `narrator.py` module is created; broken async calls in the narration worker are fixed; config defaults are corrected to English; microphone device selection is wired through the CLI.

**Tech Stack:** Python 3.8+, OpenAI GPT-4o, Edge TTS, sounddevice, OpenCV, asyncio, threading, Tkinter

---

## Task 1: Create `src/narrator.py` (THE MISSING MODULE)

**Files:**
- Create: `src/narrator.py`
- Test: `tests/test_modules.py` (TestNarrator class already written — just needs narrator.py to exist)

**Step 1: Write the failing test (it already exists)**

Run: `pytest tests/test_modules.py::TestNarrator -v`
Expected: `ModuleNotFoundError: No module named 'src.narrator'`

**Step 2: Create `src/narrator.py`**

```python
"""
Narrator Module

Generates, filters, and tracks real-time narration from scene analysis results.
"""

import logging
import re
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
        self._cooldown = cooldown if cooldown is not None else config.narration.interval
        self._max_length = config.narration.max_length
        self._history: List[Narration] = []
        self._last_narration_time: float = -999.0

    def should_narrate(self, timestamp: float) -> bool:
        """Return True if enough time has elapsed since the last narration."""
        return (timestamp - self._last_narration_time) >= self._cooldown

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

        if not text:
            return None

        if self._is_refusal_response(text):
            logger.warning(f"Refusal response detected, skipping: {text[:60]}")
            return None

        if self._is_duplicate(text):
            logger.debug("Duplicate narration skipped")
            return None

        if len(text) > self._max_length:
            text = self._truncate(text, self._max_length)

        narration = Narration(
            text=text,
            timestamp=timestamp,
            duration=slot[1] - slot[0],
            style=self._style,
        )

        self._history.append(narration)
        self._last_narration_time = timestamp
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

    def _is_duplicate(self, text: str, threshold: float = 0.6) -> bool:
        """Jaccard word-overlap check against the last 3 narrations."""
        if not self._history:
            return False
        words = set(text.lower().split())
        if not words:
            return False
        for recent in self._history[-3:]:
            recent_words = set(recent.text.lower().split())
            if not recent_words:
                continue
            union = words | recent_words
            if union and len(words & recent_words) / len(union) >= threshold:
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
```

**Step 3: Run the narrator tests**

Run: `pytest tests/test_modules.py::TestNarrator -v`
Expected: All 3 narrator tests PASS

**Step 4: Commit**

```bash
git add src/narrator.py
git commit -m "feat: add narrator module — the missing real-time narration coordinator"
```

---

## Task 2: Fix `src/config.py` — Wrong Defaults

**Files:**
- Modify: `src/config.py`

**Step 1: Run config tests first**

Run: `pytest tests/test_modules.py::TestConfig -v`
Expected: `AttributeError: 'PathConfig' object has no attribute 'project_root'`

**Step 2: Edit `src/config.py`**

Make these 3 targeted changes:

**Change 1** — Fix `AIConfig.openai_model` default (line 28):
```python
# OLD:
openai_model: str = "gpt-4-vision-preview"
# NEW:
openai_model: str = "gpt-4o"
```

**Change 2** — Fix `TTSConfig.voice` default (line 57):
```python
# OLD:
voice: str = "zh-CN-XiaoxiaoNeural"
# NEW:
voice: str = "en-US-AriaNeural"
```

**Change 3** — Add `project_root` to `PathConfig` (after the class definition line, before `__post_init__`):
```python
@dataclass
class PathConfig:
    """Path Configuration"""
    project_root: Path = field(default_factory=lambda: PROJECT_ROOT)
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
```

**Step 3: Run config tests**

Run: `pytest tests/test_modules.py::TestConfig -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/config.py
git commit -m "fix: update config defaults to English voice and gpt-4o model"
```

---

## Task 3: Add Silence Duration Tracking to `src/audio_detector.py`

**Files:**
- Modify: `src/audio_detector.py`

**Background:** `RealtimeAudioDetector.is_current_silence()` checks if the buffer is currently silent, but doesn't track *how long* silence has been ongoing. We need at least 1.5s of continuous silence before triggering narration.

**Step 1: Write a failing test**

Add to `tests/test_modules.py` inside `TestAudioDetector`:
```python
def test_silence_duration_tracking(self):
    """Test that silence duration is tracked correctly."""
    from src.audio_detector import RealtimeAudioDetector
    import numpy as np
    import time

    detector = RealtimeAudioDetector(silence_threshold_db=-20.0)

    # Feed silent audio
    silent_chunk = np.zeros(512, dtype=np.float32)
    detector.feed_audio(silent_chunk)

    # Should be silence
    assert detector.is_current_silence()
    # Silence just started, duration should be >= 0
    assert detector.get_silence_duration() >= 0.0

    # Feed loud audio
    loud_chunk = np.ones(512, dtype=np.float32) * 0.5
    detector.feed_audio(loud_chunk)

    # Should not be silence
    assert not detector.is_current_silence()
    assert detector.get_silence_duration() == 0.0
```

Run: `pytest tests/test_modules.py::TestAudioDetector::test_silence_duration_tracking -v`
Expected: `AttributeError: 'RealtimeAudioDetector' object has no attribute 'get_silence_duration'`

**Step 2: Update `RealtimeAudioDetector` in `src/audio_detector.py`**

Add `import time` at the top of the file (after the existing imports).

Replace the `RealtimeAudioDetector.__init__` and `_update_silence_state` methods and add new methods:

```python
class RealtimeAudioDetector:
    """Realtime audio detector for live audio streams"""

    def __init__(
        self,
        silence_threshold_db: float = -40.0,
        buffer_duration: float = 2.0,
        sample_rate: int = 22050
    ):
        self.silence_threshold_db = silence_threshold_db
        self.buffer_duration = buffer_duration
        self.sample_rate = sample_rate

        buffer_size = int(sample_rate * buffer_duration)
        self._buffer = np.zeros(buffer_size, dtype=np.float32)
        self._buffer_pos = 0
        self._is_silence = True
        self._silence_start: Optional[float] = None  # monotonic time when silence began

    # ... (feed_audio unchanged) ...

    def _update_silence_state(self):
        """Update silence state and track duration."""
        rms = np.sqrt(np.mean(self._buffer ** 2))
        volume_db = 20 * np.log10(rms) if rms > 0 else -100

        was_silent = self._is_silence
        self._is_silence = volume_db < self.silence_threshold_db

        now = time.monotonic()
        if self._is_silence and not was_silent:
            # Silence just started
            self._silence_start = now
        elif not self._is_silence:
            # Sound detected, reset silence tracking
            self._silence_start = None

    def is_current_silence(self) -> bool:
        """Check if current audio is silence"""
        return self._is_silence

    def get_silence_duration(self) -> float:
        """Return seconds of continuous silence elapsed. 0.0 if not silent."""
        if not self._is_silence or self._silence_start is None:
            return 0.0
        return time.monotonic() - self._silence_start

    def is_silence_long_enough(self, min_duration: float = 1.5) -> bool:
        """Return True if silence has been ongoing for at least min_duration seconds."""
        return self.get_silence_duration() >= min_duration

    def get_current_volume_db(self) -> float:
        """Get current volume (dB)"""
        rms = np.sqrt(np.mean(self._buffer ** 2))
        return 20 * np.log10(rms) if rms > 0 else -100

    def reset(self):
        """Reset detector"""
        self._buffer.fill(0)
        self._buffer_pos = 0
        self._is_silence = True
        self._silence_start = None
```

Also add `import time` at the top of `audio_detector.py`.

**Step 3: Run the new test**

Run: `pytest tests/test_modules.py::TestAudioDetector -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/audio_detector.py tests/test_modules.py
git commit -m "feat: add silence duration tracking to RealtimeAudioDetector"
```

---

## Task 4: Add Microphone Device Selection to `src/audio_input.py`

**Files:**
- Modify: `src/audio_input.py`

**Background:** Currently `AudioInputStream` always uses the default mic. Users need `--mic N` to pick a specific device.

**Step 1: Update `AudioInputStream.__init__`**

Change the signature and `start()` method:

```python
def __init__(self, detector, sample_rate: int = 22050, blocksize: int = 512, device_id: Optional[int] = None):
    """
    Initialize audio input stream.

    Args:
        detector: RealtimeAudioDetector instance to feed audio to
        sample_rate: Audio sample rate in Hz (default: 22050)
        blocksize: Audio block size in frames (default: 512, ~23ms at 22050Hz)
        device_id: Microphone device index (None = system default)
    """
    if not SOUNDDEVICE_AVAILABLE:
        raise ImportError(
            "sounddevice library not available. "
            "Install it with: pip install sounddevice"
        )

    self.detector = detector
    self.sample_rate = sample_rate
    self.blocksize = blocksize
    self.device_id = device_id
    self.stream: Optional[sd.InputStream] = None
    self._running = False
```

In `start()`, update the `sd.InputStream` call to include `device`:

```python
self.stream = sd.InputStream(
    device=self.device_id,       # <-- ADD THIS LINE
    samplerate=self.sample_rate,
    channels=1,
    callback=self._audio_callback,
    blocksize=self.blocksize,
)
```

**Step 2: Verify no regressions**

Run: `pytest tests/test_modules.py -v`
Expected: All tests that were passing still PASS

**Step 3: Commit**

```bash
git add src/audio_input.py
git commit -m "feat: add device_id parameter to AudioInputStream for mic selection"
```

---

## Task 5: Fix Broken Async in `src/camera_controller.py`

**Files:**
- Modify: `src/camera_controller.py`

**Background:** `_narration_worker` calls `self.tts_manager.synthesize()` (a coroutine) without await, and `self.audio_player.play()` (also async) without await. Both calls silently fail — no audio is ever played.

**Step 1: Understand the current broken code (lines 407–420)**

```python
# BROKEN — synthesize() returns a coroutine, not a TTSResult
audio_file = self.tts_manager.synthesize(narration.text)
# BROKEN — play() is async, called synchronously
self.audio_player.play(audio_file)
```

**Step 2: Replace the `_narration_worker` method entirely**

```python
def _narration_worker(self):
    """TTS generation and playback worker thread"""
    # Own event loop for this thread (same pattern as _analysis_worker)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        while not self._stop_event.is_set():
            try:
                narration = self._narration_queue.get(timeout=1.0)
            except Empty:
                continue

            # Wait if paused
            self._pause_event.wait()

            try:
                # Notify GUI
                if self._on_narration_callback:
                    try:
                        self._on_narration_callback(narration.text)
                    except Exception as e:
                        logger.error(f"Narration callback error: {e}")

                self.state.current_narration = narration.text

                # Synthesize TTS audio (async → sync via loop)
                try:
                    result = loop.run_until_complete(
                        self.tts_manager.synthesize(narration.text)
                    )
                except Exception as e:
                    logger.error(f"TTS synthesis error: {e}")
                    continue

                if not result.success:
                    logger.error(f"TTS synthesis failed: {result.error}")
                    continue

                # Play audio — blocks until playback completes
                try:
                    loop.run_until_complete(
                        self.audio_player.play(result.audio_path, block=True)
                    )
                    self._narration_count += 1
                except Exception as e:
                    logger.error(f"Audio playback error: {e}")

                # Clean up temp file
                try:
                    import os as _os
                    if _os.path.exists(result.audio_path):
                        _os.remove(result.audio_path)
                except Exception:
                    pass

                self.state.current_narration = None

            except Exception as e:
                logger.error(f"Narration worker error: {e}")
    finally:
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        logger.info("Narration worker stopped")
```

**Step 3: Also add `mic_device_index` and `cooldown` parameters to `__init__`**

Update `__init__` signature:
```python
def __init__(
    self,
    camera_index: int = 0,
    characters_config: Optional[str] = None,
    camera_width: int = 1280,
    camera_height: int = 720,
    camera_fps: int = 30,
    mic_device_index: Optional[int] = None,   # NEW
    cooldown: float = 5.0,                     # NEW
):
```

Pass `device_id` when creating `AudioInputStream`:
```python
# OLD:
self.audio_stream = AudioInputStream(
    detector=self.audio_detector,
    sample_rate=22050
)
# NEW:
self.audio_stream = AudioInputStream(
    detector=self.audio_detector,
    sample_rate=22050,
    device_id=mic_device_index
)
```

Pass `cooldown` to `Narrator`:
```python
# OLD:
self.narrator = Narrator()
# NEW:
self.narrator = Narrator(cooldown=cooldown)
```

**Step 4: Fix silence check in `_analysis_worker` to use duration (line 316)**

```python
# OLD:
if not self.audio_detector.is_current_silence():
    logger.debug(f"Skipping narration: dialogue detected")
    continue
# NEW:
if not self.audio_detector.is_silence_long_enough(min_duration=1.5):
    logger.debug("Skipping narration: insufficient silence (need 1.5s)")
    continue
```

**Step 5: Verify import still works**

Run: `python -c "from src.camera_controller import CameraRealtimeController; print('OK')"`
Expected: `OK`

**Step 6: Commit**

```bash
git add src/camera_controller.py
git commit -m "fix: repair async TTS calls in narration worker, add mic/cooldown params"
```

---

## Task 6: Add CLI Flags to `src/main.py`

**Files:**
- Modify: `src/main.py`

**Background:** The `camera` CLI command needs `--mic`, `--silence-threshold`, and `--cooldown` flags so users can configure the system from the command line without editing `.env`.

**Step 1: Update the `camera` command options**

Find the `camera` command definition (around line 381). Add 3 new options after the existing `--fps` option:

```python
@click.option(
    "--mic", "-m",
    default=0,
    type=int,
    help="Microphone device index (default: 0). Run with --list-mics to see options."
)
@click.option(
    "--silence-threshold",
    default=-35.0,
    type=float,
    help="Silence threshold in dB (default: -35). Lower = more sensitive."
)
@click.option(
    "--cooldown",
    default=5.0,
    type=float,
    help="Minimum seconds between narrations (default: 5)."
)
@click.option(
    "--list-mics",
    is_flag=True,
    help="List available microphone devices and exit."
)
```

**Step 2: Update the `camera` function signature and body**

```python
def camera(
    camera: int,
    characters: Optional[str],
    width: int,
    height: int,
    fps: int,
    mic: int,
    silence_threshold: float,
    cooldown: float,
    list_mics: bool
):
    # Handle --list-mics
    if list_mics:
        from .audio_input import AudioInputStream
        devices = AudioInputStream.list_devices()
        if not devices:
            console.print("[yellow]No microphone devices found[/yellow]")
        else:
            table = Table(title="Available Microphones", show_header=True)
            table.add_column("ID", style="cyan")
            table.add_column("Name", style="green")
            table.add_column("Channels")
            for dev in devices:
                table.add_row(str(dev["id"]), dev["name"], str(dev["channels"]))
            console.print(table)
        return

    # ... (existing panel print) ...

    try:
        from .gui.camera_app import CameraApp

        app = CameraApp(
            camera_index=camera,
            characters_config=characters,
            mic_device_index=mic,
            cooldown=cooldown,
        )
        # ... rest unchanged ...
```

**Step 3: Update `CameraApp.__init__` in `src/gui/camera_app.py` to accept and forward the new params**

Find `CameraApp.__init__` in `src/gui/camera_app.py`. Add the new parameters:

```python
def __init__(
    self,
    camera_index: int = 0,
    characters_config: Optional[str] = None,
    mic_device_index: int = 0,       # NEW
    cooldown: float = 5.0,           # NEW
):
    self.camera_index = camera_index
    self.characters_config = characters_config

    self.controller = CameraRealtimeController(
        camera_index=camera_index,
        characters_config=characters_config,
        mic_device_index=mic_device_index,   # NEW
        cooldown=cooldown,                   # NEW
    )
    # ... rest unchanged ...
```

**Step 4: Smoke test the CLI**

Run: `python -m src.main camera --help`
Expected: Shows `--mic`, `--silence-threshold`, `--cooldown`, `--list-mics` options

Run: `python -m src.main camera --list-mics`
Expected: Lists available microphones (or "No microphone devices found")

**Step 5: Commit**

```bash
git add src/main.py src/gui/camera_app.py
git commit -m "feat: add --mic, --silence-threshold, --cooldown, --list-mics CLI flags"
```

---

## Task 7: Fix `tests/test_modules.py` — Broken Tests

**Files:**
- Modify: `tests/test_modules.py`

**Background:** Several tests reference wrong field names or non-existent attributes. All must pass before claiming the system works.

**Step 1: Run ALL tests to see current state**

Run: `pytest tests/test_modules.py -v 2>&1 | head -60`
Expected: Multiple failures from field name mismatches

**Step 2: Fix `TestAudioDetector.test_audio_segment_creation`**

Wrong field names. `AudioSegment` has `has_speech` and `volume_db`, not `is_speech` and `volume`:

```python
def test_audio_segment_creation(self):
    """Test audio segment creation"""
    from src.audio_detector import AudioSegment

    segment = AudioSegment(
        start_time=0.0,
        end_time=1.0,
        has_speech=True,    # was: is_speech=True
        volume_db=-20.0     # was: volume=0.5
    )

    assert segment.duration == 1.0
```

**Step 3: Fix `TestConfig.test_config_paths`**

Add assertion for `project_root` (now exists after Task 2):
```python
def test_config_paths(self):
    """Test path configuration"""
    from src.config import get_config
    config = get_config()

    assert config.paths.project_root.exists()
    assert config.paths.cache_dir is not None
```
(No change needed — this test was already correct, it just needed `project_root` to exist.)

**Step 4: Fix `TestSceneAnalyzer.test_set_characters`**

`_known_characters` is private; the test tries to read it as `known_characters`:

```python
def test_set_characters(self):
    """Test setting characters"""
    from src.scene_analyzer import SceneAnalyzer

    analyzer = SceneAnalyzer()
    analyzer.set_characters(["John", "Mary"])

    # Access private attribute correctly
    assert len(analyzer._known_characters) == 2
```

**Step 5: Fix `test_all_modules_import`**

Remove the `realtime_player` import (file may not exist):

```python
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
    # realtime_player removed — merged into tts_engine

    print("All modules imported successfully!")
```

**Step 6: Run all tests**

Run: `pytest tests/test_modules.py -v`
Expected: All tests PASS (except any that require actual hardware/API keys)

**Step 7: Commit**

```bash
git add tests/test_modules.py
git commit -m "fix: correct test field names and imports to match actual module interfaces"
```

---

## Task 8: Final Verification

**Step 1: Run the full test suite**

Run: `pytest tests/test_modules.py -v`
Expected: All tests PASS

**Step 2: Verify imports are clean**

Run: `python -c "from src.camera_controller import CameraRealtimeController; from src.narrator import Narrator, NarrationStyle, Narration; print('All imports OK')"`
Expected: `All imports OK`

**Step 3: Verify CLI is functional**

Run: `python -m src.main --help`
Expected: Shows `process` and `camera` commands

Run: `python -m src.main camera --help`
Expected: Shows all flags including `--mic`, `--silence-threshold`, `--cooldown`, `--list-mics`

**Step 4: List available mics (for user to identify correct device)**

Run: `python -m src.main camera --list-mics`
Expected: Table of available microphone devices

**Step 5: Final commit and tag**

```bash
git add -A
git commit -m "chore: final verification — real-time narration pipeline complete"
```

---

## How to Run (After All Tasks Complete)

```bash
# Basic usage — camera 0, default mic
python -m src.main camera --camera 0

# With specific mic (use --list-mics to find the right ID)
python -m src.main camera --camera 0 --mic 1

# Looser silence detection (better for noisy rooms)
python -m src.main camera --camera 0 --mic 1 --silence-threshold -45

# Faster narrations (every 3s instead of 5s)
python -m src.main camera --camera 0 --mic 1 --cooldown 3
```

Make sure `.env` has:
```
OPENAI_API_KEY=sk-...
AI_PROVIDER=openai
OPENAI_MODEL=gpt-4o
TTS_ENGINE=edge
TTS_VOICE=en-US-AriaNeural
```
