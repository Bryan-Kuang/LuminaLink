# LuminaLink Camera Real-time Narration System - Functional Requirements Document (FRD)

**Version**: 1.0
**Date**: 2026-02-06
**Status**: Draft

---

## 1. Document Overview

This document details the functional requirements, technical architecture, interface design, and implementation details of the LuminaLink Camera Real-time Narration System, serving as an implementation guide for the development team.

---

## 2. System Architecture

### 2.1 Overall Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    GUI Layer (Tkinter)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Video Preview │  │   Controls   │  │   Settings   │      │
│  │   Canvas     │  │Start/Stop/Cfg│  │    Panel     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└────────────────────┬────────────────────────────────────────┘
                     │ (Thread-safe Queue Communication)
┌────────────────────▼────────────────────────────────────────┐
│              Application Controller                          │
│           (CameraRealtimeController)                        │
└───┬─────────────┬──────────────┬─────────────┬─────────────┘
    │             │              │             │
┌───▼───┐  ┌──────▼──────┐  ┌───▼────┐  ┌────▼──────┐
│Camera │  │Scene Analysis│  │Narration│ │AudioInput │
│Input  │  │   Thread     │  │ Thread  │ │  Thread   │
│Thread │  │              │  │         │ │           │
└───┬───┘  └──────┬──────┘  └───┬────┘  └────┬──────┘
    │             │              │             │
    │      ┌──────▼──────┐       │      ┌─────▼──────┐
    │      │SceneAnalyzer│       │      │AudioDetect │
    │      │  (GPT-4V)   │       │      │  (RMS/VAD) │
    │      └─────────────┘       │      └────────────┘
    │                            │
    ├────────────────────────────┤
    │                            │
┌───▼──────────┐          ┌─────▼──────┐
│CharacterRecog│          │  Narrator  │
│  (Optional)  │          │            │
└──────────────┘          └─────┬──────┘
                                │
                         ┌──────▼──────┐
                         │ TTS Engine  │
                         │  (EdgeTTS)  │
                         └─────────────┘
```

### 2.2 Tech Stack

| Layer                | Technology           | Version                 |
| -------------------- | -------------------- | ----------------------- |
| **GUI Framework**    | Tkinter              | Python Standard Library |
| **Video Processing** | OpenCV               | ≥4.5.0                  |
| **AI Vision**        | OpenAI GPT-4 Vision  | Latest API              |
| **TTS Engine**       | EdgeTTS              | ≥6.1.0                  |
| **Audio Processing** | sounddevice, librosa | ≥0.4.6, ≥0.10.0         |
| **Concurrency**      | threading, asyncio   | Python Standard Library |
| **Data Processing**  | NumPy, Pillow        | ≥1.21.0, ≥10.0.0        |

---

## 3. Detailed Module Design

### 3.1 VideoFrame Type Definition

**File**: `src/luminalink/types.py`

**Responsibility**: Unify video frame data structures for shared use by CameraInput and VideoProcessor.

**Class Definition**:

```python
from dataclasses import dataclass
import numpy as np

@dataclass
class VideoFrame:
    """Unified video frame for camera and file modes"""

    pts_ms: int              # Presentation timestamp in milliseconds
    frame_index: int         # Sequential frame number (0-indexed)
    image_bgr: np.ndarray   # BGR image data (OpenCV format)

    @property
    def timestamp(self) -> float:
        """Timestamp in seconds"""
        return self.pts_ms / 1000.0

    @property
    def frame(self) -> np.ndarray:
        """Alias for image_bgr (compatibility with existing code)"""
        return self.image_bgr
```

**Key Points**:

- `pts_ms`: Absolute timestamp (milliseconds). Camera mode uses the system clock; file mode uses video PTS.
- `frame_index`: Frame sequence number, incrementing from 0.
- `image_bgr`: OpenCV standard BGR format image (H×W×3 NumPy array).

---

### 3.2 CameraInput (Camera Input)

**File**: `src/luminalink/input/camera_input.py` (Existing, integration required)

**Responsibility**: Capture real-time video stream from the device camera.

**Interface**:

```python
class CameraInput:
    def __init__(self, camera_index: int = 0,
                 width: int = 1280, height: int = 720, fps: int = 30)

    def open(self) -> None
        """Open camera device and configure parameters"""

    def frames(self) -> Iterator[VideoFrame]
        """Yield frames with system clock timestamps"""

    def close(self) -> None
        """Release camera resources"""
```

**Implementation Details**:

- Use `cv2.VideoCapture(camera_index)` to open the camera.
- Set resolution and FPS: `cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)`.
- Timestamp uses the system clock: `pts_ms = int((time.time() - start_time) * 1000)`.
- Iterator pattern: `while True: ok, frame = cap.read()`.

**Error Handling**:

- Camera open failure: Raise `ValueError`, GUI displays error dialog.
- Frame read failure: Stop the iterator, triggering `StopIteration`.

---

### 3.3 AudioInputStream (Audio Input)

**File**: `src/audio_input.py` (Creation required)

**Responsibility**: Capture audio stream from the microphone and feed it to the audio detector.

**Class Definition**:

```python
import sounddevice as sd
import numpy as np
from .audio_detector import RealtimeAudioDetector

class AudioInputStream:
    """Captures microphone audio and feeds to RealtimeAudioDetector"""

    def __init__(self, detector: RealtimeAudioDetector,
                 sample_rate: int = 22050,
                 blocksize: int = 512):
        self.detector = detector
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.stream: sd.InputStream | None = None
        self._running = False

    def start(self):
        """Start audio capture"""
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio status: {status}")
            # indata shape: (frames, channels)
            audio_mono = indata[:, 0] if indata.ndim > 1 else indata
            self.detector.feed_audio(audio_mono)

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,  # Mono
            callback=audio_callback,
            blocksize=self.blocksize
        )
        self.stream.start()
        self._running = True

    def stop(self):
        """Stop audio capture"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self._running = False

    def is_running(self) -> bool:
        return self._running
```

**Dependencies**:

- `sounddevice>=0.4.6`: Cross-platform audio capture library.
- `RealtimeAudioDetector`: Existing in `src/audio_detector.py`.

**Parameter Description**:

- `sample_rate=22050`: Consistent with the default value of RealtimeAudioDetector.
- `blocksize=512`: Approx 23ms latency (512/22050≈0.023s).
- `channels=1`: Mono, reducing processing overhead.

**Thread Safety**:

- sounddevice internally uses a standalone thread to call the callback.
- Directly calling `detector.feed_audio()` in the callback is thread-safe.

---

### 3.4 CameraRealtimeController (Camera Real-time Controller)

**File**: `src/camera_controller.py` (Creation required)

**Responsibility**: Orchestrates the complete flow of camera input, audio input, scene analysis, and narration generation.

**Class Definition**:

```python
class CameraRealtimeController:
    """Orchestrates real-time camera narration pipeline"""

    def __init__(self, camera_index: int = 0,
                 characters_config: Optional[str] = None):
        # Components
        self.camera_input = CameraInput(camera_index)
        self.audio_detector = RealtimeAudioDetector()
        self.audio_stream = AudioInputStream(self.audio_detector)
        self.scene_analyzer = SceneAnalyzer()
        self.narrator = Narrator()
        self.tts_manager = TTSManager()
        self.character_recognizer = CharacterRecognizer()

        if characters_config:
            self.character_recognizer.load_config(characters_config)

        # State
        self.state = PlaybackState()
        self._stop_event = Event()
        self._pause_event = Event()
        self._pause_event.set()  # Not paused initially

        # Queues
        self._frame_queue = Queue(maxsize=5)
        self._analysis_queue = Queue(maxsize=10)
        self._narration_queue = Queue(maxsize=10)

        # Callbacks for GUI
        self._on_frame_callback: Optional[Callable] = None
        self._on_narration_callback: Optional[Callable] = None
        self._on_status_callback: Optional[Callable] = None

        # Threads
        self._threads: List[Thread] = []

    def start(self):
        """Start all worker threads"""
        # Open devices
        self.camera_input.open()
        self.audio_stream.start()

        # Start threads
        threads = [
            Thread(target=self._camera_worker, daemon=True, name="CameraThread"),
            Thread(target=self._analysis_worker, daemon=True, name="AnalysisThread"),
            Thread(target=self._narration_worker, daemon=True, name="NarrationThread")
        ]

        for t in threads:
            t.start()
            self._threads.append(t)

        self.state.is_playing = True
        self._notify_status("Running")

    def pause(self):
        """Pause narration"""
        self._pause_event.clear()
        self.state.is_paused = True
        self._notify_status("Paused")

    def resume(self):
        """Resume narration"""
        self._pause_event.set()
        self.state.is_paused = False
        self._notify_status("Running")

    def stop(self):
        """Stop all threads and release resources"""
        self._stop_event.set()

        # Wait for threads
        for t in self._threads:
            t.join(timeout=2.0)

        # Cleanup
        self.camera_input.close()
        self.audio_stream.stop()

        self.state.is_playing = False
        self._notify_status("Stopped")

    # Worker threads (Details in next section)
    def _camera_worker(self): ...
    def _analysis_worker(self): ...
    def _narration_worker(self): ...
```

**Threading Model**:

| Thread              | Responsibility              | Input                | Output          |
| ------------------- | --------------------------- | -------------------- | --------------- |
| **Main (GUI)**      | Tkinter event loop          | User actions         | UI updates      |
| **CameraThread**    | Capture camera frames       | CameraInput.frames() | frame_queue     |
| **AudioThread**     | Capture microphone audio    | sounddevice callback | AudioDetector   |
| **AnalysisThread**  | Scene analysis              | analysis_queue       | narration_queue |
| **NarrationThread** | TTS generation and playback | narration_queue      | Speaker output  |

**Queue Communication**:

- `_frame_queue`: CameraThread → GUI display
- `_analysis_queue`: CameraThread → AnalysisThread
- `_narration_queue`: AnalysisThread → NarrationThread

---

#### 3.4.1 Camera Worker Thread

```python
def _camera_worker(self):
    """Capture frames from camera"""
    last_analysis_time = 0
    analysis_interval = self.config.video.keyframe_interval  # Default 1.0 second

    for video_frame in self.camera_input.frames():
        if self._stop_event.is_set():
            break

        # Wait if paused
        self._pause_event.wait()

        # Send to GUI for display (non-blocking)
        if self._on_frame_callback:
            try:
                self._on_frame_callback(video_frame.image_bgr)
            except Exception as e:
                logger.error(f"Frame callback error: {e}")

        # Send to analysis queue at intervals
        current_time = video_frame.timestamp
        if current_time - last_analysis_time >= analysis_interval:
            try:
                self._analysis_queue.put_nowait(video_frame)
                last_analysis_time = current_time
            except queue.Full:
                logger.warning("Analysis queue full, skipping frame")
```

**Key Points**:

- **Frame Rate Control**: Capture at 30 FPS, but only send 1-2 FPS to the analysis queue.
- **Non-blocking**: Both GUI callbacks and queues use non-blocking operations to avoid UI lag.
- **Pause Support**: `_pause_event.wait()` blocks the thread until resumed.

---

#### 3.4.2 Analysis Worker Thread

```python
def _analysis_worker(self):
    """Scene analysis and narration generation"""
    while not self._stop_event.is_set():
        try:
            video_frame = self._analysis_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # Wait if paused
        self._pause_event.wait()

        # Check if should narrate (minimum interval)
        if not self.narrator.should_narrate(video_frame.timestamp):
            continue

        # Check if silence (no dialogue)
        if not self.audio_detector.is_current_silence():
            logger.debug(f"Skipping narration: dialogue detected")
            continue

        try:
            # Recognize characters (optional)
            characters = []
            if self.character_recognizer.is_enabled():
                characters = self.character_recognizer.get_characters_in_frame(
                    video_frame.image_bgr,
                    timestamp=video_frame.timestamp
                )

            # Analyze scene (async AI call)
            analysis = await self.scene_analyzer.analyze_frame_async(
                video_frame.image_bgr,
                characters_in_frame=characters,
                timestamp=video_frame.timestamp
            )

            # Generate narration
            narration = self.narrator.generate_narration(
                analysis,
                slot=(video_frame.timestamp, video_frame.timestamp + 5.0),
                characters_in_frame=characters
            )

            if narration:
                self._narration_queue.put(narration)

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            # Continue processing next frame
```

**Key Logic**:

1. **Interval Control**: `narrator.should_narrate()` checks if the time since the last narration is >= 5 seconds.
2. **Dialogue Detection**: `audio_detector.is_current_silence()` checks if it's currently silent.
3. **Async AI Call**: `analyze_frame_async()` uses asyncio to avoid blocking.
4. **Error Handling**: Catch exceptions, log errors, and continue processing the next frame.

---

#### 3.4.3 Narration Worker Thread

```python
def _narration_worker(self):
    """TTS generation and playback"""
    while not self._stop_event.is_set():
        try:
            narration = self._narration_queue.get(timeout=1.0)
        except queue.Empty:
            continue

        # Wait if paused
        self._pause_event.wait()

        try:
            # Notify GUI
            if self._on_narration_callback:
                self._on_narration_callback(narration.text)

            # Generate TTS audio
            audio_file = self.tts_manager.synthesize(narration.text)

            # Play audio (blocking)
            self.audio_player.play(audio_file)

            # Update narrator state
            self.narrator.record_narration(narration)

        except Exception as e:
            logger.error(f"Narration playback error: {e}")
```

**Key Points**:

- **TTS Caching**: `tts_manager.synthesize()` internally caches audio for the same text.
- **Blocking Playback**: `audio_player.play()` blocks until playback is complete.
- **Status Update**: `narrator.record_narration()` records the timestamp for interval control.

---

### 3.5 CameraApp（GUI应用）

**文件**: `src/gui/camera_app.py` (需新建)

**职责**: 提供用户界面，显示视频、控制按钮、状态信息。

**类结构**:

```python
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class CameraApp:
    """Main GUI application for camera narration"""

    def __init__(self, camera_index: int = 0,
                 characters_config: Optional[str] = None):
        # Controller
        self.controller = CameraRealtimeController(
            camera_index, characters_config
        )

        # Tkinter setup
        self.root = tk.Tk()
        self.root.title("LuminaLink Camera Narration")
        self.root.geometry("800x700")

        # UI components
        self._setup_ui()
        self._bind_callbacks()
        self._bind_shortcuts()

    def _setup_ui(self):
        """Build UI components"""
        # Video preview canvas
        self.video_canvas = tk.Canvas(
            self.root, width=640, height=480, bg='black'
        )
        self.video_canvas.pack(pady=10)

        # Control buttons frame
        control_frame = ttk.Frame(self.root)
        control_frame.pack(pady=5)

        self.start_btn = ttk.Button(
            control_frame, text="▶ Start", command=self.start
        )
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.pause_btn = ttk.Button(
            control_frame, text="⏸ Pause", command=self.pause,
            state=tk.DISABLED
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(
            control_frame, text="⏹ Stop", command=self.stop,
            state=tk.DISABLED
        )
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Status label
        self.status_label = ttk.Label(
            self.root, text="Status: Ready", font=('Arial', 12)
        )
        self.status_label.pack(pady=5)

        # Settings panel (collapsible)
        self._create_settings_panel()

        # Narration log (collapsible)
        self._create_narration_log()

    def _create_settings_panel(self):
        """Create settings panel"""
        settings_frame = ttk.LabelFrame(self.root, text="Settings")
        settings_frame.pack(pady=5, padx=10, fill=tk.X)

        # TTS voice selection
        ttk.Label(settings_frame, text="TTS Voice:").grid(
            row=0, column=0, padx=5, pady=5
        )
        self.voice_var = tk.StringVar(value="en-US-AriaNeural")
        voice_combo = ttk.Combobox(
            settings_frame, textvariable=self.voice_var,
            values=["en-US-AriaNeural", "en-US-GuyNeural",
                    "en-GB-SoniaNeural"],
            state="readonly"
        )
        voice_combo.grid(row=0, column=1, padx=5, pady=5)

        # More settings...

    def _create_narration_log(self):
        """Create narration history log"""
        log_frame = ttk.LabelFrame(self.root, text="Narration Log")
        log_frame.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        self.log_text = tk.Text(
            log_frame, height=10, width=80, state=tk.DISABLED,
            font=('Arial', 10)
        )
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            log_frame, command=self.log_text.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.config(yscrollcommand=scrollbar.set)

    def _bind_callbacks(self):
        """Connect controller callbacks to GUI"""
        self.controller._on_frame_callback = self.update_video_frame
        self.controller._on_narration_callback = self.display_narration
        self.controller._on_status_callback = self.update_status

    def _bind_shortcuts(self):
        """Keyboard shortcuts"""
        self.root.bind("<space>", lambda e: self.pause())
        self.root.bind("<Escape>", lambda e: self.stop())

    # Event handlers
    def start(self):
        """Start narration"""
        self.controller.start()
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)

    def pause(self):
        """Toggle pause"""
        if self.controller.state.is_paused:
            self.controller.resume()
            self.pause_btn.config(text="⏸ Pause")
        else:
            self.controller.pause()
            self.pause_btn.config(text="▶ Resume")

    def stop(self):
        """Stop narration"""
        self.controller.stop()
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)

    def update_video_frame(self, frame_bgr: np.ndarray):
        """Update video preview (called from camera thread)"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Resize to canvas size
        frame_resized = cv2.resize(frame_rgb, (640, 480))

        # Convert to PIL Image
        image = Image.fromarray(frame_resized)
        photo = ImageTk.PhotoImage(image=image)

        # Schedule UI update in main thread
        self.root.after(0, self._update_canvas, photo)

    def _update_canvas(self, photo):
        """Actually update canvas (main thread only)"""
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self.video_canvas.image = photo  # Keep reference

    def display_narration(self, text: str):
        """Display narration text (subtitle + log)"""
        # Add subtitle overlay
        self.root.after(0, self._add_subtitle, text)

        # Add to log
        self.root.after(0, self._add_to_log, text)

    def _add_subtitle(self, text: str):
        """Draw subtitle on canvas"""
        # Draw semi-transparent background
        self.video_canvas.create_rectangle(
            0, 430, 640, 480, fill='black', stipple='gray50'
        )

        # Draw text
        self.video_canvas.create_text(
            320, 455, text=text, fill='white',
            font=('Arial', 14, 'bold'), width=600, anchor=tk.CENTER
        )

        # Auto-clear after 3 seconds
        self.root.after(3000, self._clear_subtitle)

    def _clear_subtitle(self):
        """Clear subtitle overlay"""
        # Redraw canvas (subtitle will be removed)
        pass

    def _add_to_log(self, text: str):
        """Add narration to log"""
        import time
        timestamp = time.strftime("%H:%M:%S")

        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{timestamp} - {text}\n")
        self.log_text.see(tk.END)  # Auto-scroll
        self.log_text.config(state=tk.DISABLED)

    def update_status(self, status: str):
        """Update status label"""
        self.status_label.config(text=f"Status: {status}")

    def run(self):
        """Start GUI event loop"""
        self.root.mainloop()
```

**Thread Safety Keys**:

- ✅ **Use `root.after()`**: All UI updates must be executed in the main thread.
- ✅ **Callback Functions**: Worker threads call callbacks, which use `after()` internally to forward to the main thread.
- ❌ **Direct Operations**: Never call Tkinter methods directly from worker threads.

---

### 3.6 Main Program Entry Update

**File**: `src/main.py` (requires modification)

**Updates**:

```python
import click

@click.group()
def cli():
    """LuminaLink - AI Movie Audio Description"""
    pass

@cli.command()
@click.option("--video", "-v", required=True, type=str, help="Video file path")
@click.option("--characters", "-c", type=str, help="Characters config JSON")
@click.option("--preview", is_flag=True, help="Show preview window")
@click.option("--output", "-o", type=str, help="Output subtitle file (SRT)")
def process(video, characters, preview, output):
    """Process video file (existing functionality)"""
    # Existing LuminaLink class logic
    app = LuminaLink(
        video_path=video,
        characters_config=characters,
        preview=preview,
        output_subtitles=output
    )
    asyncio.run(app.run())

@cli.command()
@click.option("--camera", "-c", default=0, type=int, help="Camera device index")
@click.option("--characters", type=str, help="Characters config JSON")
def camera(camera, characters):
    """Real-time camera narration mode"""
    from .gui.camera_app import CameraApp

    app = CameraApp(
        camera_index=camera,
        characters_config=characters
    )
    app.run()

if __name__ == "__main__":
    cli()
```

**Backward Compatibility**:

- Old command still works: `python -m src.main process --video movie.mp4`
- New command starts GUI: `python -m src.main camera`
- Simplified command (future): `python -m src.main camera` can be aliased to `luminalink camera`

---

## 4. Data Flow Details

### 4.1 Normal Flow

```
1. User clicks "Start"
   → GUI calls controller.start()

2. CameraThread starts
   → Reads frames from camera (30 FPS)
   → Sends to GUI for display (30 FPS)
   → Sends to analysis queue (1-2 FPS)

3. AudioThread starts
   → Reads audio from microphone (22050 Hz)
   → Feeds to RealtimeAudioDetector
   → Updates silence state

4. AnalysisThread processes frames
   → Checks narration interval (>=5 seconds)
   → Checks silence status (is_current_silence)
   → Recognizes characters (optional)
   → Calls GPT-4V to analyze the scene
   → Generates narration text
   → Places it in narration queue

5. NarrationThread plays
   → Pushes narration from queue
   → Calls EdgeTTS to generate audio
   → Plays audio
   → Notifies GUI to display subtitles

6. GUI Updates
   → Displays video frames
   → Displays subtitle overlays
   → Updates status bar
   → Records logs
```

### 4.2 Pause Flow

```
1. User presses space or clicks "Pause"
   → GUI calls controller.pause()

2. Controller sets pause event
   → _pause_event.clear()

3. All worker threads pause
   → CameraThread: _pause_event.wait() blocked
   → AnalysisThread: _pause_event.wait() blocked
   → NarrationThread: _pause_event.wait() blocked

4. GUI Updates
   → Button text changes to "Resume"
   → Status changes to "Paused"

5. User resumes
   → GUI calls controller.resume()
   → _pause_event.set()
   → All threads resume running
```

### 4.3 Stop Flow

```
1. User presses ESC or clicks "Stop"
   → GUI calls controller.stop()

2. Controller sets stop event
   → _stop_event.set()

3. All worker threads exit loops
   → while not self._stop_event.is_set(): break

4. Wait for threads to end
   → thread.join(timeout=2.0)

5. Release resources
   → camera_input.close()
   → audio_stream.stop()

6. GUI Updates
   → Button states reset
   → Status changes to "Stopped"
```

---

## 5. Interface Definitions

### 5.1 CameraRealtimeController Interface

| Method                      | Parameters                      | Return | Description                  |
| --------------------------- | ------------------------------- | ------ | ---------------------------- |
| `__init__`                  | camera_index, characters_config | -      | Initializes the controller   |
| `start()`                   | -                               | None   | Starts all threads           |
| `pause()`                   | -                               | None   | Pauses narration             |
| `resume()`                  | -                               | None   | Resumes narration            |
| `stop()`                    | -                               | None   | Stops and releases resources |
| `set_on_frame_callback`     | callback: Callable              | None   | Sets frame callback          |
| `set_on_narration_callback` | callback: Callable              | None   | Sets narration callback      |
| `set_on_status_callback`    | callback: Callable              | None   | Sets status callback         |

### 5.2 Callback Interfaces

**Frame Callback**:

```python
def on_frame(frame_bgr: np.ndarray) -> None:
    """Called when new frame is available (30 FPS)"""
```

**Narration Callback**:

```python
def on_narration(text: str) -> None:
    """Called when narration is generated"""
```

**Status Callback**:

```python
def on_status(status: str) -> None:
    """Called when state changes (Running/Paused/Stopped)"""
```

---

## 6. Configuration and Settings

### 6.1 User Settings File

**Location**: `~/.luminalink/settings.json`

**Format**:

```json
{
  "camera_index": 0,
  "tts_voice": "en-US-AriaNeural",
  "tts_speed": 1.0,
  "narration_interval": 5,
  "narration_style": "CONCISE",
  "show_subtitles": true,
  "audio_threshold_db": -40,
  "video_width": 1280,
  "video_height": 720,
  "analysis_interval": 1.0
}
```

**Loading Logic**:

```python
import json
from pathlib import Path

settings_file = Path.home() / ".luminalink" / "settings.json"

if settings_file.exists():
    with open(settings_file) as f:
        settings = json.load(f)
else:
    settings = default_settings()
```

**Saving Logic**:

```python
def save_settings(settings: dict):
    settings_file = Path.home() / ".luminalink" / "settings.json"
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
```

---

## 7. Error Handling

### 7.1 Camera Errors

| Error                | Handling Method                                                       |
| -------------------- | --------------------------------------------------------------------- |
| Camera fails to open | Show error dialog, list available devices, provide file mode fallback |
| Camera disconnected  | Show error notification, stop capture, support reconnection           |
| Frame read failure   | Log warning, skip frame, continue processing                          |

**Example Code**:

```python
try:
    self.camera_input.open()
except ValueError as e:
    messagebox.showerror(
        "Camera Error",
        f"Failed to open camera: {e}\n\n"
        "Please check:\n"
        "1. Camera is connected\n"
        "2. Camera is not used by other apps\n"
        "3. System permissions granted"
    )
    return
```

### 7.2 API Errors

| Error                | Handling Method                                         |
| -------------------- | ------------------------------------------------------- |
| OpenAI API failure   | Log error, skip frame, show warning                     |
| API rate limit       | Reduce analysis frequency, show rate limit notification |
| Network interruption | Show network error, provide retry button                |

### 7.3 Audio Errors

| Error                  | Handling Method                                              |
| ---------------------- | ------------------------------------------------------------ |
| Microphone unavailable | Show warning, disable dialogue detection, continue narration |
| TTS failure            | Use backup engine (gTTS → pyttsx3)                           |
| Audio playback failure | Log error, show subtitles but do not play audio              |

---

## 8. Test Plan

### 8.1 Unit Tests

**File**: `tests/test_camera_controller.py`

```python
def test_controller_initialization():
    """Test controller can be initialized"""
    controller = CameraRealtimeController(camera_index=0)
    assert controller.camera_input is not None
    assert controller.audio_detector is not None

def test_controller_start_stop():
    """Test controller can start and stop"""
    controller = CameraRealtimeController(camera_index=0)
    controller.start()
    assert controller.state.is_playing == True

    time.sleep(2)

    controller.stop()
    assert controller.state.is_playing == False

def test_callbacks():
    """Test callbacks are called"""
    controller = CameraRealtimeController(camera_index=0)

    frames_received = []
    def on_frame(frame):
        frames_received.append(frame)

    controller.set_on_frame_callback(on_frame)
    controller.start()

    time.sleep(5)
    controller.stop()

    assert len(frames_received) > 0
```

### 8.2 Integration Tests

**Test Scenarios**:

```python
def test_end_to_end_narration():
    """Test complete pipeline from camera to narration"""
    controller = CameraRealtimeController(camera_index=0)

    narrations = []
    def on_narration(text):
        narrations.append(text)

    controller.set_on_narration_callback(on_narration)
    controller.start()

    # Run for 30 seconds
    time.sleep(30)

    controller.stop()

    # Verify narrations were generated
    assert len(narrations) > 0

    # Verify narration quality (manual review needed)
    for narration in narrations:
        print(f"Narration: {narration}")
```

### 8.3 Performance Tests

**Test Metrics**:

```python
def test_latency():
    """Measure end-to-end latency"""
    controller = CameraRealtimeController(camera_index=0)

    latencies = []

    def measure_latency():
        start_time = time.time()
        # Simulate frame → narration
        # ...
        end_time = time.time()
        latencies.append(end_time - start_time)

    # Run test
    # ...

    avg_latency = sum(latencies) / len(latencies)
    p95_latency = sorted(latencies)[int(len(latencies) * 0.95)]

    assert avg_latency < 0.5  # <500ms
    assert p95_latency < 0.7  # <700ms (95th percentile)
```

---

## 9. Deployment and Release

### 9.1 Dependency Installation

**requirements.txt Update**:

```
# Existing dependencies
opencv-python>=4.5.0
numpy>=1.21.0
librosa>=0.10.0
# ... existing ...

# New dependencies for camera mode
sounddevice>=0.4.6
Pillow>=10.0.0
```

**Installation Command**:

```bash
pip install -r requirements.txt
```

### 9.2 Configuration Wizard

**First Run**:

```bash
python -m src.main camera

# If the API key is not configured, show the setup wizard
# 1. Detect .env file
# 2. If it does not exist, prompt the user to enter the API key
# 3. Save to .env file
```

### 9.3 Packaging and Distribution

**Packaging with PyInstaller**:

```bash
pyinstaller --onefile --windowed \
  --name LuminaLink \
  --icon icon.icns \
  src/main.py
```

---

## 10. Maintenance and Extension

### 10.1 Logging

**Log Levels**:

- `DEBUG`: Detailed debugging information (frame processing, queue status)
- `INFO`: Critical events (start, stop, narration generation)
- `WARNING`: Non-critical errors (queue full, frame skip)
- `ERROR`: Severe errors (API failure, device error)

**Log Files**:

```
~/.luminalink/logs/
├── luminalink.log (Main log)
├── camera.log (Camera log)
└── api.log (API call log)
```

### 10.2 Performance Monitoring

**Performance Metrics Collection**:

```python
class PerformanceMonitor:
    def __init__(self):
        self.frame_times = []
        self.api_times = []
        self.tts_times = []

    def record_frame_time(self, duration):
        self.frame_times.append(duration)

    def get_stats(self):
        return {
            "avg_frame_time": np.mean(self.frame_times),
            "p95_frame_time": np.percentile(self.frame_times, 95),
            # ...
        }
```

---

## 11. Appendix

### 11.1 Complete File List

**To be created**:

1. `src/luminalink/__init__.py`
2. `src/luminalink/types.py`
3. `src/luminalink/input/__init__.py`
4. `src/audio_input.py`
5. `src/camera_controller.py`
6. `src/gui/__init__.py`
7. `src/gui/camera_app.py`
8. `tests/test_audio_input.py`
9. `tests/test_camera_controller.py`
10. `tests/test_gui.py`

**To be modified**:

1. `src/main.py` - Add `camera` subcommand
2. `src/video_processor.py` - Import unified `VideoFrame`
3. `requirements.txt` - Add `sounddevice` and `Pillow`

**To be reused**:

1. `src/luminalink/input/camera_input.py` - Already implemented
2. `src/realtime_player.py` - Reference multi-threaded architecture
3. `src/audio_detector.py` - `RealtimeAudioDetector`
4. All other core modules (`scene_analyzer`, `narrator`, `tts_engine`, etc.)

### 11.2 Dependency Graph

```
camera_app.py
    ├─ camera_controller.py
    │   ├─ camera_input.py (luminalink/input)
    │   ├─ audio_input.py
    │   │   └─ audio_detector.py (RealtimeAudioDetector)
    │   ├─ scene_analyzer.py
    │   ├─ narrator.py
    │   ├─ tts_engine.py
    │   └─ character_recognizer.py (optional)
    └─ tkinter (GUI components)
```

---

**Document Maintenance**:

- **Creator**: Claude (LuminaLink Planning Agent)
- **Last Updated**: 2026-02-06
- **Review Status**: Pending Review
- **Version History**:
  - v1.0 (2026-02-06): Initial Version
