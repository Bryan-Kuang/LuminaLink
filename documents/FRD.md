# LuminaLink摄像头实时解说系统 - 功能需求文档 (FRD)

**版本**: 1.0
**日期**: 2026-02-06
**状态**: 草案

---

## 1. 文档概述

本文档详细描述LuminaLink摄像头实时解说系统的功能需求、技术架构、接口设计和实现细节，作为开发团队的实施指南。

---

## 2. 系统架构

### 2.1 整体架构图

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

### 2.2 技术栈

| 层次 | 技术 | 版本 |
|------|------|------|
| **GUI框架** | Tkinter | Python标准库 |
| **视频处理** | OpenCV | ≥4.5.0 |
| **AI视觉** | OpenAI GPT-4 Vision | 最新API |
| **TTS引擎** | EdgeTTS | ≥6.1.0 |
| **音频处理** | sounddevice, librosa | ≥0.4.6, ≥0.10.0 |
| **并发** | threading, asyncio | Python标准库 |
| **数据处理** | NumPy, Pillow | ≥1.21.0, ≥10.0.0 |

---

## 3. 核心模块详细设计

### 3.1 VideoFrame类型定义

**文件**: `src/luminalink/types.py`

**职责**: 统一视频帧数据结构，供CameraInput和VideoProcessor共用。

**类定义**:
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

**关键点**:
- `pts_ms`: 绝对时间戳（毫秒），摄像头模式使用系统时钟，文件模式使用视频PTS
- `frame_index`: 帧序号，从0开始递增
- `image_bgr`: OpenCV标准BGR格式图像（H×W×3 NumPy数组）

---

### 3.2 CameraInput（摄像头输入）

**文件**: `src/luminalink/input/camera_input.py` (已存在，需集成)

**职责**: 从设备摄像头捕获实时视频流。

**接口**:
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

**实现细节**:
- 使用`cv2.VideoCapture(camera_index)`打开摄像头
- 设置分辨率和FPS：`cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)`
- 时间戳使用系统时钟：`pts_ms = int((time.time() - start_time) * 1000)`
- 迭代器模式：`while True: ok, frame = cap.read()`

**错误处理**:
- 摄像头打开失败：抛出`ValueError`，GUI显示错误对话框
- 读帧失败：停止迭代器，触发`StopIteration`

---

### 3.3 AudioInputStream（音频输入）

**文件**: `src/audio_input.py` (需新建)

**职责**: 从麦克风捕获音频流，喂给音频检测器。

**类定义**:
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

**依赖**:
- `sounddevice>=0.4.6`：跨平台音频捕获库
- `RealtimeAudioDetector`：已存在于`src/audio_detector.py`

**参数说明**:
- `sample_rate=22050`: 与RealtimeAudioDetector默认值一致
- `blocksize=512`: 约23ms延迟（512/22050≈0.023s）
- `channels=1`: 单声道，降低处理开销

**线程安全**:
- sounddevice内部使用独立线程调用callback
- callback中直接调用`detector.feed_audio()`是线程安全的

---

### 3.4 CameraRealtimeController（摄像头实时控制器）

**文件**: `src/camera_controller.py` (需新建)

**职责**: 编排摄像头输入、音频输入、场景分析、解说生成的完整流程。

**类定义**:
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

    # Worker threads (详见下节)
    def _camera_worker(self): ...
    def _analysis_worker(self): ...
    def _narration_worker(self): ...
```

**线程模型**:

| 线程 | 职责 | 输入 | 输出 |
|------|------|------|------|
| **Main (GUI)** | Tkinter事件循环 | 用户操作 | UI更新 |
| **CameraThread** | 捕获摄像头帧 | CameraInput.frames() | frame_queue |
| **AudioThread** | 捕获麦克风音频 | sounddevice callback | AudioDetector |
| **AnalysisThread** | 场景分析 | analysis_queue | narration_queue |
| **NarrationThread** | TTS生成和播放 | narration_queue | 扬声器输出 |

**队列通信**:
- `_frame_queue`: CameraThread → GUI显示
- `_analysis_queue`: CameraThread → AnalysisThread
- `_narration_queue`: AnalysisThread → NarrationThread

---

#### 3.4.1 Camera Worker线程

```python
def _camera_worker(self):
    """Capture frames from camera"""
    last_analysis_time = 0
    analysis_interval = self.config.video.keyframe_interval  # 默认1.0秒

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

**关键点**:
- **帧率控制**: 捕获30 FPS，但只发送1-2 FPS到分析队列
- **非阻塞**: GUI回调和队列都使用非阻塞操作，避免卡顿
- **暂停支持**: `_pause_event.wait()`阻塞线程直到恢复

---

#### 3.4.2 Analysis Worker线程

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

**关键逻辑**:
1. **间隔控制**: `narrator.should_narrate()`检查距上次解说是否>=5秒
2. **对话检测**: `audio_detector.is_current_silence()`检查当前是否静音
3. **异步AI调用**: `analyze_frame_async()`使用asyncio避免阻塞
4. **错误处理**: 捕获异常，记录日志，继续处理下一帧

---

#### 3.4.3 Narration Worker线程

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

**关键点**:
- **TTS缓存**: `tts_manager.synthesize()`内部缓存相同文本的音频
- **阻塞播放**: `audio_player.play()`阻塞直到播放完成
- **状态更新**: `narrator.record_narration()`记录时间戳，用于间隔控制

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

**线程安全关键**:
- ✅ **使用`root.after()`**: 所有UI更新必须在主线程执行
- ✅ **回调函数**: 工作线程调用回调，回调内部使用`after()`转发到主线程
- ❌ **直接操作**: 绝不在工作线程直接调用Tkinter方法

---

### 3.6 主程序入口修改

**文件**: `src/main.py` (需修改)

**修改内容**:
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
    # 现有的 LuminaLink 类逻辑
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

**向后兼容**:
- 旧命令仍然工作：`python -m src.main process --video movie.mp4`
- 新命令启动GUI：`python -m src.main camera`
- 简化命令（未来）：`python -m src.main camera` 可以缩写为 `luminalink camera`

---

## 4. 数据流详解

### 4.1 正常流程

```
1. 用户点击"Start"
   → GUI调用controller.start()

2. CameraThread启动
   → 从摄像头读取帧（30 FPS）
   → 发送到GUI显示（30 FPS）
   → 发送到分析队列（1-2 FPS）

3. AudioThread启动
   → 从麦克风读取音频（22050 Hz）
   → 喂给RealtimeAudioDetector
   → 更新静音状态

4. AnalysisThread处理帧
   → 检查解说间隔（>=5秒）
   → 检查静音状态（is_current_silence）
   → 识别角色（可选）
   → 调用GPT-4V分析场景
   → 生成解说文本
   → 放入解说队列

5. NarrationThread播放
   → 从队列取解说
   → 调用EdgeTTS生成音频
   → 播放音频
   → 通知GUI显示字幕

6. GUI更新
   → 显示视频帧
   → 显示字幕叠加
   → 更新状态栏
   → 记录日志
```

### 4.2 暂停流程

```
1. 用户按空格或点击"Pause"
   → GUI调用controller.pause()

2. Controller设置暂停事件
   → _pause_event.clear()

3. 所有工作线程暂停
   → CameraThread: _pause_event.wait() 阻塞
   → AnalysisThread: _pause_event.wait() 阻塞
   → NarrationThread: _pause_event.wait() 阻塞

4. GUI更新
   → 按钮文本变为"Resume"
   → 状态变为"Paused"

5. 用户恢复
   → GUI调用controller.resume()
   → _pause_event.set()
   → 所有线程恢复运行
```

### 4.3 停止流程

```
1. 用户按ESC或点击"Stop"
   → GUI调用controller.stop()

2. Controller设置停止事件
   → _stop_event.set()

3. 所有工作线程退出循环
   → while not self._stop_event.is_set(): break

4. 等待线程结束
   → thread.join(timeout=2.0)

5. 释放资源
   → camera_input.close()
   → audio_stream.stop()

6. GUI更新
   → 按钮状态重置
   → 状态变为"Stopped"
```

---

## 5. 接口定义

### 5.1 CameraRealtimeController接口

| 方法 | 参数 | 返回值 | 说明 |
|------|------|--------|------|
| `__init__` | camera_index, characters_config | - | 初始化控制器 |
| `start()` | - | None | 启动所有线程 |
| `pause()` | - | None | 暂停解说 |
| `resume()` | - | None | 恢复解说 |
| `stop()` | - | None | 停止并释放资源 |
| `set_on_frame_callback` | callback: Callable | None | 设置帧回调 |
| `set_on_narration_callback` | callback: Callable | None | 设置解说回调 |
| `set_on_status_callback` | callback: Callable | None | 设置状态回调 |

### 5.2 回调接口

**帧回调**:
```python
def on_frame(frame_bgr: np.ndarray) -> None:
    """Called when new frame is available (30 FPS)"""
```

**解说回调**:
```python
def on_narration(text: str) -> None:
    """Called when narration is generated"""
```

**状态回调**:
```python
def on_status(status: str) -> None:
    """Called when state changes (Running/Paused/Stopped)"""
```

---

## 6. 配置与设置

### 6.1 用户设置文件

**位置**: `~/.luminalink/settings.json`

**格式**:
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

**加载逻辑**:
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

**保存逻辑**:
```python
def save_settings(settings: dict):
    settings_file = Path.home() / ".luminalink" / "settings.json"
    settings_file.parent.mkdir(parents=True, exist_ok=True)
    with open(settings_file, 'w') as f:
        json.dump(settings, f, indent=2)
```

---

## 7. 错误处理

### 7.1 摄像头错误

| 错误 | 处理方式 |
|------|---------|
| 摄像头打开失败 | 显示错误对话框，列出可用设备，提供文件模式降级 |
| 摄像头断开 | 显示错误提示，停止捕获，支持重新连接 |
| 读帧失败 | 记录警告日志，跳过该帧，继续处理 |

**示例代码**:
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

### 7.2 API错误

| 错误 | 处理方式 |
|------|---------|
| OpenAI API失败 | 记录错误，跳过该帧，显示警告提示 |
| API限流 | 降低分析频率，显示限流提示 |
| 网络中断 | 显示网络错误，提供重试按钮 |

### 7.3 音频错误

| 错误 | 处理方式 |
|------|---------|
| 麦克风不可用 | 显示警告，禁用对话检测，继续解说 |
| TTS失败 | 使用备用引擎（gTTS → pyttsx3） |
| 音频播放失败 | 记录错误，显示字幕但不播放 |

---

## 8. 测试计划

### 8.1 单元测试

**文件**: `tests/test_camera_controller.py`

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

### 8.2 集成测试

**测试场景**:
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

### 8.3 性能测试

**测试指标**:
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

## 9. 部署与发布

### 9.1 依赖安装

**requirements.txt更新**:
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

**安装命令**:
```bash
pip install -r requirements.txt
```

### 9.2 配置向导

**首次运行**:
```bash
python -m src.main camera

# 如果没有配置API key，显示设置向导
# 1. 检测.env文件
# 2. 如果不存在，提示用户输入API key
# 3. 保存到.env文件
```

### 9.3 打包分发

**使用PyInstaller打包**:
```bash
pyinstaller --onefile --windowed \
  --name LuminaLink \
  --icon icon.icns \
  src/main.py
```

---

## 10. 维护与扩展

### 10.1 日志记录

**日志级别**:
- `DEBUG`: 详细的调试信息（帧处理、队列状态）
- `INFO`: 关键事件（启动、停止、解说生成）
- `WARNING`: 非关键错误（队列满、帧跳过）
- `ERROR`: 严重错误（API失败、设备错误）

**日志文件**:
```
~/.luminalink/logs/
├── luminalink.log (主日志)
├── camera.log (摄像头日志)
└── api.log (API调用日志)
```

### 10.2 性能监控

**性能指标收集**:
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

## 11. 附录

### 11.1 完整文件清单

**需创建**:
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

**需修改**:
1. `src/main.py` - 添加camera子命令
2. `src/video_processor.py` - 导入统一VideoFrame
3. `requirements.txt` - 添加sounddevice和Pillow

**需复用**:
1. `src/luminalink/input/camera_input.py` - 已实现
2. `src/realtime_player.py` - 参考多线程架构
3. `src/audio_detector.py` - RealtimeAudioDetector
4. 所有其他核心模块（scene_analyzer, narrator, tts_engine等）

### 11.2 依赖关系图

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

**文档维护**:
- **创建者**: Claude (LuminaLink Planning Agent)
- **最后更新**: 2026-02-06
- **审阅状态**: 待审阅
- **版本历史**:
  - v1.0 (2026-02-06): 初始版本
