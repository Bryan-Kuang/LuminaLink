# Blind-Friendly Movie Narration System (Phase 1: Vision-to-Text) PRD

## 1. Background & Goal
To provide "dialogue-gap" visual narration for visually impaired users. The system analyzes video input in real-time, detects scene changes, identifies key visual elements and characters, and generates natural language descriptions aligned with timestamps when no dialogue is present. Phase 2 will extend this to Text-to-Speech (TTS) output.

## 2. Scope
### Phase 1 (Covered by this PRD)
- Vision-to-Text: Scene change detection, object/action extraction, character recognition, dialogue detection, narration generation, time synchronization, output interface.

### Phase 2 (Reserved)
- Text-to-Speech (TTS): Speech synthesis, audio mixing, and playback/export.

## 3. Users & Scenarios
### 3.1 Primary Users
- Visually Impaired Users (End users)
- Content Creators/Tuners (Configuring narration detail and style)
- System Integrators (Feeding video streams from external devices/players)

### 3.2 Typical Workflow
1) User provides a local video file (MP4/AVI, etc.) or external device stream.
2) System decodes frames and audio, establishing a time base.
3) Performs scene change detection, visual element recognition, and character recognition.
4) Dialogue detection determines if narration should be paused.
5) Generates narration text segments (with timestamps) within allowed time windows.
6) Outputs structured data to downstream (TTS/Player/Logs).

## 4. Functional Requirements
### 4.1 Visual Input
- Support standard video files: MP4, AVI (Priority), and future extensions (MKV, etc.).
- Universal Input Interface: Support file input and external device/network stream input (Phase 1 implements file input; stream interface reserved).

### 4.2 Scene Change Detection
- Real-time video stream analysis to detect shot cuts/scene changes.
- Output: `SceneSegment` (start/end timestamp) or `SceneChangeEvent` (timestamp).
- Metrics: Recall and precision acceptable for engineering use under common movie editing (Phase 1 focuses on configurable thresholds and visual tuning).

### 4.3 Object Detection & Action/Element Extraction
- Identify key elements using object detection: People, Objects, Significant Scene Elements.
- For "Actions", Phase 1 may use heuristics (pose/interaction/displacement) or reserve action recognition interfaces for future specialized models.
- Output unified `VisualFacts`: Includes object category, confidence, bbox/region, keyframe timestamp, etc.

### 4.4 Face Recognition & Character Database
- Establish a character database storing facial feature vectors and metadata (Name, Alias, Optional Avatar) for main characters.
- Real-time Character Recognition: Extract features from detected faces and compare with the database, returning Character ID and Name.
- Narration prefers character names (e.g., "Joe") over generic terms (e.g., "A man").

### 4.5 Dialogue Detection
- Accurately identify dialogue segments in the movie (detect human voice/speech from audio).
- Automatically pause visual narration during dialogue (visual information can still be buffered for summary after dialogue ends).
- Output: `DialogueSegment` (start/end timestamp) or real-time boolean state `is_dialogue`.

### 4.6 Narration Generation (NLG)
- Generate natural language descriptions based on `VisualFacts`, `SceneContext`, and Character Recognition results within non-dialogue windows.
- Support configuration for narration detail and style (e.g., Concise/Detailed; Objective/Cinematic).
- Phase 1 Output: Timestamped `NarrationSegment` (text + start/end timestamp).

### 4.7 Time Synchronization
- Unified System Time Base: Use Video PTS or Audio Clock as the master clock to ensure output events align with the video.
- Support output latency control and buffering strategy configuration (e.g., Max buffer time, sliding window).

### 4.8 Output Interface (Reserved for TTS)
- Standardized Data Interface: Output JSON Lines or gRPC/HTTP (Phase 1 can implement JSON Lines / Python callback).
- Minimal Fields: `timestamp_start_ms`, `timestamp_end_ms`, `text`, `confidence`, `metadata`.

### 4.9 Configuration Interface
- Provide configuration files (YAML/JSON) and runtime parameter overrides.
- Tunable items:
  - Narration detail, style
  - Scene change threshold
  - Dialogue detection sensitivity
  - Character recognition threshold
  - Performance strategy (Sampling rate, Parallelism, GPU/CPU selection)

## 5. System Architecture Requirements
### 5.1 Modular Design
Phase 1 must be split into clear modules, decoupled by data structures/interfaces:
- `InputAdapter`: Video/Audio input
- `FramePipeline`: Frame preprocessing and scheduling
- `SceneChangeDetector`
- `ObjectDetector` (PyTorch)
- `FaceDetector` + `FaceEmbedder` (PyTorch)
- `CharacterStore` (Character Database)
- `DialogueDetector`
- `NarrationGenerator` (TensorFlow)
- `Synchronizer`: Time alignment
- `OutputSink`: Output

### 5.2 Extensibility
- Modules primarily defined by interfaces and pluggable implementations (e.g., replacing different detection models).
- Support for future additions: Action recognition, OCR subtitle recognition, Emotion/Scene atmosphere, etc.

## 6. Technical Implementation Constraints
- Visual processing uses PyTorch (Inference primarily).
- Natural Language Generation uses TensorFlow (Inference primarily).
- Frame processing pipeline must be efficient (Multi-threading/Multi-processing/Async Queues).
- Must consider 1080p, ≥24fps throughput.

## 7. Quality & Performance Metrics
- Character Recognition Accuracy: ≥90% (Based on specified evaluation set).
- Narration End-to-End Latency: ≤300ms (From video frame to text output).
- Dialogue Detection False Trigger Rate: <5%.
- Processing Capability: 1080p, frame rate not lower than 24fps (Verified on target hardware configuration).

## 8. Milestones (Suggested)
- M1: Interface and skeleton running (File Input → Output Text Events).
- M2: Scene Change + Dialogue Detection + Basic Object Detection integration.
- M3: Character Database and Real-time Character Recognition integration.
- M4: NLG (TensorFlow) integration and Style/Detail configuration.
- M5: Performance stress testing and metric alignment.

## 9. Acceptance Criteria
- On a given sample video:
  - Can output timestamp-aligned narration segments;
  - Does not output visual narration during dialogue;
  - Narration includes character names;
  - Features configurable style and detail levels;
  - Output can be consumed by downstream TTS modules.

