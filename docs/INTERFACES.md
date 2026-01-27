# Interfaces (Phase 1)

This specification defines key data interfaces for external/internal communication in Phase 1, facilitating future module replacement and Phase 2 TTS integration.

## 1. Visual Input Interface

### 1.1 File Input
- Input: Local path or URI
- Supported Formats: MP4, AVI (Priority)

### 1.2 Streaming Input (Reserved)
- `InputAdapter` should support input from external devices/network streams, uniformly producing frames and audio chunks.

## 2. Core Data Structures (Conceptual)

### 2.1 VideoFrame
- `pts_ms`: int, Frame timestamp (milliseconds)
- `image`: RGB/BGR Image Matrix (Internal passing)
- `frame_index`: int, Optional

### 2.2 AudioChunk
- `start_ms`/`end_ms`: int, Audio time range (milliseconds)
- `pcm_s16le`: bytes/ndarray, Mono PCM (Internal passing)
- `sample_rate_hz`: int

### 2.3 VisualFacts
- `pts_ms`: int
- `objects[]`:
  - `label`: string
  - `confidence`: float
  - `bbox`: `[x1,y1,x2,y2]` (Pixel coordinates)
- `characters[]`:
  - `character_id`: string
  - `display_name`: string
  - `confidence`: float
  - `bbox`: Optional (Face box)
- `scene_change`: bool

### 2.4 DialogueState
- `is_dialogue`: bool
- `confidence`: float
- `active_segment_start_ms`: int, Optional

## 3. Narration Output Interface (Standardized)

Phase 1 suggests implementing **JSON Lines** (One JSON object per line) for easy consumption by Players/TTS/Logging systems.

### 3.1 NarrationSegment (JSONL)
Fields:
- `type`: Fixed as `"narration"`
- `run_id`: string
- `timestamp_start_ms`: int
- `timestamp_end_ms`: int
- `text`: string
- `style`: string (e.g., `concise`/`detailed`/`cinematic`)
- `detail_level`: int (e.g., 1-5)
- `confidence`: float
- `metadata`: object (Optional, containing object lists, character lists, scene IDs, etc.)

Example:
```json
{"type":"narration","run_id":"run_001","timestamp_start_ms":12340,"timestamp_end_ms":12900,"text":"Joe walks quickly into the dim corridor, clutching a key tightly.","style":"concise","detail_level":2,"confidence":0.82,"metadata":{"characters":[{"id":"ch_joe","name":"Joe"}],"objects":[{"label":"key","conf":0.74}]}}
```

### 3.2 Event Output (Optional)
Besides narration, optional outputs for debugging/visualization:
- `scene_change`: Shot cut events
- `dialogue_segment`: Dialogue intervals
- `metrics`: Real-time throughput/latency

## 4. Configuration Interface

Suggestion: `config.yaml` + CLI overrides.

### 4.1 Key Configuration Items (Examples)
- `performance.max_end_to_end_latency_ms`
- `performance.target_fps`
- `vision.scene_change.threshold`
- `vision.object_detection.model_name`
- `vision.face_recognition.match_threshold`
- `audio.dialogue.vad_mode`
- `nlg.style`
- `nlg.detail_level`
- `output.format` (`jsonl`/`callback`)

