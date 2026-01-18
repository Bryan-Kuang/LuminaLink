# 接口规范（一期）

本规范定义一期工程对外/对内的关键数据接口，便于后续替换模块与接入二期 TTS。

## 1. 视觉输入接口

### 1.1 文件输入

- 输入：本地路径或 URI
- 支持格式：MP4、AVI（优先）

### 1.2 流式输入（预留）

- `InputAdapter` 应支持从外部设备/网络流接入，统一产出帧与音频 chunk。

## 2. 核心数据结构（概念）

### 2.1 VideoFrame

- `pts_ms`：int，帧时间戳（毫秒）
- `image`：RGB/BGR 图像矩阵（内部传递）
- `frame_index`：int，可选

### 2.2 AudioChunk

- `start_ms`/`end_ms`：int，音频时间范围（毫秒）
- `pcm_s16le`：bytes/ndarray，单声道 PCM（内部传递）
- `sample_rate_hz`：int

### 2.3 VisualFacts

- `pts_ms`：int
- `objects[]`：
  - `label`：string
  - `confidence`：float
  - `bbox`：`[x1,y1,x2,y2]`（像素坐标）
- `characters[]`：
  - `character_id`：string
  - `display_name`：string
  - `confidence`：float
  - `bbox`：可选（人脸框）
- `scene_change`：bool

### 2.4 DialogueState

- `is_dialogue`：bool
- `confidence`：float
- `active_segment_start_ms`：int，可选

## 3. 解说输出接口（标准化）

一期建议实现 **JSON Lines**（每行一个 JSON 对象），便于被播放器/TTS/日志系统消费。

### 3.1 NarrationSegment（JSONL）

字段：

- `type`：固定为 `"narration"`
- `run_id`：string
- `timestamp_start_ms`：int
- `timestamp_end_ms`：int
- `text`：string
- `style`：string（例如：`concise`/`detailed`/`cinematic`）
- `detail_level`：int（例如：1-5）
- `confidence`：float
- `metadata`：object（可选，包含对象列表、角色列表、场景ID等）

示例：

```json
{
  "type": "narration",
  "run_id": "run_001",
  "timestamp_start_ms": 12340,
  "timestamp_end_ms": 12900,
  "text": "乔快步走进昏暗的走廊，手里紧握着钥匙。",
  "style": "concise",
  "detail_level": 2,
  "confidence": 0.82,
  "metadata": {
    "characters": [{ "id": "ch_joe", "name": "乔" }],
    "objects": [{ "label": "key", "conf": 0.74 }]
  }
}
```

### 3.2 事件输出（可选）

除解说外，可选输出用于调试/可视化的事件：

- `scene_change`：镜头切换事件
- `dialogue_segment`：对话区间
- `metrics`：实时吞吐/延迟

## 4. 配置接口

建议：`config.yaml` + CLI 覆盖。

### 4.1 关键配置项（示例）

- `performance.max_end_to_end_latency_ms`
- `performance.target_fps`
- `vision.scene_change.threshold`
- `vision.object_detection.model_name`
- `vision.face_recognition.match_threshold`
- `audio.dialogue.vad_mode`
- `nlg.style`
- `nlg.detail_level`
- `output.format`（`jsonl`/`callback`）
