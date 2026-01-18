# 盲人电影解说系统（一期）ERD

本ERD用于描述一期工程中“角色库、处理运行、时间片段产物”等核心数据结构。存储可使用 SQLite/PostgreSQL；一期可先用 SQLite 方便本地开发。

## 1. Mermaid ER 图

```mermaid
erDiagram
  MOVIE {
    string movie_id PK
    string title
    int duration_ms
    string source_uri
    datetime created_at
  }

  PROCESSING_RUN {
    string run_id PK
    string movie_id FK
    string config_preset_id FK
    string device
    string notes
    datetime started_at
    datetime finished_at
  }

  CONFIG_PRESET {
    string config_preset_id PK
    string name
    string json_blob
    datetime created_at
  }

  CHARACTER {
    string character_id PK
    string display_name
    string aliases_json
    datetime created_at
  }

  FACE_EMBEDDING {
    string embedding_id PK
    string character_id FK
    string model_name
    string vector_blob
    float l2_norm
    string source_image_uri
    datetime created_at
  }

  SCENE_SEGMENT {
    string scene_segment_id PK
    string run_id FK
    int start_ms
    int end_ms
    float confidence
  }

  DIALOGUE_SEGMENT {
    string dialogue_segment_id PK
    string run_id FK
    int start_ms
    int end_ms
    float confidence
  }

  NARRATION_SEGMENT {
    string narration_segment_id PK
    string run_id FK
    int start_ms
    int end_ms
    string text
    string style
    float confidence
    string metadata_json
  }

  MOVIE ||--o{ PROCESSING_RUN : has
  CONFIG_PRESET ||--o{ PROCESSING_RUN : uses
  CHARACTER ||--o{ FACE_EMBEDDING : owns
  PROCESSING_RUN ||--o{ SCENE_SEGMENT : produces
  PROCESSING_RUN ||--o{ DIALOGUE_SEGMENT : produces
  PROCESSING_RUN ||--o{ NARRATION_SEGMENT : produces
```

## 2. 说明

- `CHARACTER` + `FACE_EMBEDDING`：角色库与其人脸特征向量（可多个样本）。
- `PROCESSING_RUN`：一次处理运行，便于追溯配置与性能。
- `SCENE_SEGMENT`/`DIALOGUE_SEGMENT`/`NARRATION_SEGMENT`：按时间轴产出的核心结果。
