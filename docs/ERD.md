# Blind-Friendly Movie Narration System (Phase 1) ERD

This ERD describes the core data structures for "Character Store, Processing Run, Time Segment Artifacts" in Phase 1. Storage can use SQLite/PostgreSQL; Phase 1 can use SQLite for convenient local development.

## 1. Mermaid ER Diagram

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

## 2. Description
- `CHARACTER` + `FACE_EMBEDDING`: Character database and their face feature vectors (supports multiple samples).
- `PROCESSING_RUN`: A single processing run, facilitating traceability of configuration and performance.
- `SCENE_SEGMENT`/`DIALOGUE_SEGMENT`/`NARRATION_SEGMENT`: Core results generated along the timeline.

