from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from luminalink.config import AppConfig
from luminalink.pipeline.engine import NarrationEngine


def run_self_check() -> None:
    """Run lightweight self-checks to validate basic wiring and IO."""

    tmp_dir = Path(".tmp")
    tmp_dir.mkdir(exist_ok=True)
    sample = tmp_dir / "sample.mp4"
    if not sample.exists():
        _make_synthetic_video(str(sample))

    cfg = AppConfig()
    cfg.dialogue.enabled = False
    cfg.character_db.path = str(tmp_dir / "characters.sqlite")
    engine = NarrationEngine(cfg)

    t0 = time.time()
    events = engine.run_video_file(str(sample), output_jsonl_path=str(tmp_dir / "out.jsonl"))
    dt = time.time() - t0

    if dt <= 0:
        raise RuntimeError("Invalid timing")
    if len(events) == 0:
        raise RuntimeError("No narration events produced")


def _make_synthetic_video(path: str) -> None:
    """Create a small MP4 video for offline tests (no audio)."""

    import cv2

    w, h = 640, 360
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for i in range(72):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        color = (0, 255, 0) if i < 36 else (0, 0, 255)
        cv2.rectangle(img, (50 + i * 2 % 200, 80), (250, 260), color, -1)
        out.write(img)
    out.release()
