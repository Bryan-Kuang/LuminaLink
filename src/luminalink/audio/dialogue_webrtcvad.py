from __future__ import annotations

from dataclasses import dataclass

import webrtcvad

from luminalink.types import AudioFrame, DialogueState


@dataclass
class VADConfig:
    """Configuration for WebRTC VAD-based dialogue detection."""

    vad_mode: int
    start_trigger_frames: int
    end_trigger_frames: int


class WebRTCVADDialogueDetector:
    """Detect dialogue segments in audio using WebRTC VAD."""

    def __init__(self, cfg: VADConfig):
        self._vad = webrtcvad.Vad(cfg.vad_mode)
        self._cfg = cfg
        self._active_start_ms: int | None = None
        self._speech_run = 0
        self._silence_run = 0

    def update(self, frame: AudioFrame) -> DialogueState:
        """Consume one audio frame and return updated dialogue state."""

        is_speech = self._vad.is_speech(frame.pcm_s16le, frame.sample_rate_hz)
        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
        else:
            self._silence_run += 1
            self._speech_run = 0

        if self._active_start_ms is None:
            if self._speech_run >= self._cfg.start_trigger_frames:
                self._active_start_ms = frame.start_ms
                return DialogueState(is_dialogue=True, confidence=0.8, active_segment_start_ms=self._active_start_ms)
            return DialogueState(is_dialogue=False, confidence=0.2)

        if self._silence_run >= self._cfg.end_trigger_frames:
            self._active_start_ms = None
            return DialogueState(is_dialogue=False, confidence=0.7)

        return DialogueState(is_dialogue=True, confidence=0.85, active_segment_start_ms=self._active_start_ms)

