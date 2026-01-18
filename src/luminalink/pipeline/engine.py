from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass

from luminalink.audio.dialogue_webrtcvad import VADConfig, WebRTCVADDialogueDetector
from luminalink.config import AppConfig
from luminalink.input.audio_ffmpeg import FFmpegAudioInput
from luminalink.input.video_file import VideoFileInput
from luminalink.nlg.generator import RuleBasedNarrationGenerator
from luminalink.nlg.tf_template import TensorFlowTemplateNarrator
from luminalink.output.jsonl_sink import JSONLSink
from luminalink.store.character_store import CharacterStore
from luminalink.types import CharacterMatch, DialogueState, NarrationEvent, VisualFacts
from luminalink.vision.face_pipeline import FacePipeline
from luminalink.vision.object_detector import NoopObjectDetector
from luminalink.vision.object_detector_torchvision import TorchVisionObjectDetector
from luminalink.vision.scene_change import HistogramSceneChangeDetector, SceneChangeConfig


@dataclass
class _DialogueShared:
    state: DialogueState
    lock: threading.Lock


class NarrationEngine:
    """Orchestrate phase-1 streaming pipeline from video/audio to narration events."""

    def __init__(self, cfg: AppConfig):
        self._cfg = cfg

        sc_cfg = SceneChangeConfig(
            hist_correlation_threshold=cfg.vision.scene_change.hist_correlation_threshold,
            warmup_frames=cfg.vision.scene_change.warmup_frames,
        )
        self._scene = HistogramSceneChangeDetector(sc_cfg)
        self._faces = FacePipeline()

        if cfg.vision.object_detector == "torchvision":
            self._objects = TorchVisionObjectDetector(score_threshold=0.55)
        else:
            self._objects = NoopObjectDetector()

        self._store = CharacterStore(cfg.character_db.path)
        self._store.migrate()

        self._narrator = self._select_narrator(cfg)

    def run_video_file(self, video_path: str, output_jsonl_path: str | None) -> list[NarrationEvent]:
        """Run narration on a local video file and return produced events."""

        run_id = f"run_{uuid.uuid4().hex[:10]}"
        sink = JSONLSink(output_jsonl_path)

        dialogue_shared = _DialogueShared(state=DialogueState(is_dialogue=False, confidence=0.0), lock=threading.Lock())
        audio_thread = None
        if self._cfg.dialogue.enabled:
            audio_thread = threading.Thread(
                target=self._run_audio_dialogue_loop,
                args=(video_path, dialogue_shared),
                daemon=True,
            )
            audio_thread.start()

        events: list[NarrationEvent] = []
        input_ = VideoFileInput(video_path)
        try:
            last_emit_ms = -10**9
            target_period_ms = int(1000.0 / self._cfg.performance.target_fps)
            last_processed_ms = -10**9

            for frame in input_.frames():
                if frame.pts_ms < last_processed_ms + target_period_ms:
                    continue
                last_processed_ms = frame.pts_ms

                scene_change = self._scene.is_scene_change(frame.image_bgr)
                objects = self._objects.detect(frame.image_bgr) if (scene_change or frame.frame_index % 8 == 0) else []

                faces = self._faces.detect_faces(frame.image_bgr)
                embeddings = self._faces.embed_faces(frame.image_bgr, faces)
                characters: list[CharacterMatch] = []
                for face, emb in zip(faces, embeddings, strict=False):
                    match = self._store.match(self._faces.model_name, emb)
                    if match is None:
                        continue
                    ch_id, name, sim = match
                    if sim < self._cfg.character_db.match_threshold:
                        continue
                    characters.append(CharacterMatch(character_id=ch_id, display_name=name, confidence=sim, bbox_xyxy=face.bbox_xyxy))

                facts = VisualFacts(pts_ms=frame.pts_ms, scene_change=scene_change, objects=objects, characters=characters)
                with dialogue_shared.lock:
                    dstate = dialogue_shared.state

                allow_emit = not dstate.is_dialogue
                interval_ok = frame.pts_ms >= last_emit_ms + self._cfg.performance.narration_min_interval_ms
                trigger = scene_change or (frame.frame_index % 24 == 0)

                if allow_emit and interval_ok and trigger:
                    text, conf = self._narrator.generate(facts, self._cfg.nlg)
                    event = NarrationEvent(
                        run_id=run_id,
                        timestamp_start_ms=frame.pts_ms,
                        timestamp_end_ms=frame.pts_ms + self._cfg.performance.narration_duration_ms,
                        text=text,
                        style=self._cfg.nlg.style,
                        detail_level=self._cfg.nlg.detail_level,
                        confidence=float(conf),
                        metadata={
                            "scene_change": bool(scene_change),
                            "objects": [{"label": o.label, "conf": o.confidence} for o in objects],
                            "characters": [{"id": c.character_id, "name": c.display_name, "conf": c.confidence} for c in characters],
                            "dialogue": {"is_dialogue": dstate.is_dialogue, "confidence": dstate.confidence},
                        },
                    )
                    sink.emit(event)
                    events.append(event)
                    last_emit_ms = frame.pts_ms

        finally:
            input_.close()
            sink.close()

        if audio_thread is not None:
            audio_thread.join(timeout=0.2)
        return events

    def _run_audio_dialogue_loop(self, video_path: str, shared: _DialogueShared) -> None:
        """Continuously update dialogue state from the video's audio stream."""

        vad_cfg = VADConfig(
            vad_mode=self._cfg.dialogue.vad_mode,
            start_trigger_frames=self._cfg.dialogue.start_trigger_frames,
            end_trigger_frames=self._cfg.dialogue.end_trigger_frames,
        )
        detector = WebRTCVADDialogueDetector(vad_cfg)
        audio = FFmpegAudioInput(
            video_path=video_path,
            sample_rate_hz=self._cfg.dialogue.sample_rate_hz,
            frame_ms=self._cfg.dialogue.frame_ms,
        )
        try:
            for aframe in audio.frames():
                state = detector.update(aframe)
                with shared.lock:
                    shared.state = state
        finally:
            audio.close()

    @staticmethod
    def _select_narrator(cfg: AppConfig):
        """Select a narration generator based on configuration and available deps."""

        if cfg.nlg.backend == "rule":
            return RuleBasedNarrationGenerator()
        if cfg.nlg.backend == "tensorflow":
            return TensorFlowTemplateNarrator()

        try:
            return TensorFlowTemplateNarrator()
        except Exception:
            return RuleBasedNarrationGenerator()

