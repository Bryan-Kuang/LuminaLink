# LuminaLink Real-Time Narration — Design Document
**Date:** 2026-03-18
**Status:** Approved

---

## Purpose

LuminaLink describes what is visually happening in a film, in real time, for blind and low-vision users. A webcam is pointed at the TV. LuminaLink listens to the TV audio via microphone, detects silence/pause windows between dialogue, captures the current frame, sends it to GPT-4o Vision, and speaks a 1–2 sentence description through Edge TTS — all within 3–6 seconds of the silence beginning.

---

## Constraints

- **Input source:** Webcam pointed at TV screen
- **Latency budget:** 3–6 seconds from silence start to voice output
- **Narration trigger:** Silence in mic audio (TV audio), not a fixed timer
- **TTS engine:** Edge TTS (Microsoft neural voices, free)
- **Vision AI:** OpenAI GPT-4o (primary, no fallback provider)
- **Audio output:** Spoken aloud during silence windows only

---

## Architecture: 4-Thread Pipeline

```
Thread 1: Frame Capture
  Camera @ 10fps → 1-slot frame buffer (always latest frame)

Thread 2: Audio Monitor
  Microphone → RMS energy measurement → silence event (< -35dB for 1.5s)

Thread 3: Narrator (new module)
  silence_event + latest_frame → GPT-4o Vision → narration text → TTS queue

Thread 4: TTS Player
  narration_text → Edge TTS synthesis → audio playback
```

**Coordination rules:**
- Thread 3 is gated by Thread 4: will not trigger a new analysis while TTS is playing
- Minimum 5s cooldown between narrations regardless of silence events
- If GPT-4o response takes >4s, result is discarded (silence window likely over)

---

## The `narrator.py` Module (New)

Central coordinator — the missing piece of the existing codebase.

```python
class RealtimeNarrator:
    def on_silence_detected(self, frame, silence_duration)
    def _analyze_and_speak(self, frame)  # async, runs in thread 3
```

**GPT-4o prompt:**
> "You are an audio describer for a blind person watching a film. Describe what is visually happening RIGHT NOW in 1–2 sentences. Focus on: actions, expressions, scene changes, important on-screen text. Be concise and vivid. Do NOT describe dialogue — only visuals."

**Deduplication:** Skips narration if cosine similarity to any of the last 3 narrations exceeds 0.85 (simple word-overlap check, no embedding API call needed).

---

## What Changes

| File | Action | Reason |
|------|--------|--------|
| `src/narrator.py` | **Create** | The missing real-time orchestration module |
| `src/camera_controller.py` | **Delete** | Replaced by narrator.py + main.py wiring |
| `src/realtime_player.py` | **Delete** | Merged into tts_engine.py (redundant) |
| `src/audio_detector.py` | **Simplify** | Remove batch path, keep only real-time RMS silence detection |
| `src/main.py` | **Update** | Wire the 4 threads, remove batch camera logic |
| `src/gui/camera_app.py` | **Simplify** | Remove dead controls, keep camera feed + narration log |
| `src/scene_analyzer.py` | **Keep** | Already solid |
| `src/tts_engine.py` | **Keep** | Already solid, absorb playback from realtime_player |
| `src/config.py` | **Keep** | Already solid |
| `src/audio_input.py` | **Keep** | Already solid |

---

## Error Handling

| Failure | Behavior |
|---------|----------|
| GPT-4o timeout (>4s) | Discard, wait for next silence window |
| GPT-4o API error | Log warning, retry once, then skip |
| TTS synthesis fails | Fall back to pyttsx3 for that narration only |
| Camera frame unreadable | Skip analysis, reuse last good frame |
| Microphone unavailable | Startup error with clear message + `--mic` flag hint |
| No silence detected for 60s | Terminal log hint: "No silence detected — check mic level or threshold" |

---

## Entry Point

```bash
python -m src.main camera --camera 0 --mic 0
```

Optional flags:
- `--camera N` — camera device index (default: 0)
- `--mic N` — microphone device index (default: 0)
- `--silence-threshold -35` — RMS threshold in dB (default: -35)
- `--cooldown 5` — minimum seconds between narrations (default: 5)

---

## Success Criteria

1. Camera feed appears in GUI within 2s of launch
2. First narration fires within 6s of the first silence window
3. No two narrations overlap in audio
4. No crash on API timeout or mic disconnect
5. Narration is audibly distinct from the film's own audio (separate channel or during silence only)
