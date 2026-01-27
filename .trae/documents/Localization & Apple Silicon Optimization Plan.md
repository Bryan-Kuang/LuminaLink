I will update the project to support Apple Silicon (M2), switch the language to English, and refine the dialogue detection policy.

### 1. Localization (Chinese -> English)
- **Documentation**: Translate `docs/PRD.md`, `docs/ERD.md`, `docs/ARCHITECTURE.md`, `docs/INTERFACES.md`, and `README.md` to English.
- **NLG Modules**: Update `src/luminalink/nlg/generator.py` (RuleBased) and `src/luminalink/nlg/tf_template.py` (TensorFlow) to generate natural English narration (e.g., "Joe appears." instead of "Joe出现了。").
- **CLI & Logs**: Ensure all CLI outputs and internal logs are in English.

### 2. Apple Silicon Optimization (MPS Support)
- **Object Detector**: Update `src/luminalink/vision/object_detector_torchvision.py` to check for `torch.backends.mps.is_available()` and use the `mps` device for acceleration.
- **Face Pipeline**: Update `src/luminalink/vision/face_pipeline.py` to support `mps` device for Facenet PyTorch backend.
- **Verification**: Ensure the system falls back gracefully if MPS is not available (though on M2 it should be).

### 3. Dialogue Detection Refinement
- **Policy**: "Pause only for dialogue/voiceover" implies ignoring background noise.
- **Configuration**:
  - In `src/luminalink/config.py`, I will adjust the default `vad_mode` to `3` (most aggressive filtering of non-speech) to minimize false positives from ambient noise.
  - I will add a comment/docstring explaining this tuning for the "dialogue-only" requirement.

### 4. Verification
- **Self-Check**: Run `luminalink self-check` to ensure the pipeline remains functional after these changes.
- **Unit Tests**: Run `uv run pytest` to pass existing tests.
