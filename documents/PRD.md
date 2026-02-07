# LuminaLink Camera Real-time Narration System - Product Requirements Document (PRD)

**Version**: 1.0
**Date**: 2026-02-06
**Status**: Draft

---

## 1. Product Overview

### 1.1 Product Vision

LuminaLink Camera Real-time Narration System aims to help visually impaired users obtain real-time AI-generated scene narrations by pointing a camera at a screen (cinema or home theater). The system automatically mutes when character dialogue is detected and provides scene descriptions during dialogue gaps, offering an accessible movie-watching experience for visually impaired users.

### 1.2 Product Positioning

- **Product Type**: Desktop GUI Application (based on Python + Tkinter)
- **Target Platform**: macOS (Priority), future support for Windows/Linux
- **Core Value**: Real-time, intelligent, and accessible movie narration service

### 1.3 Background and Motivation

**Current Issues**:

- Low coverage of Audio Description (AD) services in cinemas and streaming platforms
- Existing AD services need to be produced in advance and cannot be applied to all content
- Visually impaired individuals lack real-time auxiliary tools during home movie watching

**Solution**:

- Utilize AI vision understanding technology (GPT-4 Vision) for real-time scene analysis
- Capture screen content via camera without modifying playback devices
- Intelligently detect dialogue to avoid conflict with the original audio track
- Provide a simple and easy-to-use GUI interface supporting accessible operations

---

## 2. Target Users

### 2.1 Primary User Personas

**User A: Blind Movie Enthusiast**

- **Age**: 25-65 years old
- **Needs**: Obtain scene descriptions while watching movies at home or in the cinema
- **Pain Points**: Most existing movies do not have audio description tracks
- **Expectations**: Be able to use the device independently, start narration quickly, and have accurate narration when the picture is clear

**User B: Low Vision User**

- **Age**: 40-70 years old
- **Needs**: Can see the picture but cannot identify details, needing supplementary explanation
- **Pain Points**: Unable to identify distant characters or action details
- **Expectations**: Narration synchronized with the picture, without obscuring visible parts

### 2.2 Secondary Users

- **Family members/Caregivers**: Help visually impaired users set up and use the system
- **Accessibility Advocates**: Promote and test the system
- **Researchers**: Study AI-assisted accessibility technologies

---

## 3. Core Functional Requirements

### 3.1 Must-Have Features (MVP)

| Module                       | Description                                                                                   | Priority |
| ---------------------------- | --------------------------------------------------------------------------------------------- | -------- |
| **Camera Input**             | Supports real-time image capture from the device camera                                       | P0       |
| **GUI Interface**            | Provides a visual window displaying camera preview, control buttons, and status information   | P0       |
| **Real-time Scene Analysis** | Uses AI (GPT-4 Vision) to analyze captured frames and generate scene descriptions             | P0       |
| **Narration Generation**     | Converts scene analysis results into concise narration text (50-100 characters)               | P0       |
| **TTS Broadcast**            | Converts narration text to speech and plays it (supports EdgeTTS)                             | P0       |
| **Dialogue Detection**       | Captures ambient sound via microphone, detects dialogue, and pauses narration during dialogue | P0       |
| **Subtitle Overlay**         | Displays the current narration text at the bottom of the video preview window                 | P0       |
| **Basic Controls**           | Start, Pause, Stop buttons                                                                    | P0       |
| **Camera Selection**         | Supports selecting different camera devices (drop-down menu)                                  | P0       |
| **Status Display**           | Displays running status (Ready/Running/Paused/Stopped), latency, and FPS                      | P0       |
| **Keyboard Shortcuts**       | Space bar to pause/resume, ESC key to stop                                                    | P0       |

### 3.2 Should-Have Features

| Module                    | Description                                                                          | Priority |
| ------------------------- | ------------------------------------------------------------------------------------ | -------- |
| **Settings Panel**        | Configures TTS voice, speed, narration interval, subtitle display, etc.              | P1       |
| **Narration History Log** | Displays past narration records (with timestamps)                                    | P1       |
| **Performance Metrics**   | Displays detailed latency, frame rate, API call counts, and error statistics         | P1       |
| **Character Recognition** | Loads character configuration files, identifies specific people, and uses real names | P1       |
| **Audio Level Indicator** | Displays real-time current audio level (volume bar)                                  | P1       |
| **Error Prompts**         | Friendly error messages and resolution suggestions                                   | P1       |

### 3.3 Could-Have Features (Future)

| Module                     | Description                                                                 | Priority |
| -------------------------- | --------------------------------------------------------------------------- | -------- |
| **Recording and Playback** | Records video files with narration                                          | P2       |
| **Multi-language Support** | Supports multiple narration languages such as Chinese and English           | P2       |
| **Cloud Processing**       | Offloads AI inference to cloud servers                                      | P2       |
| **Mobile App**             | iOS/Android versions                                                        | P3       |
| **Advanced VAD**           | Uses WebRTC VAD or deep learning models for voice detection                 | P2       |
| **System Audio Capture**   | Supports virtual audio devices (BlackHole) to directly capture system audio | P2       |
| **Subtitle Export**        | Exports SRT subtitle files                                                  | P2       |
| **Multi-camera Switching** | Supports multiple cameras working and switching simultaneously              | P3       |
| **Performance Monitor**    | Detailed performance analysis and optimization suggestions                  | P2       |

---

## 4. User Journeys

### 4.1 First-time Use Flow

```
1. User downloads and installs LuminaLink
   ├─ Installs Python dependencies
   └─ Configures OpenAI API key

2. User starts the application
   ├─ Runs command: python -m src.main camera
   └─ GUI window opens, status displays "Ready"

3. User selects camera (if multiple)
   └─ Selects "FaceTime HD Camera" from drop-down menu

4. User clicks "Start" button
   ├─ Camera view appears in preview window
   └─ Status changes to "Running"

5. User points camera at movie screen
   └─ Adjusts angle to ensure a clear picture

6. System starts analysis and narration
   ├─ Analyzes the picture every 5-10 seconds
   ├─ Generates and plays narration during silent periods
   └─ Subtitles display at the bottom of the screen

7. During movie watching
   ├─ Hearing narration: continues watching
   ├─ Needing to pause: presses space bar or clicks "Pause"
   └─ Ending movie: presses ESC or clicks "Stop"
```

### 4.2 Daily Use Flow

```
1. Start application: python -m src.main camera
2. Click "Start" (last settings saved)
3. Point at screen and start watching
4. Click "Stop" after watching
```

### 4.3 Advanced Use Flow (Character Recognition)

```
1. Prepared character configuration file
   └─ Create characters.json containing character information

2. Start application and load configuration
   └─ python -m src.main camera --characters characters.json

3. System recognizes characters and uses real names in narration
   └─ Example: "Tony Stark enters the room" instead of "A man enters the room"
```

---

## 5. Non-Functional Requirements

### 5.1 Performance Requirements

| Metric                       | Target Value          | Measurement Method                        |
| ---------------------------- | --------------------- | ----------------------------------------- |
| **End-to-End Latency**       | <500ms (Ideal <300ms) | From camera capture to TTS playback start |
| **Video Capture Frame Rate** | 30 FPS                | Actual camera capture rate                |
| **Scene Analysis Frequency** | 5-10 FPS              | AI analysis call frequency                |
| **GUI Refresh Rate**         | ≥15 FPS               | Video preview window refresh              |
| **Memory Usage**             | <500 MB               | Peak runtime memory                       |
| **CPU Usage**                | <50% (Idle <10%)      | Average CPU usage                         |
| **Startup Time**             | <5 seconds            | From running command to GUI display       |

### 5.2 Accessibility Requirements

| Requirement                     | Implementation                                                    |
| ------------------------------- | ----------------------------------------------------------------- |
| **Keyboard-only Operation**     | All functions accessible via keyboard (Tab navigation, shortcuts) |
| **Screen Reader Compatibility** | Fully compatible with macOS VoiceOver                             |
| **Voice Status Prompts**        | All status changes (Start/Pause/Stop/Error) announced via TTS     |
| **High Contrast Mode**          | Supports system high contrast themes                              |
| **Adjustable Font Size**        | Interface text size is configurable                               |
| **Audio Feedback**              | Alert sounds for success/failure operations                       |

### 5.3 Reliability Requirements

| Scenario                          | Expected Behavior                                                    |
| --------------------------------- | -------------------------------------------------------------------- |
| **Camera Disconnected**           | Display error prompt, no crash, support reconnection                 |
| **API Failure**                   | Display warning, skip current frame, continue with subsequent frames |
| **Network Interruption**          | Display connection error, prompt to check network, support retry     |
| **Microphone Unavailable**        | Display warning, disable dialogue detection, continue narration      |
| **Insufficient System Resources** | Lower analysis frequency, display performance warning                |
| **Long-running**                  | Run continuously for over 30 minutes without memory leaks            |

### 5.4 Compatibility Requirements

| Platform    | Version Requirement        | Priority |
| ----------- | -------------------------- | -------- |
| **macOS**   | 12.0+ (Monterey and above) | P0       |
| **Windows** | 10/11                      | P2       |
| **Linux**   | Ubuntu 20.04+              | P2       |
| **Python**  | 3.8+                       | P0       |

---

## 6. User Interface Requirements

### 6.1 Main Window Layout

```
┌────────────────────────────────────────────────────────────┐
│  LuminaLink Camera Narration               [─] [□] [×]     │
├────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │                                                        │  │
│  │              Camera Preview (640x480)                 │  │
│  │                                                        │  │
│  │     [Subtitle Overlay: "A man walks into room"]      │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  Camera: [FaceTime HD Camera ▼]  FPS: 30  ●REC            │
│                                                              │
│  ┌─ Controls ──────────────────────────────────────────┐   │
│  │  [▶ Start]  [⏸ Pause]  [⏹ Stop]   Volume: [████░░] │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌─ Status ────────────────────────────────────────────┐   │
│  │  State: Running  │  Latency: 245ms  │  🎤 Silence   │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                              │
│  [▼ Settings] [▼ Narration Log]                            │
└────────────────────────────────────────────────────────────┘
```

### 6.2 UI Component Specifications

| Component          | Specification                                     |
| ------------------ | ------------------------------------------------- |
| **Window Size**    | Default 800x700, Minimum 640x600                  |
| **Preview Screen** | 640x480, Maintain 16:9 or 4:3 ratio               |
| **Button Size**    | Minimum 44x44 pixels (easy to click)              |
| **Font**           | System default font, Minimum 12pt                 |
| **Color**          | Follow system theme, support dark/light modes     |
| **Icons**          | Use Unicode characters (▶⏸⏹ etc.) or system icons |

---

## 7. Data Requirements

### 7.1 Input Data

| Data Type            | Source               | Format                        | Description                 |
| -------------------- | -------------------- | ----------------------------- | --------------------------- |
| **Video Stream**     | Camera device        | BGR Image (NumPy array)       | Real-time capture, 30 FPS   |
| **Audio Stream**     | Microphone           | PCM audio (22050 Hz)          | Used for dialogue detection |
| **Character Config** | JSON file (optional) | `{"characters": [...]}`       | Person information          |
| **User Settings**    | JSON file            | `~/.luminalink/settings.json` | User preferences            |

### 7.2 Output Data

| Data Type           | Output Target        | Format            | Description         |
| ------------------- | -------------------- | ----------------- | ------------------- |
| **Narration Audio** | Speaker              | MP3/WAV (EdgeTTS) | Real-time playback  |
| **Narration Text**  | GUI Subtitles        | UTF-8 String      | Displayed on screen |
| **Logs**            | Text file (optional) | Timestamp + Text  | For review          |

### 7.3 Configuration Data

**Character Configuration Example** (`characters.json`):

```json
{
  "characters": [
    {
      "name": "Tony Stark",
      "aliases": ["Iron Man", "Tony"],
      "description": "The main character, genius billionaire"
    }
  ]
}
```

**User Settings Example** (`settings.json`):

```json
{
  "camera_index": 0,
  "tts_voice": "en-US-AriaNeural",
  "tts_speed": 1.0,
  "narration_interval": 5,
  "show_subtitles": true,
  "audio_threshold": -40
}
```

---

## 8. Technical Constraints

### 8.1 External Dependencies

| Dependency            | Purpose                     | Constraint                                  |
| --------------------- | --------------------------- | ------------------------------------------- |
| **OpenAI API**        | GPT-4 Vision scene analysis | Requires API key, usage costs               |
| **EdgeTTS**           | Text-to-Speech              | Requires network connection                 |
| **Camera Device**     | Video input                 | Must have available camera                  |
| **Microphone Device** | Audio input                 | Recommended to have a microphone (optional) |

### 8.2 Performance Constraints

- **AI Inference Latency**: GPT-4V API calls usually take 150-300ms, which is the main source of latency
- **Network Bandwidth**: Each API call uploads ~100KB image data
- **Local Computing**: Video processing and audio analysis consume CPU resources

---

## 9. Success Metrics

### 9.1 Functional Metrics

- [ ] 100% core features (P0) implemented and passed testing
- [ ] Camera view displays correctly with latency <100ms
- [ ] AI narration accuracy ≥80% (via manual evaluation)
- [ ] Dialogue detection accuracy ≥85% (correctness during silent periods)

### 9.2 Performance Metrics

- [ ] End-to-end latency <500ms (95th percentile)
- [ ] GUI response time <100ms
- [ ] No crashes during 30 minutes of continuous operation
- [ ] CPU usage <50% (during analysis), <10% (when idle)

### 9.3 User Experience Metrics

- [ ] Startup time <5 seconds
- [ ] All functions support keyboard-only operation
- [ ] Error messages are clear and 100% provide resolution suggestions
- [ ] 100% compatible with VoiceOver

### 9.4 Quality Metrics

- [ ] Unit test coverage ≥70%
- [ ] Integration tests cover core flows
- [ ] No Critical or High priority bugs

---

## 10. Risks and Assumptions

### 10.1 Risks

| Risk                         | Probability | Impact | Mitigation                                          |
| ---------------------------- | ----------- | ------ | --------------------------------------------------- |
| High OpenAI API Latency      | Medium      | High   | Add local model support (LLaVA)                     |
| Inaccurate Audio Detection   | High        | Medium | Provide manual pause; adjustable threshold          |
| Camera Compatibility Issues  | Low         | High   | Detect at startup; provide device list              |
| High Cost (API Fees)         | Medium      | Medium | Provide local model options; control call frequency |
| Accessibility Non-compliance | Low         | High   | Invite visually impaired users for early testing    |

### 10.2 Assumptions

- ✓ User has an available camera device
- ✓ User has a stable network connection (for API calls)
- ✓ User has an OpenAI API key or is willing to register
- ✓ User device performance is sufficient (not too old)
- ✓ Camera can clearly capture the screen

---

## 11. Release Plan

### 11.1 MVP Release

**Version**: v0.1.0 (Beta)
**Date**: Within 1 week after implementation
**Scope**: All P0 features
**Goal**: Invite 5-10 visually impaired users for beta testing and collect feedback

### 11.2 Official Release

**Version**: v1.0.0
**Date**: 2-4 weeks after Beta testing
**Scope**: P0 + P1 features, fix issues found during Beta
**Goal**: Public release with full documentation and tutorials

### 11.3 Future Iterations

- **v1.1**: Add P2 features (Recording/Playback, Multi-language)
- **v2.0**: Cross-platform support (Windows/Linux)
- **v3.0**: Mobile app

---

## 12. Appendix

### 12.1 Glossary

| Term                       | Definition                                                         |
| -------------------------- | ------------------------------------------------------------------ |
| **Audio Description (AD)** | Scene narration service provided for visually impaired individuals |
| **TTS**                    | Text-to-Speech                                                     |
| **VAD**                    | Voice Activity Detection                                           |
| **GUI**                    | Graphical User Interface                                           |
| **FPS**                    | Frames Per Second                                                  |
| **Latency**                | The time difference between input and output                       |

### 12.2 References

- OpenAI GPT-4 Vision API Docs: https://platform.openai.com/docs/guides/vision
- EdgeTTS Docs: https://github.com/rany2/edge-tts
- WCAG 2.1 Accessibility Guidelines: https://www.w3.org/WAI/WCAG21/quickref/

---

**Document Maintenance**:

- **Creator**: Claude (LuminaLink Planning Agent)
- **Last Updated**: 2026-02-06
- **Next Review**: After implementation is complete
