# LuminaLink

**AI-Powered Movie Audio Description for the Visually Impaired**

LuminaLink is an intelligent video analysis system that provides real-time audio descriptions of movies and videos for visually impaired audiences. It uses advanced AI to analyze scenes, recognize characters, and generate natural language narrations.

## Features

- 🎬 **Smart Video Analysis**: Automatically detects key scenes and important visual changes
- 👤 **Character Recognition**: Identifies and tracks characters throughout the video, using actual names instead of generic descriptions
- 🎙️ **Natural Narration**: Generates concise, descriptive narrations that fit during dialogue pauses
- 🔊 **Text-to-Speech**: High-quality voice synthesis for audio output
- 📝 **Subtitle Export**: Export narrations as SRT subtitle files

## Installation

### Prerequisites

- Python 3.8+
- FFmpeg (for audio processing)
- OpenAI API key

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/LuminaLink.git
cd LuminaLink
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Configure environment variables:

```bash
cp .env.example .env
# Edit .env and add your API keys
```

## Usage

### Basic Usage

Generate narration subtitles for a video:

```bash
python -m src.main --video path/to/video.mp4 --output narration.srt
```

### With Character Configuration

Create a character configuration file to improve character recognition:

```json
{
  "characters": [
    {
      "name": "Charlie",
      "aliases": ["Lt. Col. Slade", "Colonel"],
      "description": "A retired Army officer"
    },
    {
      "name": "Frank",
      "aliases": ["Frank Slade"],
      "description": "The main character"
    }
  ]
}
```

Then run with the config:

```bash
python -m src.main --video video.mp4 --characters characters.json --output narration.srt
```

### Real-time Preview

Enable preview mode to see the video with narration overlays:

```bash
python -m src.main --video video.mp4 --preview
```

## Configuration

### Environment Variables

| Variable               | Description                      | Default            |
| ---------------------- | -------------------------------- | ------------------ |
| `OPENAI_API_KEY`       | OpenAI API key (required)        | -                  |
| `OPENAI_MODEL`         | Model to use for analysis        | `gpt-4o`           |
| `TTS_ENGINE`           | TTS engine (edge, gtts, pyttsx3) | `edge`             |
| `TTS_VOICE`            | Voice for narration              | `en-US-AriaNeural` |
| `TTS_SPEED`            | Speech rate                      | `1.0`              |
| `NARRATION_INTERVAL`   | Min seconds between narrations   | `5`                |
| `NARRATION_MAX_LENGTH` | Max characters per narration     | `100`              |

### TTS Voices

For Edge TTS, you can use voices like:

- `en-US-AriaNeural` - Female, US English
- `en-US-GuyNeural` - Male, US English
- `en-GB-SoniaNeural` - Female, UK English
- `zh-CN-XiaoxiaoNeural` - Female, Chinese

## Project Structure

```
LuminaLink/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── video_processor.py # Video frame extraction
│   ├── audio_detector.py  # Audio/silence detection
│   ├── character_recognizer.py  # Character recognition
│   ├── scene_analyzer.py  # AI scene analysis
│   ├── narrator.py        # Narration generation
│   ├── tts_engine.py      # Text-to-speech
│   ├── main.py            # Main entry point
│   └── realtime_player.py # Realtime playback
├── tests/
│   └── test_modules.py    # Unit tests
├── .env.example           # Environment template
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## How It Works

1. **Video Processing**: Extracts frames at regular intervals and detects scene changes
2. **Audio Analysis**: Identifies silent periods suitable for narration insertion
3. **Character Recognition**: Uses face detection/recognition to identify characters
4. **Scene Analysis**: Sends frames to GPT-4o Vision for detailed scene understanding
5. **Narration Generation**: Creates concise, natural descriptions of the action
6. **TTS Synthesis**: Converts narrations to speech audio
7. **Subtitle Export**: Outputs timed subtitles in SRT format

## API Usage

### Python API

```python
from src.main import LuminaLink
import asyncio

app = LuminaLink(
    video_path="movie.mp4",
    output_subtitles="narration.srt"
)

asyncio.run(app.run())
```

### Character Recognition API

```python
from src.character_recognizer import CharacterRecognizer

recognizer = CharacterRecognizer()
recognizer.add_character(
    name="John",
    aliases=["Johnny", "Mr. Smith"],
    description="The protagonist"
)

# Get characters in a frame
characters = recognizer.get_characters_in_frame(frame, timestamp=0.0)
```

## Troubleshooting

### Common Issues

1. **"Module not found" errors**
   - Make sure you've activated the virtual environment
   - Run `pip install -r requirements.txt`

2. **OpenAI API errors**
   - Check your API key in `.env`
   - Ensure you have API credits available

3. **Audio extraction fails**
   - Install FFmpeg: `brew install ffmpeg` (macOS) or `apt install ffmpeg` (Linux)

4. **Character recognition not working**
   - Install face_recognition: `pip install face_recognition`
   - On macOS, you may need: `brew install cmake dlib`

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4 Vision API
- Microsoft for Edge TTS
- OpenCV and face_recognition libraries
