# Video Speech Processing Tool

A powerful tool for extracting speech from videos, rephrasing it using AI, and replacing the original audio while maintaining precise timing synchronization (±1 second accuracy).

## Features

- 🎤 **Speech-to-Text**: Extract speech with precise timestamps using OpenAI Whisper
- 🤖 **AI Rephrasing**: Rephrase content using OpenAI's GPT models while preserving meaning
- 🔊 **Text-to-Speech**: Convert rephrased text back to natural speech
- 🎬 **Video Processing**: Replace audio in videos with perfect timing synchronization
- ⚡ **High Accuracy**: Maintains timing precision within 1 second
- 🛠️ **Flexible Configuration**: Customizable models, voices, and processing parameters

## Installation

1. Clone the repository:
```bash
git clone https://github.com/xiangj1/tiktok-live-tool.git
cd tiktok-live-tool
```

2. Install dependencies:
```bash
pip install -r requirements.txt
# or
pip install -e .
```

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
# or create a .env file with your API key
cp .env.example .env
# Edit .env and add your API key
```

## Quick Start

### Basic Usage

Process a video with default settings:
```bash
python -m video_processor.main process input_video.mp4 output_video.mp4
```

### Analyze Speech First

Analyze what speech will be extracted:
```bash
python -m video_processor.main analyze input_video.mp4
```

### Preview Rephrasing

See how the first few segments will be rephrased:
```bash
python -m video_processor.main preview input_video.mp4
```

## Advanced Usage

### Custom Configuration

```bash
python -m video_processor.main process input.mp4 output.mp4 \
  --whisper-model medium \
  --openai-model gpt-4 \
  --tts-voice nova \
  --preserve-original \
  --original-volume 0.2 \
  --max-timing-error 0.5
```

### Available Options

- `--whisper-model`: Choose from `tiny`, `base`, `small`, `medium`, `large`
- `--openai-model`: GPT model for rephrasing (e.g., `gpt-3.5-turbo`, `gpt-4`)
- `--tts-voice`: Voice for speech synthesis (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`)
- `--preserve-original`: Keep original audio as background
- `--original-volume`: Volume of original audio (0.0-1.0)
- `--new-volume`: Volume of new audio (0.0-1.0)
- `--max-timing-error`: Maximum allowed timing error in seconds

## How It Works

1. **Audio Extraction**: Extracts audio track from input video
2. **Speech Recognition**: Uses Whisper AI to transcribe speech with timestamps
3. **Text Rephrasing**: Employs GPT models to rephrase while preserving meaning
4. **Speech Synthesis**: Converts rephrased text to speech using OpenAI TTS
5. **Timing Alignment**: Precisely matches new audio to original timestamps
6. **Video Assembly**: Replaces audio in video with synchronized result

## Technical Details

### Timing Accuracy
- Target accuracy: ±1 second
- Uses segment-level timing control
- Automatic speed adjustment for length matching
- Handles gaps and overlaps intelligently

### Supported Formats
- **Input**: MP4, AVI, MOV, MKV (any format supported by moviepy)
- **Audio**: WAV, MP3, AAC, FLAC
- **Output**: MP4 with H.264 video and AAC audio

### System Requirements
- Python 3.8+
- FFmpeg (for video processing)
- CUDA-compatible GPU (optional, for faster Whisper processing)
- OpenAI API key

## API Usage

```python
from video_processor import VideoProcessor, ProcessingConfig

# Configure processing
config = ProcessingConfig(
    whisper_model="base",
    openai_model="gpt-3.5-turbo",
    tts_voice="alloy",
    max_timing_error=1.0
)

# Process video
processor = VideoProcessor(config)
result = processor.process_video("input.mp4", "output.mp4")
```

## Examples

See the `examples/` directory for:
- Sample processing scripts
- Configuration examples
- Integration patterns

## Troubleshooting

### Common Issues

1. **API Key Error**: Ensure `OPENAI_API_KEY` is set correctly
2. **FFmpeg Not Found**: Install FFmpeg system-wide
3. **CUDA Out of Memory**: Use smaller Whisper model (`tiny` or `base`)
4. **Timing Errors**: Adjust `--max-timing-error` parameter

### Performance Tips

- Use `tiny` or `base` Whisper models for faster processing
- Enable GPU acceleration for Whisper if available
- Process shorter videos first to test configuration
- Use batch processing for multiple files

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

- 📚 Documentation: Check the `/docs` folder
- 🐛 Issues: Open an issue on GitHub
- 💬 Discussions: Use GitHub Discussions

---

Built with ❤️ using OpenAI Whisper, GPT, and TTS APIs