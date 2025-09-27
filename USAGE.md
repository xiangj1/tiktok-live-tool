# Usage Guide

## Quick Start

### 1. Installation
```bash
# Clone repository
git clone https://github.com/xiangj1/tiktok-live-tool.git
cd tiktok-live-tool

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Basic Usage

Process a video with default settings:
```bash
python -m video_processor.main process input_video.mp4 output_video.mp4
```

## Commands

### `process` - Process Video
Complete video processing with speech rephrasing:
```bash
python -m video_processor.main process INPUT OUTPUT [OPTIONS]
```

**Options:**
- `--api-key`: OpenAI API key (or set OPENAI_API_KEY env var)
- `--whisper-model`: Speech recognition model (tiny/base/small/medium/large)
- `--openai-model`: GPT model for rephrasing (gpt-3.5-turbo/gpt-4)
- `--tts-voice`: Voice for speech synthesis (alloy/echo/fable/onyx/nova/shimmer)
- `--preserve-original`: Keep original audio as background
- `--original-volume`: Volume level for original audio (0.0-1.0)
- `--new-volume`: Volume level for new audio (0.0-1.0)
- `--max-timing-error`: Maximum allowed timing error in seconds

### `analyze` - Analyze Speech 
Extract and analyze speech without processing:
```bash
python -m video_processor.main analyze INPUT [OPTIONS]
```

### `preview` - Preview Rephrasing
See how first few segments will be rephrased:
```bash
python -m video_processor.main preview INPUT [OPTIONS]
```

## Configuration Examples

### High Quality Processing
```bash
python -m video_processor.main process input.mp4 output.mp4 \
  --whisper-model large \
  --openai-model gpt-4 \
  --tts-voice nova \
  --max-timing-error 0.5
```

### Fast Processing
```bash
python -m video_processor.main process input.mp4 output.mp4 \
  --whisper-model tiny \
  --openai-model gpt-3.5-turbo \
  --tts-voice alloy
```

### Mixed Audio (Preserve Original)
```bash
python -m video_processor.main process input.mp4 output.mp4 \
  --preserve-original \
  --original-volume 0.2 \
  --new-volume 0.8
```

## Python API

```python
from video_processor import VideoProcessor, ProcessingConfig

# Configure processing
config = ProcessingConfig(
    whisper_model="base",
    openai_model="gpt-3.5-turbo",
  tts_voice="alloy",
  max_timing_error=1.0,
  allow_tts_time_stretch=True  # optional: re-enable audio stretching if preferred
)

# Process video
processor = VideoProcessor(config)
result = processor.process_video("input.mp4", "output.mp4")
```

## Workflow Steps

1. **Audio Extraction**: Video → Audio file
2. **Speech Recognition**: Audio → Text segments with timestamps
3. **Text Rephrasing**: Original text → Rephrased text (preserving meaning)
4. **Speech Synthesis**: Rephrased text → New audio with timing control
5. **Audio Replacement**: New audio → Video with replaced track

## Tips for Best Results

### Speech Recognition
- Use higher quality Whisper models for better accuracy
- Ensure clear audio without too much background noise
- Consider pre-processing audio if quality is poor

### Rephrasing
- GPT-4 provides better rephrasing quality than GPT-3.5-turbo
- Longer segments generally get better rephrasing results
- The tool automatically merges very short segments

### Speech Synthesis
- Different voices work better for different content types
- `nova` and `shimmer` are female voices
- `alloy`, `echo`, `fable`, `onyx` are male voices

### Timing Accuracy
- Default 1-second tolerance works for most content
- Use smaller `--max-timing-error` for precise synchronization
- Very short segments may have larger timing variations
- When generated speech ends sooner, the video tail is trimmed automatically; when speech runs longer (within reasonable bounds), the video extends using reverse playback to keep everything aligned.
- Time-stretching of TTS audio is disabled by default to preserve natural voice tone; you can re-enable it via `ProcessingConfig(allow_tts_time_stretch=True)` if you prefer audio stretching over video trimming/extensions.

## Troubleshooting

### Common Issues

**"No speech detected"**
- Check audio quality and volume
- Try a more sensitive Whisper model (medium/large)
- Verify the video has an audio track

**"API key error"**
- Ensure OPENAI_API_KEY environment variable is set
- Check API key has sufficient credits
- Verify key has access to required models

**"Timing errors too large"**
- Increase `--max-timing-error` parameter
- Check if speech is very fast or slow
- Consider manual segment adjustment

**"Out of memory"**
- Use smaller Whisper model (tiny/base)
- Process shorter video segments
- Close other applications

### Performance Optimization

**For Speed:**
- Use `whisper-model tiny` or `base`
- Use `gpt-3.5-turbo` instead of `gpt-4`
- Process in smaller batches

**For Quality:**
- Use `whisper-model large`
- Use `gpt-4` for rephrasing
- Use high-quality TTS voices

**For Cost:**
- Use smaller models when possible
- Preview rephrasing before full processing
- Batch process multiple videos efficiently