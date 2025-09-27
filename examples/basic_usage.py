#!/usr/bin/env python3
"""
Basic usage example for video speech processing
"""
import os
import sys
from pathlib import Path

# Add parent directory to path to import video_processor
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processor import VideoProcessor, ProcessingConfig

def main():
    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Configuration for processing
    config = ProcessingConfig(
        whisper_model="base",           # Fast and accurate enough for most use cases
        openai_model="gpt-3.5-turbo",  # Cost-effective rephrasing
        tts_voice="alloy",             # Natural sounding voice
        max_timing_error=1.0,          # Allow up to 1 second timing error
        preserve_original_audio=False  # Replace audio completely
    )
    
    # Input and output paths
    input_video = "input_video.mp4"
    output_video = "output_video.mp4"
    
    # Check if input video exists
    if not Path(input_video).exists():
        print(f"Error: Input video not found: {input_video}")
        print("Please provide a video file named 'input_video.mp4' in the current directory")
        sys.exit(1)
    
    # Create processor
    processor = VideoProcessor(config)
    
    try:
        print(f"Processing video: {input_video}")
        print("This may take several minutes depending on video length...")
        
        # Process the video
        result_path = processor.process_video(input_video, output_video)
        
        print(f"✅ Success! Processed video saved to: {result_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    finally:
        # Always cleanup
        processor.cleanup()

if __name__ == "__main__":
    main()