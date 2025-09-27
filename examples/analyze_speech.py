#!/usr/bin/env python3
"""
Example script to analyze speech in a video without processing
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processor import VideoProcessor, ProcessingConfig

def main():
    # Basic configuration (no API key needed for analysis only)
    config = ProcessingConfig(
        whisper_model="base"
    )
    
    input_video = "input_video.mp4"
    
    if not Path(input_video).exists():
        print(f"Error: Input video not found: {input_video}")
        sys.exit(1)
    
    processor = VideoProcessor(config)
    
    try:
        print(f"Analyzing speech in: {input_video}")
        
        # Extract and analyze speech segments
        segments = processor.extract_and_analyze_speech(input_video)
        
        if not segments:
            print("No speech detected in the video")
            return
        
        print(f"\n📊 Found {len(segments)} speech segments:")
        print("=" * 80)
        
        total_speech_duration = 0
        
        for i, segment in enumerate(segments, 1):
            print(f"\n{i:2d}. [{segment.start:6.2f}s - {segment.end:6.2f}s] ({segment.duration:5.2f}s)")
            print(f"    Confidence: {segment.confidence:.3f}")
            print(f"    Text: {segment.text}")
            total_speech_duration += segment.duration
        
        print("\n" + "=" * 80)
        print(f"📈 Statistics:")
        print(f"   Total segments: {len(segments)}")
        print(f"   Total speech time: {total_speech_duration:.2f} seconds")
        print(f"   Average segment length: {total_speech_duration/len(segments):.2f} seconds")
        
        # Get video info for context
        video_info = processor.video_editor.get_video_info(input_video)
        speech_ratio = total_speech_duration / video_info['duration'] * 100
        print(f"   Video duration: {video_info['duration']:.2f} seconds")
        print(f"   Speech coverage: {speech_ratio:.1f}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main()