#!/usr/bin/env python3
"""
Batch processing example for multiple videos
"""
import os
import sys
from pathlib import Path
import glob

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processor import VideoProcessor, ProcessingConfig

def process_videos_in_directory(input_dir: str, output_dir: str):
    """Process all videos in a directory"""
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: Please set OPENAI_API_KEY environment variable")
        sys.exit(1)
    
    # Configuration for batch processing
    config = ProcessingConfig(
        whisper_model="base",           # Good balance of speed/quality
        openai_model="gpt-3.5-turbo",  # Cost effective
        tts_voice="alloy",
        max_timing_error=1.0,
        preserve_original_audio=False
    )
    
    # Find all video files
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.wmv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(input_path.glob(ext))
        video_files.extend(input_path.glob(ext.upper()))
    
    if not video_files:
        print(f"No video files found in: {input_dir}")
        return
    
    print(f"Found {len(video_files)} video files to process")
    
    # Process each video
    successful = 0
    failed = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing {i}/{len(video_files)}: {video_file.name}")
        print('='*60)
        
        # Create output filename
        output_file = output_path / f"processed_{video_file.stem}.mp4"
        
        # Skip if output already exists
        if output_file.exists():
            print(f"⏭️  Skipping (output exists): {output_file}")
            continue
        
        # Create new processor for each video to avoid memory issues
        processor = VideoProcessor(config)
        
        try:
            result_path = processor.process_video(str(video_file), str(output_file))
            print(f"✅ Success: {result_path}")
            successful += 1
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            failed += 1
            
        finally:
            processor.cleanup()
    
    print(f"\n{'='*60}")
    print(f"Batch processing completed!")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Output directory: {output_dir}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python batch_process.py <input_directory> <output_directory>")
        print("\nExample:")
        print("  python batch_process.py ./videos ./processed_videos")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    if not Path(input_dir).exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    process_videos_in_directory(input_dir, output_dir)

if __name__ == "__main__":
    main()