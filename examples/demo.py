#!/usr/bin/env python3
"""
Demo script showing the complete video processing workflow
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def demonstrate_workflow():
    """Demonstrate the complete video processing workflow"""
    
    print("🎬 Video Speech Processing Tool Demo")
    print("=" * 50)
    
    print("\n📋 This tool implements the complete workflow:")
    print("1. 📹 Input: Video file")
    print("2. 🎤 Speech-to-Text: Extract speech with timestamps using Whisper")
    print("3. 🤖 LLM Rephrasing: Rephrase each sentence using GPT while preserving meaning")
    print("4. 🔊 Text-to-Speech: Convert rephrased text to audio using OpenAI TTS")
    print("5. ⏱️  Timing Sync: Match new audio to original timing (±1 second accuracy)")
    print("6. 🎬 Output: Video with replaced audio track")
    
    print("\n🛠️ Usage Examples:")
    print("=" * 30)
    
    print("\n1. Process a video (complete workflow):")
    print("   python -m video_processor.main process input.mp4 output.mp4")
    
    print("\n2. Analyze speech in video first:")
    print("   python -m video_processor.main analyze input.mp4")
    
    print("\n3. Preview rephrasing (first few segments):")
    print("   python -m video_processor.main preview input.mp4")
    
    print("\n4. Batch process multiple videos:")
    print("   python examples/batch_process.py ./videos ./processed")
    
    print("\n⚙️ Configuration Options:")
    print("=" * 25)
    print("• --whisper-model: tiny, base, small, medium, large")
    print("• --openai-model: gpt-3.5-turbo, gpt-4")
    print("• --tts-voice: alloy, echo, fable, onyx, nova, shimmer")
    print("• --preserve-original: Keep background original audio")
    print("• --max-timing-error: Maximum allowed timing error (default: 1.0s)")
    
    print("\n🔧 Setup Required:")
    print("=" * 17)
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set OpenAI API key: export OPENAI_API_KEY='your-key'")
    print("3. Provide input video file")
    
    print("\n✨ Key Features:")
    print("=" * 15)
    print("✅ Precise timing synchronization (±1 second)")
    print("✅ Intelligent segment merging for smooth audio")
    print("✅ Automatic speed adjustment for timing match")
    print("✅ Preserves original meaning during rephrasing")
    print("✅ Comprehensive error handling and logging")
    print("✅ Support for multiple video formats")
    print("✅ CLI and Python API interfaces")
    
    print("\n🎯 Perfect for:")
    print("=" * 13)
    print("• Content creators improving video scripts")
    print("• Language learning and pronunciation practice")
    print("• Accessibility improvements")
    print("• Video dubbing and localization")
    print("• Creative content transformation")
    
    print("\n" + "=" * 50)
    print("🚀 Ready to process your videos!")

if __name__ == "__main__":
    demonstrate_workflow()