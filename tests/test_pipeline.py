"""
Tests for the video processing pipeline
"""
import unittest
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from video_processor.speech_to_text import SpeechSegment, SpeechToTextProcessor
from video_processor.pipeline import ProcessingConfig


class TestSpeechSegment(unittest.TestCase):
    """Test the SpeechSegment class"""
    
    def test_speech_segment_creation(self):
        """Test creating a speech segment"""
        segment = SpeechSegment(
            text="Hello world",
            start=1.0,
            end=3.5,
            confidence=0.95
        )
        
        self.assertEqual(segment.text, "Hello world")
        self.assertEqual(segment.start, 1.0)
        self.assertEqual(segment.end, 3.5)
        self.assertEqual(segment.confidence, 0.95)
        self.assertEqual(segment.duration, 2.5)
    
    def test_speech_segment_duration(self):
        """Test duration calculation"""
        segment = SpeechSegment(
            text="Test",
            start=0.5,
            end=2.0
        )
        self.assertEqual(segment.duration, 1.5)


class TestProcessingConfig(unittest.TestCase):
    """Test the ProcessingConfig class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = ProcessingConfig()
        
        self.assertEqual(config.whisper_model, "whisper-1")
        self.assertEqual(config.openai_model, "gpt-3.5-turbo")
        self.assertEqual(config.tts_voice, "alloy")
        self.assertEqual(config.min_segment_duration, 2.0)
        self.assertEqual(config.max_timing_error, 1.0)
        self.assertFalse(config.preserve_original_audio)
        self.assertFalse(config.allow_tts_time_stretch)
        self.assertAlmostEqual(config.duration_match_tolerance, 0.01)
        self.assertAlmostEqual(config.max_duration_overrun_ratio, 0.5)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = ProcessingConfig(
            whisper_model="whisper-large-v3",
            openai_model="gpt-4",
            tts_voice="nova",
            max_timing_error=0.5,
            preserve_original_audio=True,
            allow_tts_time_stretch=True,
            duration_match_tolerance=0.02,
            max_duration_overrun_ratio=0.3
        )
        
        self.assertEqual(config.whisper_model, "whisper-large-v3")
        self.assertEqual(config.openai_model, "gpt-4")
        self.assertEqual(config.tts_voice, "nova")
        self.assertEqual(config.max_timing_error, 0.5)
        self.assertTrue(config.preserve_original_audio)
        self.assertTrue(config.allow_tts_time_stretch)
        self.assertAlmostEqual(config.duration_match_tolerance, 0.02)
        self.assertAlmostEqual(config.max_duration_overrun_ratio, 0.3)


class TestSpeechToTextProcessor(unittest.TestCase):
    """Test the SpeechToTextProcessor class"""
    
    def test_merge_short_segments(self):
        """Test merging of short segments"""
        # Create mock processor (without loading actual model)
        processor = SpeechToTextProcessor.__new__(SpeechToTextProcessor)
        processor.model_name = "base"
        processor.model = None
        
        # Create test segments
        segments = [
            SpeechSegment("Hello", 0.0, 1.0),      # Short segment
            SpeechSegment("world", 1.2, 2.0),      # Short segment, close to previous
            SpeechSegment("This is a longer segment", 3.0, 6.0),  # Long segment
            SpeechSegment("End", 7.0, 7.5)         # Short segment, far from previous
        ]
        
        merged = processor.merge_short_segments(segments, min_duration=2.0)
        
        # First two segments should be merged
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged[0].text, "Hello world")
        self.assertEqual(merged[0].start, 0.0)
        self.assertEqual(merged[0].end, 2.0)
        self.assertEqual(merged[1].text, "This is a longer segment")
        self.assertEqual(merged[2].text, "End")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)