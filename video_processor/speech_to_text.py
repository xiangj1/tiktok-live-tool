"""
Speech-to-Text processor using OpenAI Whisper
"""
import whisper
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SpeechSegment:
    """Represents a speech segment with text and timing"""
    text: str
    start: float
    end: float
    confidence: float = 1.0
    
    @property
    def duration(self) -> float:
        return self.end - self.start

class SpeechToTextProcessor:
    """Process audio to extract speech segments with timestamps"""
    
    def __init__(self, model_name: str = "base"):
        """
        Initialize the speech-to-text processor
        
        Args:
            model_name: Whisper model name (tiny, base, small, medium, large)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the Whisper model"""
        try:
            logger.info(f"Loading Whisper model: {self.model_name}")
            self.model = whisper.load_model(self.model_name)
            logger.info("Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> List[SpeechSegment]:
        """
        Transcribe audio file to speech segments with timestamps
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of SpeechSegment objects with text and timing
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Transcribe with word-level timestamps
            result = self.model.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=False
            )
            
            segments = []
            for segment in result["segments"]:
                # Clean up the text (remove extra whitespace, etc.)
                text = segment["text"].strip()
                if text:  # Only add non-empty segments
                    speech_segment = SpeechSegment(
                        text=text,
                        start=segment["start"],
                        end=segment["end"],
                        confidence=segment.get("avg_logprob", 1.0)
                    )
                    segments.append(speech_segment)
            
            logger.info(f"Extracted {len(segments)} speech segments")
            return segments
            
        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}")
            raise
    
    def merge_short_segments(self, segments: List[SpeechSegment], 
                           min_duration: float = 2.0) -> List[SpeechSegment]:
        """
        Merge very short segments to avoid choppy audio replacement
        
        Args:
            segments: List of speech segments
            min_duration: Minimum duration for a segment (seconds)
            
        Returns:
            List of merged segments
        """
        if not segments:
            return segments
        
        merged = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # If current segment is too short and close to next, merge them
            if (current_segment.duration < min_duration and 
                next_segment.start - current_segment.end < 1.0):
                
                # Merge segments
                merged_text = f"{current_segment.text} {next_segment.text}"
                current_segment = SpeechSegment(
                    text=merged_text,
                    start=current_segment.start,
                    end=next_segment.end,
                    confidence=min(current_segment.confidence, next_segment.confidence)
                )
            else:
                merged.append(current_segment)
                current_segment = next_segment
        
        # Add the last segment
        merged.append(current_segment)
        
        logger.info(f"Merged segments: {len(segments)} -> {len(merged)}")
        return merged
    
    def process_audio_file(self, audio_path: str) -> List[SpeechSegment]:
        """
        Complete processing of audio file
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            List of processed speech segments
        """
        segments = self.transcribe_audio(audio_path)
        merged_segments = self.merge_short_segments(segments)
        return merged_segments