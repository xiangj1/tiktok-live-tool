"""
Speech-to-Text processor using OpenAI Cloud API
"""
import openai
import os
from typing import List
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
    """Process audio to extract speech segments with timestamps using OpenAI API"""

    def __init__(self, api_key: str = None, model_name: str = "whisper-1"):
        """
        Initialize the speech-to-text processor

        Args:
            api_key: OpenAI API key
            model_name: OpenAI Whisper model name (default: whisper-1)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model_name
        if not self.api_key:
            raise ValueError("OpenAI API key is required for cloud STT.")
        self.client = openai.OpenAI(api_key=self.api_key)

    def transcribe_audio(self, audio_path: str) -> List[SpeechSegment]:
        """
        Transcribe audio file to speech segments with timestamps using OpenAI API

        Args:
            audio_path: Path to the audio file

        Returns:
            List of SpeechSegment objects with text and timing
        """
        try:
            logger.info(f"Transcribing audio via OpenAI API: {audio_path}")
            with open(audio_path, "rb") as audio_file:
                response = self.client.audio.transcriptions.create(
                    model=self.model_name,
                    file=audio_file,
                    response_format="verbose_json"
                )

            segments = []
            for segment in response.segments:
                if isinstance(segment, dict):
                    text = segment.get("text", "").strip()
                    start = float(segment.get("start", 0.0))
                    end = float(segment.get("end", start))
                    confidence = segment.get("confidence")
                    if confidence is None:
                        confidence = segment.get("avg_logprob", 1.0)
                else:
                    text = getattr(segment, "text", "").strip()
                    start = getattr(segment, "start", 0.0)
                    end = getattr(segment, "end", start)
                    confidence = getattr(segment, "confidence", 1.0)

                if text:
                    speech_segment = SpeechSegment(
                        text=text,
                        start=start,
                        end=end,
                        confidence=confidence if confidence is not None else 1.0
                    )
                    segments.append(speech_segment)

            logger.info(f"Extracted {len(segments)} speech segments from cloud API")
            return segments
        except Exception as e:
            logger.error(f"Failed to transcribe audio via OpenAI API: {e}")
            raise

    def merge_short_segments(self, segments: List[SpeechSegment], min_duration: float = 2.0) -> List[SpeechSegment]:
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
            if (current_segment.duration < min_duration and next_segment.start - current_segment.end < 1.0):
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