"""
Text-to-Speech processor using OpenAI TTS API
"""
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import openai

from .speech_to_text import SpeechSegment


class TTSDurationMismatch(Exception):
    """Raised when generated TTS audio stays outside the allowed duration tolerance."""

    def __init__(
        self,
        message: str,
        audio_data: np.ndarray,
        sample_rate: int,
        target_duration: float,
        actual_duration: float,
    ):
        super().__init__(message)
        self.audio_data = audio_data
        self.sample_rate = sample_rate
        self.target_duration = target_duration
        self.actual_duration = actual_duration

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AudioSegment:
    """Represents an audio segment with timing"""
    audio_data: np.ndarray
    sample_rate: int
    start: float
    end: float
    duration: float

class TextToSpeechProcessor:
    """Convert text to speech with timing control"""
    
    def __init__(self, api_key: Optional[str] = None, voice: str = "alloy"):
        """
        Initialize the TTS processor
        
        Args:
            api_key: OpenAI API key
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.voice = voice
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required for TTS")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def generate_speech(
        self,
        text: str,
        target_duration: Optional[float] = None,
        *,
        max_attempts: int = 1,
        duration_tolerance_ratio: float = 0.15,
        backoff_seconds: float = 2.0,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate speech audio from text
        
        Args:
            text: Text to convert to speech
            target_duration: Target duration in seconds (for speed adjustment)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        attempt = 0
        last_error: Optional[Exception] = None

        while attempt < max_attempts:
            attempt += 1
            try:
                logger.debug(
                    "Generating speech attempt %d/%d for segment '%.50s'...",
                    attempt,
                    max_attempts,
                    text,
                )

                response = self.client.audio.speech.create(
                    model="tts-1",
                    voice=self.voice,
                    input=text,
                    response_format="mp3"
                )

                temp_file = self.temp_dir / f"tts_{hash((text, attempt))}.mp3"
                response.stream_to_file(temp_file)

                audio_data, sample_rate = librosa.load(temp_file, sr=None)
                temp_file.unlink(missing_ok=True)

                if target_duration and target_duration > 0:
                    current_duration = len(audio_data) / sample_rate
                    ratio = current_duration / target_duration if target_duration else 1.0

                    if abs(ratio - 1.0) > duration_tolerance_ratio:
                        logger.warning(
                            "TTS duration ratio %.2f outside tolerance ±%.2f (target %.2fs, actual %.2fs)",
                            ratio,
                            duration_tolerance_ratio,
                            target_duration,
                            current_duration,
                        )

                        if attempt < max_attempts:
                            time.sleep(backoff_seconds)
                            continue

                        raise TTSDurationMismatch(
                            "Generated TTS audio remains outside tolerance after retries",
                            audio_data,
                            sample_rate,
                            target_duration,
                            current_duration,
                        )

                return audio_data, sample_rate

            except Exception as e:
                last_error = e
                logger.warning(
                    "TTS attempt %d/%d failed: %s",
                    attempt,
                    max_attempts,
                    e,
                )
                if attempt < max_attempts:
                    time.sleep(backoff_seconds)

        logger.error("All TTS attempts failed for segment: %.50s", text)
        if last_error:
            raise last_error
        raise RuntimeError("Unknown TTS failure")
    
    def process_segments(
        self,
        segments: List[SpeechSegment],
        *,
        max_attempts: int = 1,
        duration_tolerance_ratio: float = 0.15,
        retry_backoff_seconds: float = 2.0
    ) -> List[AudioSegment]:
        """
        Convert speech segments to audio segments
        
        Args:
            segments: List of speech segments with rephrased text
            
        Returns:
            List of audio segments
        """
        audio_segments = []
        
        logger.info(f"Converting {len(segments)} segments to speech...")
        
        for i, segment in enumerate(segments):
            logger.info(f"Processing TTS segment {i + 1}/{len(segments)}")
            
            try:
                target_duration = segment.duration
                audio_data, sample_rate = self.generate_speech(
                    segment.text,
                    target_duration,
                    max_attempts=max_attempts,
                    duration_tolerance_ratio=duration_tolerance_ratio,
                    backoff_seconds=retry_backoff_seconds,
                )
                
                audio_segment = AudioSegment(
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    start=segment.start,
                    end=segment.end,
                    duration=len(audio_data) / sample_rate
                )
                
                audio_segments.append(audio_segment)
                
                # Log timing information
                actual_duration = len(audio_data) / sample_rate
                timing_error = abs(actual_duration - target_duration)
                logger.debug(f"Segment {i + 1}: target={target_duration:.2f}s, actual={actual_duration:.2f}s, error={timing_error:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to process segment {i + 1}: {e}")
                # Create silent audio segment as fallback
                silence_duration = segment.duration
                silence_samples = int(silence_duration * 22050)  # Default sample rate
                silence_audio = np.zeros(silence_samples)
                
                audio_segment = AudioSegment(
                    audio_data=silence_audio,
                    sample_rate=22050,
                    start=segment.start,
                    end=segment.end,
                    duration=silence_duration
                )
                audio_segments.append(audio_segment)
        
        logger.info("TTS processing completed")
        return audio_segments
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass