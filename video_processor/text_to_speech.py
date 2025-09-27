"""
Text-to-Speech processor using OpenAI TTS API
"""
import openai
import os
from typing import List, Optional
import logging
import tempfile
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
from dataclasses import dataclass
from .speech_to_text import SpeechSegment

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
        enable_time_stretch: bool = False,
        stretch_threshold: float = 0.1
    ) -> tuple[np.ndarray, int]:
        """
        Generate speech audio from text
        
        Args:
            text: Text to convert to speech
            target_duration: Target duration in seconds (for speed adjustment)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            logger.debug(f"Generating speech for: {text[:50]}...")
            
            # Generate speech using OpenAI TTS
            response = self.client.audio.speech.create(
                model="tts-1",
                voice=self.voice,
                input=text,
                response_format="mp3"
            )
            
            # Save to temporary file
            temp_file = self.temp_dir / f"tts_{hash(text)}.mp3"
            response.stream_to_file(temp_file)
            
            # Load audio data
            audio_data, sample_rate = librosa.load(temp_file, sr=None)
            
            # Adjust speed if target duration is specified
            if enable_time_stretch and target_duration and target_duration > 0:
                current_duration = len(audio_data) / sample_rate
                speed_ratio = current_duration / target_duration
                
                # Only adjust if the difference is significant (> 10%)
                if abs(speed_ratio - 1.0) > stretch_threshold:
                    logger.debug(
                        f"Applying TTS time stretch: ratio={speed_ratio:.2f}, "
                        f"threshold={stretch_threshold:.2f}"
                    )
                    audio_data = librosa.effects.time_stretch(audio_data, rate=speed_ratio)
            
            # Clean up temp file
            temp_file.unlink(missing_ok=True)
            
            return audio_data, sample_rate
            
        except Exception as e:
            logger.error(f"Failed to generate speech: {e}")
            raise
    
    def process_segments(
        self,
        segments: List[SpeechSegment],
        enable_time_stretch: bool = False,
        stretch_threshold: float = 0.1
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
                    enable_time_stretch=enable_time_stretch,
                    stretch_threshold=stretch_threshold
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
    
    def create_full_audio_track(self, audio_segments: List[AudioSegment], 
                               total_duration: float, 
                               sample_rate: int = 22050) -> np.ndarray:
        """
        Create a complete audio track from segments
        
        Args:
            audio_segments: List of audio segments
            total_duration: Total duration of the output audio
            sample_rate: Target sample rate
            
        Returns:
            Complete audio track as numpy array
        """
        total_samples = int(total_duration * sample_rate)
        full_audio = np.zeros(total_samples)
        
        for segment in audio_segments:
            start_sample = int(segment.start * sample_rate)
            end_sample = int(segment.end * sample_rate)
            
            # Resample audio if necessary
            if segment.sample_rate != sample_rate:
                resampled_audio = librosa.resample(
                    segment.audio_data,
                    orig_sr=segment.sample_rate,
                    target_sr=sample_rate
                )
            else:
                resampled_audio = segment.audio_data
            
            # Adjust length to fit exactly in the time slot
            target_length = end_sample - start_sample
            if len(resampled_audio) != target_length:
                if len(resampled_audio) > target_length:
                    # Trim audio
                    resampled_audio = resampled_audio[:target_length]
                else:
                    # Pad with silence
                    padding = target_length - len(resampled_audio)
                    resampled_audio = np.pad(resampled_audio, (0, padding), mode='constant')
            
            # Place audio in the correct position
            full_audio[start_sample:end_sample] = resampled_audio
        
        return full_audio
    
    def save_audio(self, audio_data: np.ndarray, sample_rate: int, output_path: str):
        """
        Save audio data to file
        
        Args:
            audio_data: Audio data as numpy array
            sample_rate: Sample rate
            output_path: Output file path
        """
        try:
            sf.write(output_path, audio_data, sample_rate)
            logger.info(f"Audio saved to: {output_path}")
        except Exception as e:
            logger.error(f"Failed to save audio: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass