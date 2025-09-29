"""
Main pipeline for video speech processing
"""
import os
import tempfile
from pathlib import Path
from typing import Optional, List
import logging
from dataclasses import dataclass
import numpy as np
from .speech_to_text import SpeechToTextProcessor, SpeechSegment
from .llm_rephraser import LLMRephraser
from .text_to_speech import TextToSpeechProcessor, AudioSegment, TTSDurationMismatch
from .video_editor import VideoEditor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for video processing pipeline"""
    # Whisper model configuration (OpenAI API model name)
    whisper_model: str = "whisper-1"
    
    # OpenAI API configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-3.5-turbo"
    tts_voice: str = "alloy"
    
    # Processing parameters
    min_segment_duration: float = 2.0
    max_timing_error: float = 1.0
    fade_duration: float = 0.1
    duration_match_tolerance: float = 0.01
    max_duration_overrun_ratio: float = 0.5
    tts_max_attempts: int = 3
    tts_duration_tolerance_ratio: float = 0.12
    tts_retry_backoff_seconds: float = 2.5
    max_rephrase_attempts: int = 2
    rephrase_words_per_second: float = 2.5
    rephrase_length_ratio: float = 0.9
    
    # Output settings
    preserve_original_audio: bool = False
    original_audio_volume: float = 0.1
    new_audio_volume: float = 1.0

class VideoProcessor:
    """Main pipeline for processing videos with speech rephrasing"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the video processor
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Initialize processors
        self.stt_processor = None
        self.llm_rephraser = None
        self.tts_processor = None
        self.video_editor = None
        
        logger.info(f"VideoProcessor initialized with temp dir: {self.temp_dir}")
    
    def _initialize_processors(self):
        """Initialize all processors lazily"""
        if self.stt_processor is None:
            logger.info("Initializing Speech-to-Text processor...")
            self.stt_processor = SpeechToTextProcessor(
                api_key=self.config.openai_api_key,
                model_name=self.config.whisper_model
            )
        
        if self.llm_rephraser is None:
            logger.info("Initializing LLM rephraser...")
            self.llm_rephraser = LLMRephraser(
                api_key=self.config.openai_api_key,
                model=self.config.openai_model
            )
        
        if self.tts_processor is None:
            logger.info("Initializing Text-to-Speech processor...")
            self.tts_processor = TextToSpeechProcessor(
                api_key=self.config.openai_api_key,
                voice=self.config.tts_voice
            )
        
        if self.video_editor is None:
            logger.info("Initializing Video editor...")
            self.video_editor = VideoEditor()
    
    def process_video(self, input_video_path: str, output_video_path: str) -> str:
        """
        Process video with speech rephrasing
        
        Args:
            input_video_path: Path to input video file
            output_video_path: Path for output video file
            
        Returns:
            Path to processed video
        """
        try:
            logger.info(f"Starting video processing: {input_video_path}")
            
            # Initialize processors
            self._initialize_processors()
            
            # Step 1: Extract audio from video
            logger.info("Step 1: Extracting audio from video...")
            extracted_audio_path = self.video_editor.extract_audio(
                input_video_path,
                str(self.temp_dir / "original_audio.wav")
            )
            
            # Get video information
            video_info = self.video_editor.get_video_info(input_video_path)
            total_duration = video_info["duration"]
            logger.info(f"Video duration: {total_duration:.2f} seconds")
            
            # Step 2: Speech-to-text with timestamps
            logger.info("Step 2: Converting speech to text with timestamps...")
            speech_segments = self.stt_processor.process_audio_file(extracted_audio_path)
            
            if not speech_segments:
                raise ValueError("No speech detected in the video")
            
            logger.info(f"Extracted {len(speech_segments)} speech segments")
            self._log_segments(speech_segments, "Original")
            
            # Step 3: Rephrase text using LLM
            logger.info("Step 3: Rephrasing text using LLM...")
            rephrased_segments = self.llm_rephraser.rephrase_segments(speech_segments)
            self._log_segments(rephrased_segments, "Rephrased")
            
            # Step 4: Convert rephrased text to speech
            logger.info("Step 4: Converting rephrased text to speech...")
            audio_segments, adjusted_segments = self._synthesize_segments_with_rephrase(
                rephrased_segments
            )

            rephrased_segments = adjusted_segments
            self._log_segments(rephrased_segments, "Adjusted rephrased")

            # Validate timing accuracy
            self._validate_timing(audio_segments, rephrased_segments)
            
            # Step 5: Create new audio track
            logger.info("Step 5: Creating new audio track...")
            new_audio_path = str(self.temp_dir / "new_audio.wav")
            self.video_editor.create_audio_from_segments(
                audio_segments,
                total_duration,
                new_audio_path
            )

            # Match loudness to original audio
            matched_audio_path = str(self.temp_dir / "new_audio_matched.wav")
            new_audio_path = self.video_editor.match_audio_loudness(
                extracted_audio_path,
                new_audio_path,
                matched_audio_path
            )

            # Align video duration with generated audio duration
            audio_duration = self.video_editor.get_audio_duration(new_audio_path)
            duration_delta = audio_duration - total_duration
            tolerance = self.config.duration_match_tolerance
            fill_short_audio = True
            video_source_path = input_video_path

            if duration_delta > tolerance:
                max_allowed_overrun = total_duration * self.config.max_duration_overrun_ratio
                if duration_delta > max_allowed_overrun:
                    raise ValueError(
                        f"Generated audio ({audio_duration:.2f}s) exceeds original video length "
                        f"({total_duration:.2f}s) by more than the allowed ratio "
                        f"({self.config.max_duration_overrun_ratio:.2f})."
                    )

                extended_video_path = str(self.temp_dir / "extended_video.mp4")
                video_source_path = self.video_editor.extend_video_with_reverse(
                    input_video_path,
                    audio_duration,
                    extended_video_path
                )
                logger.info(
                    "Extended video by %.2fs using reverse playback to match audio duration.",
                    duration_delta
                )
            elif duration_delta < -tolerance:
                fill_short_audio = False
                logger.info(
                    "Trimming video by %.2fs to match shorter generated audio.",
                    abs(duration_delta)
                )
            
            # Step 6: Handle audio mixing if preserving original
            final_audio_path = new_audio_path
            if self.config.preserve_original_audio:
                logger.info("Step 6: Mixing with original audio...")
                mixed_audio_path = str(self.temp_dir / "mixed_audio.wav")
                final_audio_path = self.video_editor.mix_audio_tracks(
                    extracted_audio_path,
                    new_audio_path,
                    mixed_audio_path,
                    self.config.new_audio_volume,
                    self.config.original_audio_volume
                )
            
            # Step 7: Replace audio in video
            logger.info("Step 7: Replacing audio in video...")
            final_video_path = self.video_editor.replace_audio(
                video_source_path,
                final_audio_path,
                output_video_path,
                self.config.fade_duration,
                fill_short_audio=fill_short_audio
            )
            
            logger.info(f"Video processing completed: {final_video_path}")
            return final_video_path
            
        except Exception as e:
            logger.error(f"Video processing failed: {e}")
            raise
        finally:
            # Cleanup
            self.cleanup()
    
    def _log_segments(self, segments: List[SpeechSegment], label: str):
        """Log segment information for debugging"""
        logger.info(f"{label} segments:")
        for i, segment in enumerate(segments[:5]):  # Log first 5 segments
            logger.info(f"  {i+1}: [{segment.start:.2f}s-{segment.end:.2f}s] {segment.text[:100]}...")
        if len(segments) > 5:
            logger.info(f"  ... and {len(segments) - 5} more segments")
    
    def _estimate_max_words(self, duration: float) -> int:
        base_words = duration * self.config.rephrase_words_per_second * self.config.rephrase_length_ratio
        if base_words <= 0:
            return 5
        return max(5, int(round(base_words)))

    def _synthesize_segments_with_rephrase(
        self,
        segments: List[SpeechSegment],
    ) -> tuple[List[AudioSegment], List[SpeechSegment]]:
        audio_segments: List[AudioSegment] = []
        adjusted_segments: List[SpeechSegment] = []

        for idx, segment in enumerate(segments):
            logger.info("Generating TTS for segment %d/%d", idx + 1, len(segments))

            current_text = segment.text
            for rephrase_attempt in range(self.config.max_rephrase_attempts + 1):
                try:
                    audio_data, sample_rate = self.tts_processor.generate_speech(
                        current_text,
                        segment.duration,
                        max_attempts=self.config.tts_max_attempts,
                        duration_tolerance_ratio=self.config.tts_duration_tolerance_ratio,
                        backoff_seconds=self.config.tts_retry_backoff_seconds,
                    )

                    audio_segments.append(
                        AudioSegment(
                            audio_data=audio_data,
                            sample_rate=sample_rate,
                            start=segment.start,
                            end=segment.end,
                            duration=len(audio_data) / sample_rate,
                        )
                    )
                    adjusted_segments.append(
                        SpeechSegment(
                            text=current_text,
                            start=segment.start,
                            end=segment.end,
                            confidence=segment.confidence,
                        )
                    )
                    break

                except TTSDurationMismatch as mismatch:
                    if rephrase_attempt >= self.config.max_rephrase_attempts:
                        logger.warning(
                            "Segment %d remains outside duration tolerance after rephrase attempts; "
                            "using longest available audio.",
                            idx + 1,
                        )

                        audio_segments.append(
                            AudioSegment(
                                audio_data=mismatch.audio_data,
                                sample_rate=mismatch.sample_rate,
                                start=segment.start,
                                end=segment.end,
                                duration=mismatch.actual_duration,
                            )
                        )
                        adjusted_segments.append(
                            SpeechSegment(
                                text=current_text,
                                start=segment.start,
                                end=segment.end,
                                confidence=segment.confidence,
                            )
                        )
                        break

                    max_words = self._estimate_max_words(segment.duration)
                    logger.info(
                        "Rephrasing segment %d to fit within ~%d words",
                        idx + 1,
                        max_words,
                    )
                    current_text = self.llm_rephraser.rephrase_text(
                        current_text,
                        max_words=max_words,
                    )

                except Exception as e:
                    logger.error(
                        "Failed to synthesize segment %d: %s. Inserting silence as fallback.",
                        idx + 1,
                        e,
                    )
                    silence_sr = 22050
                    silence_samples = int(segment.duration * silence_sr)
                    silence_audio = np.zeros(silence_samples, dtype=np.float32)

                    audio_segments.append(
                        AudioSegment(
                            audio_data=silence_audio,
                            sample_rate=silence_sr,
                            start=segment.start,
                            end=segment.end,
                            duration=segment.duration,
                        )
                    )
                    adjusted_segments.append(
                        SpeechSegment(
                            text=current_text,
                            start=segment.start,
                            end=segment.end,
                            confidence=segment.confidence,
                        )
                    )
                    break

            else:
                # This point is reached only if the for-loop completes without break
                logger.warning(
                    "Unexpected TTS loop exit for segment %d; inserting silence.",
                    idx + 1,
                )
                silence_sr = 22050
                silence_samples = int(segment.duration * silence_sr)
                silence_audio = np.zeros(silence_samples, dtype=np.float32)

                audio_segments.append(
                    AudioSegment(
                        audio_data=silence_audio,
                        sample_rate=silence_sr,
                        start=segment.start,
                        end=segment.end,
                        duration=segment.duration,
                    )
                )
                adjusted_segments.append(segment)

        return audio_segments, adjusted_segments

    def _validate_timing(self, audio_segments: List[AudioSegment], 
                        original_segments: List[SpeechSegment]):
        """Validate timing accuracy of generated audio"""
        timing_errors = []
        
        for audio_seg, orig_seg in zip(audio_segments, original_segments):
            target_duration = orig_seg.duration
            actual_duration = audio_seg.duration
            error = abs(actual_duration - target_duration)
            timing_errors.append(error)
            
            if error > self.config.max_timing_error:
                logger.warning(
                    f"Timing error exceeds threshold: {error:.2f}s > {self.config.max_timing_error}s "
                    f"for segment '{orig_seg.text[:50]}...'"
                )
        
        avg_error = sum(timing_errors) / len(timing_errors) if timing_errors else 0
        max_error = max(timing_errors) if timing_errors else 0
        
        logger.info(f"Timing validation - Average error: {avg_error:.3f}s, Max error: {max_error:.3f}s")
        
        if max_error > self.config.max_timing_error:
            logger.warning(f"Some segments exceed maximum timing error of {self.config.max_timing_error}s")
    
    def extract_and_analyze_speech(self, video_path: str) -> List[SpeechSegment]:
        """
        Extract and analyze speech from video (for testing/preview)
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of speech segments
        """
        try:
            self._initialize_processors()
            
            # Extract audio
            audio_path = self.video_editor.extract_audio(
                video_path,
                str(self.temp_dir / "temp_audio.wav")
            )
            
            # Process speech
            segments = self.stt_processor.process_audio_file(audio_path)
            
            return segments
            
        except Exception as e:
            logger.error(f"Speech analysis failed: {e}")
            raise
    
    def preview_rephrasing(self, segments: List[SpeechSegment]) -> List[SpeechSegment]:
        """
        Preview rephrasing without processing video
        
        Args:
            segments: Original speech segments
            
        Returns:
            Rephrased segments
        """
        try:
            self._initialize_processors()
            return self.llm_rephraser.rephrase_segments(segments)
        except Exception as e:
            logger.error(f"Rephrasing preview failed: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            # Cleanup individual processors
            if self.tts_processor:
                self.tts_processor.cleanup()
            if self.video_editor:
                self.video_editor.cleanup()
            
            # Cleanup temp directory
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Cleanup completed")
            
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()