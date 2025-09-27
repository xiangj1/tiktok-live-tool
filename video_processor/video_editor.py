"""
Video editor for extracting and replacing audio in videos
"""
import os
import tempfile
from pathlib import Path
from typing import Optional
import logging
import numpy as np
import subprocess
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
    concatenate_videoclips,
    vfx,
)
import soundfile as sf
from .text_to_speech import AudioSegment

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoEditor:
    """Handle video processing operations"""
    
    def __init__(self):
        """Initialize the video editor"""
        self.temp_dir = Path(tempfile.mkdtemp())
        logger.info(f"Video editor initialized with temp dir: {self.temp_dir}")
    
    def extract_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """
        Extract audio from video file
        
        Args:
            video_path: Path to input video
            output_path: Path for output audio (if None, creates temp file)
            
        Returns:
            Path to extracted audio file
        """
        try:
            logger.info(f"Extracting audio from: {video_path}")
            
            if output_path is None:
                output_path = str(self.temp_dir / "extracted_audio.wav")
            
            # Load video and extract audio
            video = VideoFileClip(video_path)
            audio = video.audio
            
            if audio is None:
                raise ValueError("Video file has no audio track")
            
            # Write audio to file
            audio.write_audiofile(output_path, verbose=False, logger=None)
            
            # Clean up
            audio.close()
            video.close()
            
            logger.info(f"Audio extracted to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to extract audio: {e}")
            raise
    
    def get_video_info(self, video_path: str) -> dict:
        """
        Get video information
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        try:
            video = VideoFileClip(video_path)
            info = {
                "duration": video.duration,
                "fps": video.fps,
                "size": video.size,
                "has_audio": video.audio is not None
            }
            video.close()
            return info
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise
    
    def replace_audio(self, video_path: str, new_audio_path: str,
                     output_path: str, fade_duration: float = 0.1,
                     fill_short_audio: bool = True) -> str:
        """
        Replace audio in video with new audio track
        
        Args:
            video_path: Path to input video
            new_audio_path: Path to new audio file
            output_path: Path for output video
            fade_duration: Duration for fade in/out effects
            
        Returns:
            Path to output video
        """
        try:
            logger.info(f"Replacing audio in video: {video_path}")
            
            # Load video to obtain metadata/duration
            video = VideoFileClip(video_path)
            video_duration = video.duration
            video.close()

            # Prepare audio clip with fades/duration adjustments
            new_audio = AudioFileClip(new_audio_path)
            audio_to_export = new_audio

            duration_delta = new_audio.duration - video_duration
            if abs(duration_delta) > 1e-3:
                if new_audio.duration > video_duration:
                    audio_to_export = new_audio.subclip(0, video_duration)
                elif fill_short_audio:
                    loops_needed = int(video_duration / new_audio.duration) + 1
                    audio_to_export = CompositeAudioClip([new_audio] * loops_needed).subclip(0, video_duration)

            if fade_duration > 0:
                audio_to_export = audio_to_export.audio_fadein(fade_duration).audio_fadeout(fade_duration)

            processed_audio_path = str(self.temp_dir / "processed_audio.wav")
            audio_fps = getattr(audio_to_export, "fps", 44100)
            audio_to_export.write_audiofile(processed_audio_path, fps=audio_fps, verbose=False, logger=None)

            # Close audio clips
            for clip in [audio_to_export, new_audio]:
                try:
                    clip.close()
                except Exception:
                    pass

            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Use ffmpeg to mux new audio while copying video stream to preserve metadata/aspect ratio
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", processed_audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                output_path
            ]

            try:
                subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                stderr_output = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
                logger.error(f"Failed to replace audio via ffmpeg: {stderr_output}")
                raise

            logger.info(f"Video with new audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to replace audio: {e}")
            raise
    
    def mix_audio_tracks(self, original_audio_path: str, new_audio_path: str,
                        output_path: str, new_audio_volume: float = 1.0,
                        original_audio_volume: float = 0.1) -> str:
        """
        Mix original and new audio tracks
        
        Args:
            original_audio_path: Path to original audio
            new_audio_path: Path to new audio
            output_path: Path for mixed audio output
            new_audio_volume: Volume level for new audio (0.0 to 1.0)
            original_audio_volume: Volume level for original audio (0.0 to 1.0)
            
        Returns:
            Path to mixed audio file
        """
        try:
            logger.info("Mixing audio tracks")
            
            # Load audio files
            original_audio = AudioFileClip(original_audio_path)
            new_audio = AudioFileClip(new_audio_path)
            
            # Adjust volumes
            if original_audio_volume != 1.0:
                original_audio = original_audio.volumex(original_audio_volume)
            if new_audio_volume != 1.0:
                new_audio = new_audio.volumex(new_audio_volume)
            
            # Mix the audio tracks
            mixed_audio = CompositeAudioClip([original_audio, new_audio])
            
            # Write mixed audio
            mixed_audio.write_audiofile(output_path, verbose=False, logger=None)
            
            # Clean up
            original_audio.close()
            new_audio.close()
            mixed_audio.close()
            
            logger.info(f"Mixed audio saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to mix audio tracks: {e}")
            raise

    def get_audio_duration(self, audio_path: str) -> float:
        """Return audio duration in seconds"""
        info = sf.info(audio_path)
        return float(info.duration)

    def extend_video_with_reverse(self, video_path: str, target_duration: float,
                                  output_path: str) -> str:
        """Extend video by appending reverse playback until reaching target duration."""
        try:
            video_clip = VideoFileClip(video_path)
            original_duration = video_clip.duration

            if target_duration <= original_duration + 1e-3:
                video_clip.close()
                raise ValueError("Target duration must be greater than original duration for extension")

            remaining = target_duration - original_duration
            reversed_clip = video_clip.fx(vfx.time_mirror)
            segments = [video_clip]

            while remaining > 1e-3:
                segment_duration = min(remaining, original_duration)
                segment = reversed_clip.subclip(0, segment_duration)
                segments.append(segment)
                remaining -= segment_duration

            final_clip = concatenate_videoclips(segments, method="compose")
            final_clip.write_videofile(
                output_path,
                codec="libx264",
                audio=False,
                fps=video_clip.fps,
                verbose=False,
                logger=None
            )

            final_clip.close()
            for clip in segments:
                try:
                    clip.close()
                except Exception:
                    pass
            reversed_clip.close()

            logger.info(
                "Extended video saved to: %s (target duration %.2fs)",
                output_path,
                target_duration
            )
            return output_path

        except Exception as e:
            logger.error(f"Failed to extend video with reverse playback: {e}")
            raise

    def match_audio_loudness(self, reference_audio_path: str,
                             target_audio_path: str,
                             output_path: str,
                             max_gain: float = 6.0,
                             min_rms: float = 1e-4) -> str:
        """
        Adjust target audio loudness to match reference audio RMS level.

        Args:
            reference_audio_path: Path to reference audio (original video audio)
            target_audio_path: Path to target audio (newly generated speech)
            output_path: Path where adjusted audio will be saved
            max_gain: Maximum amplification factor to prevent extreme boosts
            min_rms: Minimum RMS threshold to avoid division by very small numbers

        Returns:
            Path to loudness-matched audio file
        """
        try:
            ref_audio, ref_sr = sf.read(reference_audio_path)
            tgt_audio, tgt_sr = sf.read(target_audio_path)

            def _to_mono(audio: np.ndarray) -> np.ndarray:
                if audio.ndim > 1:
                    return np.mean(audio, axis=1)
                return audio

            ref_mono = _to_mono(ref_audio)
            tgt_mono = _to_mono(tgt_audio)

            ref_rms = np.sqrt(np.mean(np.square(ref_mono))) if len(ref_mono) > 0 else 0.0
            tgt_rms = np.sqrt(np.mean(np.square(tgt_mono))) if len(tgt_mono) > 0 else 0.0

            if ref_rms < min_rms or tgt_rms < min_rms:
                logger.warning("RMS too low for reliable loudness matching, skipping adjustment")
                sf.write(output_path, tgt_audio, tgt_sr)
                return output_path

            gain = ref_rms / tgt_rms
            gain = np.clip(gain, 1.0 / max_gain, max_gain)

            adjusted_audio = tgt_audio * gain
            max_abs = np.max(np.abs(adjusted_audio))
            if max_abs > 1.0:
                adjusted_audio /= max_abs
                logger.debug("Adjusted audio clipped; rescaled to prevent clipping")

            sf.write(output_path, adjusted_audio, tgt_sr)
            logger.info(
                f"Matched audio loudness (gain applied: {gain:.2f}, ref RMS: {ref_rms:.4f}, target RMS: {tgt_rms:.4f})"
            )
            return output_path
        except Exception as e:
            logger.error(f"Failed to match audio loudness: {e}")
            raise
    
    def create_audio_from_segments(self, audio_segments: list, 
                                  total_duration: float,
                                  output_path: str,
                                  sample_rate: int = 22050) -> str:
        """
        Create audio file from audio segments with precise timing
        
        Args:
            audio_segments: List of AudioSegment objects
            total_duration: Total duration of the audio
            output_path: Path for output audio file
            sample_rate: Sample rate for output audio
            
        Returns:
            Path to created audio file
        """
        try:
            logger.info("Creating audio track from segments")
            
            # Create empty audio array
            total_samples = int(total_duration * sample_rate)
            full_audio = np.zeros(total_samples, dtype=np.float32)
            
            # Place each segment in the correct position
            for i, segment in enumerate(audio_segments):
                start_sample = int(segment.start * sample_rate)
                end_sample = int(segment.end * sample_rate)
                
                # Ensure we don't exceed array bounds
                start_sample = max(0, start_sample)
                end_sample = min(total_samples, end_sample)
                
                if start_sample >= end_sample:
                    logger.warning(f"Skipping segment {i}: invalid timing")
                    continue
                
                # Resample segment audio if necessary
                segment_audio = segment.audio_data
                if segment.sample_rate != sample_rate:
                    import librosa
                    segment_audio = librosa.resample(
                        segment_audio,
                        orig_sr=segment.sample_rate,
                        target_sr=sample_rate
                    )
                
                # Fit audio to time slot
                target_length = end_sample - start_sample
                if len(segment_audio) > target_length:
                    # Trim audio
                    segment_audio = segment_audio[:target_length]
                elif len(segment_audio) < target_length:
                    # Pad with silence
                    padding = target_length - len(segment_audio)
                    segment_audio = np.pad(segment_audio, (0, padding), mode='constant')
                
                # Place in full audio array
                full_audio[start_sample:end_sample] = segment_audio.astype(np.float32)
                
                logger.debug(f"Placed segment {i + 1} at {segment.start:.2f}s - {segment.end:.2f}s")
            
            # Save audio file
            sf.write(output_path, full_audio, sample_rate)
            logger.info(f"Audio track saved to: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to create audio from segments: {e}")
            raise
    
    def cleanup(self):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        self.cleanup()