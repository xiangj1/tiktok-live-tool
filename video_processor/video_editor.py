"""
Video editor for extracting and replacing audio in videos
"""
import os
import tempfile
from pathlib import Path
from typing import Optional
import logging
import numpy as np
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip
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
                     output_path: str, fade_duration: float = 0.1) -> str:
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
            
            # Load video (without audio)
            video = VideoFileClip(video_path)
            video_no_audio = video.without_audio()
            
            # Load new audio
            new_audio = AudioFileClip(new_audio_path)
            
            # Adjust audio duration to match video
            if new_audio.duration != video.duration:
                if new_audio.duration > video.duration:
                    # Trim audio
                    new_audio = new_audio.subclip(0, video.duration)
                else:
                    # Loop audio if it's shorter (though this shouldn't happen in our case)
                    loops_needed = int(video.duration / new_audio.duration) + 1
                    new_audio = CompositeAudioClip([new_audio] * loops_needed).subclip(0, video.duration)
            
            # Apply fade effects to avoid clicks/pops
            if fade_duration > 0:
                new_audio = new_audio.audio_fadein(fade_duration).audio_fadeout(fade_duration)
            
            # Set the new audio to the video
            final_video = video_no_audio.set_audio(new_audio)
            
            # Write the final video
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
            # Clean up
            video.close()
            video_no_audio.close()
            new_audio.close()
            final_video.close()
            
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