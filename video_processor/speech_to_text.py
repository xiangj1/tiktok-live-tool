"""
Speech-to-Text processor using OpenAI Cloud API
"""
import logging
import openai
import os
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import librosa
import numpy as np
import soundfile as sf

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

    MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB limit from OpenAI API
    TARGET_SAMPLE_RATE = 16_000
    CHUNK_SAFETY_MARGIN = 0.9  # leave headroom under API limit
    BYTES_PER_SAMPLE = 4  # float32 when writing with soundfile

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

    def _prepare_audio_chunks(self, audio_path: str) -> List[Tuple[str, float, bool]]:
        """
        Prepare one or more audio files that satisfy the OpenAI upload size limit.

        Returns a list of tuples: (file_path, start_offset_seconds, is_temporary).
        """
        file_size = os.path.getsize(audio_path)
        if file_size <= self.MAX_UPLOAD_BYTES:
            return [(audio_path, 0.0, False)]

        logger.info(
            "Audio file size %.2f MB exceeds OpenAI limit; resampling to %dHz mono FLAC",
            file_size / (1024 * 1024),
            self.TARGET_SAMPLE_RATE,
        )

        audio_data, sample_rate = sf.read(audio_path, dtype="float32")

        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != self.TARGET_SAMPLE_RATE:
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=self.TARGET_SAMPLE_RATE
            )
            sample_rate = self.TARGET_SAMPLE_RATE

        max_abs = np.max(np.abs(audio_data)) if audio_data.size else 0.0
        if max_abs > 1.0:
            audio_data = audio_data / max_abs

        # First, try compressing the whole audio into a single FLAC file.
        full_temp = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
        full_path = full_temp.name
        full_temp.close()

        sf.write(full_path, audio_data, sample_rate, format="FLAC")
        compressed_size = os.path.getsize(full_path)

        if compressed_size <= self.MAX_UPLOAD_BYTES:
            logger.info(
                "Compressed audio to %.2f MB (within limit); uploading as a single chunk.",
                compressed_size / (1024 * 1024)
            )
            return [(full_path, 0.0, True)]

        # Otherwise, split into multiple chunks.
        os.unlink(full_path)

        bytes_per_second = self.TARGET_SAMPLE_RATE * self.BYTES_PER_SAMPLE
        max_seconds = int(
            (self.MAX_UPLOAD_BYTES * self.CHUNK_SAFETY_MARGIN) / max(bytes_per_second, 1)
        )
        max_seconds = max(max_seconds, 60)  # ensure at least 60 seconds per chunk
        chunk_samples = max_seconds * self.TARGET_SAMPLE_RATE

        chunks: List[Tuple[str, float, bool]] = []
        total_samples = audio_data.shape[0]

        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk = audio_data[start_sample:end_sample]
            if chunk.size == 0:
                continue

            chunk_file = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
            chunk_path = chunk_file.name
            chunk_file.close()

            sf.write(chunk_path, chunk, self.TARGET_SAMPLE_RATE, format="FLAC")
            chunk_size = os.path.getsize(chunk_path)

            if chunk_size > self.MAX_UPLOAD_BYTES:
                os.unlink(chunk_path)
                raise ValueError(
                    "Chunked audio still exceeds OpenAI upload limit. Consider reducing chunk duration."
                )

            start_offset = start_sample / self.TARGET_SAMPLE_RATE
            chunks.append((chunk_path, float(start_offset), True))

        logger.info(
            "Split audio into %d chunks (~%.1f sec each) to satisfy upload limit.",
            len(chunks),
            chunk_samples / self.TARGET_SAMPLE_RATE
        )

        return chunks

    def transcribe_audio(
        self,
        audio_path: str,
        *,
        chunk_retry_attempts: int = 1,
        retry_backoff_seconds: float = 3.0,
    ) -> List[SpeechSegment]:
        """
        Transcribe audio file to speech segments with timestamps using OpenAI API

        Args:
            audio_path: Path to the audio file

        Returns:
            List of SpeechSegment objects with text and timing
        """
        logger.info(f"Transcribing audio via OpenAI API: {audio_path}")
        prepared_chunks = self._prepare_audio_chunks(audio_path)

        segments: List[SpeechSegment] = []

        try:
            for index, (chunk_path, offset, is_temp) in enumerate(prepared_chunks, start=1):
                logger.info(
                    "Uploading audio chunk %d/%d (offset %.2fs)...",
                    index,
                    len(prepared_chunks),
                    offset,
                )

                attempt = 0
                while True:
                    attempt += 1
                    try:
                        with open(chunk_path, "rb") as audio_file:
                            response = self.client.audio.transcriptions.create(
                                model=self.model_name,
                                file=audio_file,
                                response_format="verbose_json"
                            )
                        break
                    except Exception as chunk_error:
                        logger.warning(
                            "Chunk %d/%d failed (attempt %d/%d): %s",
                            index,
                            len(prepared_chunks),
                            attempt,
                            chunk_retry_attempts,
                            chunk_error,
                        )
                        if attempt >= chunk_retry_attempts:
                            raise
                        time.sleep(retry_backoff_seconds)
                if is_temp and os.path.exists(chunk_path):
                    os.unlink(chunk_path)

                for segment in response.segments:
                    if isinstance(segment, dict):
                        text = segment.get("text", "").strip()
                        start = float(segment.get("start", 0.0)) + offset
                        end = float(segment.get("end", start)) + offset
                        confidence = segment.get("confidence")
                        if confidence is None:
                            confidence = segment.get("avg_logprob", 1.0)
                    else:
                        text = getattr(segment, "text", "").strip()
                        start = getattr(segment, "start", 0.0) + offset
                        end = getattr(segment, "end", start) + offset
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