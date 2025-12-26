"""Speech detection using Silero VAD (Voice Activity Detection).

This module provides the SpeechDetector class for detecting when speech
ends in video or audio files using deep learning-based voice activity detection.
"""

import logging
import warnings
from typing import Tuple

import librosa
import asyncio
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

from sceneflow.shared.constants import VAD
from sceneflow.shared.exceptions import VADModelError, AudioLoadError

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", message=".*audioread.*")


class SpeechDetector:
    """
    Detects when speech ends in video/audio files using Silero VAD.

    Uses deep learning-based voice activity detection for highly accurate
    speech/silence classification. The detector operates on audio extracted
    from video files and processes it at 16kHz for optimal VAD performance.

    Attributes:
        vad_model: Loaded Silero VAD model instance

    Example:
        >>> detector = SpeechDetector()
        >>> speech_end = detector.get_speech_end_time("video.mp4")
        >>> print(f"Speech ends at {speech_end:.2f}s")
    """

    def __init__(self):
        """
        Initialize the detector with Silero VAD model.

        Raises:
            VADModelError: If Silero VAD model fails to load
        """
        try:
            self.vad_model = load_silero_vad()
            logger.info("Silero VAD model loaded")
        except Exception as e:
            logger.error("Failed to load Silero VAD model: %s", e)
            raise VADModelError(str(e)) from e

    def _load_audio_for_vad(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file for VAD processing.

        Converts audio to 16kHz mono format required by Silero VAD.
        Handles both audio files and video files (extracts audio track).

        Args:
            file_path: Path to audio/video file

        Returns:
            Tuple of (audio tensor, sample rate)

        Raises:
            AudioLoadError: If audio cannot be loaded from file
        """
        try:
            # Load audio using librosa (handles both audio and video files)
            # Silero VAD expects 16kHz mono audio
            logger.debug("Loading audio from %s at %d Hz", file_path, VAD.TARGET_SAMPLE_RATE)

            audio, sr = librosa.load(file_path, sr=VAD.TARGET_SAMPLE_RATE, mono=True)

            # Convert numpy array to PyTorch tensor
            audio_tensor = torch.from_numpy(audio).float()

            logger.debug("Audio loaded: %.2f seconds, sample_rate=%d", len(audio) / sr, sr)

            return audio_tensor, sr

        except Exception as e:
            logger.error("Failed to load audio from %s: %s", file_path, e)
            raise AudioLoadError(file_path, str(e)) from e

    def _process_vad_results(self, speech_timestamps: list, file_path: str) -> Tuple[float, float]:
        logger.debug("VAD detected %d speech segments", len(speech_timestamps))

        if not speech_timestamps:
            logger.warning("No speech detected in %s", file_path)
            return 0.0, 0.0

        last_speech_segment = speech_timestamps[-1]
        vad_end_time = float(last_speech_segment["end"])

        segment_duration = last_speech_segment["end"] - last_speech_segment["start"]
        confidence = min(1.0, segment_duration / VAD.MIN_SEGMENT_DURATION_FOR_FULL_CONFIDENCE)

        logger.info(
            "VAD analysis complete: %d segments detected, speech ends at %.4fs (confidence: %.2f)",
            len(speech_timestamps),
            vad_end_time,
            confidence,
        )

        return vad_end_time, float(confidence)

    def get_speech_end_time(self, file_path: str) -> Tuple[float, float]:
        try:
            wav, sample_rate = self._load_audio_for_vad(file_path)
            speech_timestamps = get_speech_timestamps(
                wav,
                self.vad_model,
                return_seconds=True,
                sampling_rate=sample_rate,
                threshold=VAD.THRESHOLD,
                neg_threshold=VAD.NEG_THRESHOLD,
                min_silence_duration_ms=VAD.MIN_SILENCE_DURATION_MS,
                speech_pad_ms=VAD.SPEECH_PAD_MS,
                time_resolution=VAD.TIME_RESOLUTION,
            )
            return self._process_vad_results(speech_timestamps, file_path)
        except AudioLoadError:
            raise
        except Exception as e:
            logger.error("VAD speech detection failed: %s", e)
            raise VADModelError(f"Speech detection failed: {str(e)}") from e

    async def get_speech_end_time_async(self, file_path: str) -> Tuple[float, float]:
        try:
            wav, sample_rate = await asyncio.to_thread(self._load_audio_for_vad, file_path)
            speech_timestamps = await asyncio.to_thread(
                get_speech_timestamps,
                wav,
                self.vad_model,
                return_seconds=True,
                sampling_rate=sample_rate,
                threshold=VAD.THRESHOLD,
                neg_threshold=VAD.NEG_THRESHOLD,
                min_silence_duration_ms=VAD.MIN_SILENCE_DURATION_MS,
                speech_pad_ms=VAD.SPEECH_PAD_MS,
                time_resolution=VAD.TIME_RESOLUTION,
            )
            return self._process_vad_results(speech_timestamps, file_path)
        except AudioLoadError:
            raise
        except Exception as e:
            logger.error("VAD speech detection failed: %s", e)
            raise VADModelError(f"Speech detection failed: {str(e)}") from e
