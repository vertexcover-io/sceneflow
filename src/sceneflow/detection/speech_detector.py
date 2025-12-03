"""Speech detection using Silero VAD (Voice Activity Detection).

This module provides the SpeechDetector class for detecting when speech
ends in video or audio files using deep learning-based voice activity detection.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

from sceneflow.shared.constants import VAD
from sceneflow.shared.exceptions import VADModelError, AudioLoadError, NoSpeechDetectedError

logger = logging.getLogger(__name__)


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
        logger.info("Loading Silero VAD model...")
        try:
            self.vad_model = load_silero_vad()
            logger.info("Silero VAD model loaded successfully")
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
            logger.debug(
                "Loading audio from %s at %d Hz",
                file_path,
                VAD.TARGET_SAMPLE_RATE
            )

            audio, sr = librosa.load(
                file_path,
                sr=VAD.TARGET_SAMPLE_RATE,
                mono=True
            )

            # Convert numpy array to PyTorch tensor
            audio_tensor = torch.from_numpy(audio).float()

            logger.debug(
                "Audio loaded: %.2f seconds, sample_rate=%d",
                len(audio) / sr,
                sr
            )

            return audio_tensor, sr

        except Exception as e:
            logger.error("Failed to load audio from %s: %s", file_path, e)
            raise AudioLoadError(file_path, str(e)) from e

    def get_speech_end_time(
        self,
        file_path: str,
        return_confidence: bool = False
    ) -> Union[float, Tuple[float, float]]:
        """
        Get speech end time using Silero VAD (Voice Activity Detection).

        This method uses deep learning for highly accurate speech/silence detection.
        It identifies all speech segments in the audio and returns the end timestamp
        of the last detected speech segment.

        Args:
            file_path: Path to video or audio file
            return_confidence: If True, return (timestamp, confidence) tuple

        Returns:
            Speech end timestamp in seconds.
            If return_confidence=True, returns (timestamp, confidence_score) tuple.
            Returns 0.0 if no speech detected.

        Raises:
            AudioLoadError: If audio cannot be loaded from file
            VADModelError: If VAD processing fails

        Example:
            >>> detector = SpeechDetector()
            >>> # Simple usage
            >>> end_time = detector.get_speech_end_time("video.mp4")
            >>> print(f"Speech ends at {end_time:.2f}s")
            >>>
            >>> # With confidence score
            >>> end_time, confidence = detector.get_speech_end_time("video.mp4", return_confidence=True)
            >>> print(f"Speech ends at {end_time:.2f}s (confidence: {confidence:.2f})")
        """
        logger.info("Detecting speech end time in: %s", file_path)

        try:
            # Load audio for VAD
            wav, sample_rate = self._load_audio_for_vad(file_path)

            # Get speech timestamps using VAD
            logger.debug(
                "Running VAD with threshold=%.2f, neg_threshold=%.2f",
                VAD.THRESHOLD,
                VAD.NEG_THRESHOLD
            )

            speech_timestamps = get_speech_timestamps(
                wav,
                self.vad_model,
                return_seconds=True,
                sampling_rate=sample_rate,
                threshold=VAD.THRESHOLD,
                neg_threshold=VAD.NEG_THRESHOLD,
                min_silence_duration_ms=VAD.MIN_SILENCE_DURATION_MS,
                speech_pad_ms=VAD.SPEECH_PAD_MS,
                time_resolution=VAD.TIME_RESOLUTION
            )

            logger.debug("VAD detected %d speech segments", len(speech_timestamps))

            if not speech_timestamps:
                # No speech detected
                logger.warning("No speech detected in %s", file_path)
                if return_confidence:
                    return 0.0, 0.0
                return 0.0

            # Get the end time of the last speech segment
            last_speech_segment = speech_timestamps[-1]
            vad_end_time = float(last_speech_segment['end'])

            logger.info(
                "Last speech segment: %.3f-%.3fs",
                last_speech_segment['start'],
                vad_end_time
            )

            # Calculate confidence based on VAD's detection quality
            # Higher confidence if the last segment is well-defined
            segment_duration = last_speech_segment['end'] - last_speech_segment['start']
            confidence = min(
                1.0,
                segment_duration / VAD.MIN_SEGMENT_DURATION_FOR_FULL_CONFIDENCE
            )

            logger.info(
                "Speech ends at %.3fs (confidence: %.2f)",
                vad_end_time,
                confidence
            )

            if return_confidence:
                return vad_end_time, float(confidence)
            return vad_end_time

        except AudioLoadError:
            # Re-raise audio loading errors
            raise
        except Exception as e:
            logger.error("VAD speech detection failed: %s", e)
            raise VADModelError(f"Speech detection failed: {str(e)}") from e

    def get_speech_timestamps(
        self,
        file_path: str
    ) -> Tuple[float, List[Dict[str, float]]]:
        """
        Get speech end time and all speech segment timestamps using Silero VAD.

        Args:
            file_path: Path to video or audio file

        Returns:
            Tuple of (speech_end_time, list of speech segments).
            Each segment is a dict with 'start' and 'end' keys in seconds.

        Raises:
            AudioLoadError: If audio cannot be loaded from file
            VADModelError: If VAD processing fails
        """
        logger.info("Detecting speech timestamps in: %s", file_path)

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
                time_resolution=VAD.TIME_RESOLUTION
            )

            logger.debug("VAD detected %d speech segments", len(speech_timestamps))

            if not speech_timestamps:
                logger.warning("No speech detected in %s", file_path)
                return 0.0, []

            segments = [
                {"start": float(seg["start"]), "end": float(seg["end"])}
                for seg in speech_timestamps
            ]

            vad_end_time = segments[-1]["end"]
            logger.info("Speech ends at %.3fs, %d segments detected", vad_end_time, len(segments))

            return vad_end_time, segments

        except AudioLoadError:
            raise
        except Exception as e:
            logger.error("VAD speech detection failed: %s", e)
            raise VADModelError(f"Speech detection failed: {str(e)}") from e
