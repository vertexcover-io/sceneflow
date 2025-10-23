"""Speech detection using Silero VAD (Voice Activity Detection)."""

from typing import Optional
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from pathlib import Path
import torch
import librosa
from silero_vad import load_silero_vad, get_speech_timestamps


class SpeechDetector:
    """Detects when speech ends in video/audio files using Silero VAD."""

    def __init__(self):
        """Initialize the detector with Silero VAD model."""
        try:
            self.vad_model = load_silero_vad()
        except Exception as e:
            raise RuntimeError(f"Failed to load Silero VAD model: {str(e)}")

    def _load_audio_for_vad(self, file_path: str) -> torch.Tensor:
        """
        Load audio file for VAD processing.

        Args:
            file_path: Path to audio/video file

        Returns:
            Audio as PyTorch tensor at 16kHz sample rate
        """
        # Load audio using librosa (handles both audio and video files)
        # Silero VAD expects 16kHz mono audio
        audio, sr = librosa.load(file_path, sr=16000, mono=True)

        # Convert numpy array to PyTorch tensor
        audio_tensor = torch.from_numpy(audio).float()

        return audio_tensor

    def get_speech_end_time(
        self,
        file_path: str,
        return_confidence: bool = False
    ) -> float | tuple[float, float]:
        """
        Get speech end time using Silero VAD (Voice Activity Detection).

        This method uses deep learning for highly accurate speech/silence detection.

        Args:
            file_path: Path to video or audio file
            return_confidence: If True, return (timestamp, confidence) tuple

        Returns:
            Speech end timestamp in seconds
            If return_confidence=True, returns (timestamp, confidence_score)
        """
        try:
            # Load audio for VAD
            wav = self._load_audio_for_vad(file_path)

            # Get speech timestamps using VAD
            speech_timestamps = get_speech_timestamps(
                wav,
                self.vad_model,
                return_seconds=True
            )

            if not speech_timestamps:
                # No speech detected
                if return_confidence:
                    return 0.0, 0.0
                return 0.0

            # Get the end time of the last speech segment
            last_speech_segment = speech_timestamps[-1]
            vad_end_time = last_speech_segment['end']

            # Calculate confidence based on VAD's detection quality
            # Higher confidence if the last segment is well-defined
            segment_duration = last_speech_segment['end'] - last_speech_segment['start']
            confidence = min(1.0, segment_duration / 0.5)  # 0.5s or longer = full confidence

            if return_confidence:
                return vad_end_time, confidence
            return vad_end_time

        except Exception as e:
            raise RuntimeError(f"VAD speech detection failed: {str(e)}")
