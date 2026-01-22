"""Speech detection using Silero VAD (Voice Activity Detection)."""

import logging
import warnings
from typing import Tuple

import asyncio
import torch
from silero_vad import load_silero_vad, get_speech_timestamps

from sceneflow.shared.constants import VAD
from sceneflow.shared.exceptions import VADModelError, AudioLoadError

from sceneflow.utils.video import VideoSession

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", category=FutureWarning, module="librosa")
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", message=".*audioread.*")


class SpeechDetector:
    """Detects when speech ends in video/audio files using Silero VAD."""

    def __init__(self):
        try:
            self.vad_model = load_silero_vad()
            logger.info("Silero VAD model loaded")
        except Exception as e:
            logger.error("Failed to load Silero VAD model: %s", e)
            raise VADModelError(str(e)) from e

    def _load_audio_from_session(self, session: "VideoSession") -> Tuple[torch.Tensor, int]:
        """Load audio from VideoSession's cache."""
        try:
            logger.debug("Loading audio from session at %d Hz", VAD.TARGET_SAMPLE_RATE)
            audio, sr = session.get_audio(sr=VAD.TARGET_SAMPLE_RATE)
            audio_tensor = torch.from_numpy(audio).float()
            logger.debug("Audio loaded: %.2f seconds, sample_rate=%d", len(audio) / sr, sr)
            return audio_tensor, sr
        except Exception as e:
            logger.error("Failed to load audio from session: %s", e)
            raise AudioLoadError(session.video_path, str(e)) from e

    def _process_vad_results(self, speech_timestamps: list, video_path: str) -> Tuple[float, float]:
        logger.debug("VAD detected %d speech segments", len(speech_timestamps))

        if not speech_timestamps:
            logger.warning("No speech detected in %s", video_path)
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

    def get_speech_end_time(
        self,
        session: "VideoSession",
        use_energy_refinement: bool = True,
        energy_threshold_db: float = 8.0,
        energy_lookback_frames: int = 20,
    ) -> Tuple[float, float]:
        """Detect when speech ends using VAD and optionally refine using energy analysis."""
        from sceneflow.detection.energy_refiner import refine_speech_end

        try:
            wav, sample_rate = self._load_audio_from_session(session)
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
            vad_end_time, confidence = self._process_vad_results(
                speech_timestamps, session.video_path
            )

            logger.info("Speech end detected at: %.4fs (VAD)", vad_end_time)

            speech_end_time = vad_end_time

            if use_energy_refinement:
                result = refine_speech_end(
                    session=session,
                    vad_timestamp=vad_end_time,
                    threshold_db=energy_threshold_db,
                    lookback_frames=energy_lookback_frames,
                )
                frames_adjusted = result.vad_frame - result.refined_frame
                if frames_adjusted > 0:
                    logger.info(
                        "Speech end refined to: %.4fs (adjusted %d frames backward using energy analysis)",
                        result.refined_timestamp,
                        frames_adjusted,
                    )
                speech_end_time = result.refined_timestamp
            else:
                logger.debug("Energy refinement disabled.")

            return speech_end_time, confidence
        except AudioLoadError:
            raise
        except Exception as e:
            logger.error("VAD speech detection failed: %s", e)
            raise VADModelError(f"Speech detection failed: {str(e)}") from e

    async def get_speech_end_time_async(
        self,
        session: "VideoSession",
        use_energy_refinement: bool = True,
        energy_threshold_db: float = 8.0,
        energy_lookback_frames: int = 20,
    ) -> Tuple[float, float]:
        """Async version of get_speech_end_time."""
        from sceneflow.detection.energy_refiner import refine_speech_end_async

        try:
            wav, sample_rate = await asyncio.to_thread(self._load_audio_from_session, session)
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
            vad_end_time, confidence = self._process_vad_results(
                speech_timestamps, session.video_path
            )

            logger.info("Speech end detected at: %.4fs (VAD)", vad_end_time)

            speech_end_time = vad_end_time

            if use_energy_refinement:
                result = await refine_speech_end_async(
                    session=session,
                    vad_timestamp=vad_end_time,
                    threshold_db=energy_threshold_db,
                    lookback_frames=energy_lookback_frames,
                )
                frames_adjusted = result.vad_frame - result.refined_frame
                if frames_adjusted > 0:
                    logger.info(
                        "Speech end refined to: %.4fs (adjusted %d frames backward using energy analysis)",
                        result.refined_timestamp,
                        frames_adjusted,
                    )
                speech_end_time = result.refined_timestamp
            else:
                logger.debug("Energy refinement disabled.")

            return speech_end_time, confidence
        except AudioLoadError:
            raise
        except Exception as e:
            logger.error("VAD speech detection failed: %s", e)
            raise VADModelError(f"Speech detection failed: {str(e)}") from e
